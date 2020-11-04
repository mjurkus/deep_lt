import os
from tempfile import NamedTemporaryFile

import librosa
import numpy as np
import pandas as pd
import sox
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import TimeMasking, FrequencyMasking


def load_audio(path):
    sound, sample_rate = torchaudio.load(path, normalization=True)
    return sound


class SoxAugment(nn.Module):

    def __init__(self, sample_rate=16000, debug: bool = False):
        super(SoxAugment, self).__init__()
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(42)
        self.debug = debug

    def forward(self, x):
        speed = self.rng.uniform(0.8, 1.2)
        pitch = int(self.rng.uniform(-400, 400))
        effects = [
            ['gain', '-n'],  # normalises to 0dB
            ['pitch', f"{pitch}"],  # pitch shift, valid rage -500..500
            ['speed', f'{speed:.5f}'],
            # consider adding echoes only in rare cases
            # ["echos", "0.8", "0.9", f"{int(self.rng.uniform(1, 150))}", "0.4"],
        ]

        if self.debug:
            print(effects)

        x, _ = apply_effects_tensor(x, self.sample_rate, effects, channels_first=True)

        return x


class SpecAugment(nn.Module):

    def __init__(self, rate, freq_mask=15):
        super(SpecAugment, self).__init__()
        self.rng = np.random.RandomState(42)
        self.freq_mask = freq_mask
        self.rate = rate

    def forward(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug(x)

        return x

    def specaug(self, x):
        spec_len = x.shape[2]
        time_mask = int(self.rng.uniform(spec_len * 0.05, spec_len * 0.15))
        seq = nn.Sequential(
            FrequencyMasking(freq_mask_param=self.freq_mask),
            TimeMasking(time_mask_param=time_mask),
        )

        return seq(x)


class LogMelSpec(nn.Module):

    def __init__(self, sample_rate=16000, win_length: float = 0.02, hop_length: float = 0.01):
        super(LogMelSpec, self).__init__()

        win_length = int(sample_rate * 0.02)
        hop_length = int(sample_rate * 0.01)

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=win_length,
            n_mels=161,
        )

    def forward(self, x):
        x = self.transform(x)  # mel spectrogram
        x = np.log(x + 1e-14)  # logrithmic, add small value to avoid inf
        return x


def get_featurizer(sample_rate=16000):
    return LogMelSpec(sample_rate=sample_rate, win_length=0.02, hop_length=0.01)


class NoiseInjection:
    def __init__(
            self,
            path=None,
            sample_rate=16000,
            noise_levels=(0, 0.5)
    ):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = sox.file_info.duration(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data


class SpectrogramDataset(Dataset):
    parameters = {
        "sample_rate": 16000,
        "specaug_rate": 0.3,
        "freq_mask": 15,
        "spec_augment": True,
        "sox_augment": True,
    }

    def __init__(
            self,
            manifest_filepath: str,
            labels: list,
            specaug_rate,
            freq_mask,
            validation: bool = False,
            sample_rate=16000,
            spec_augment: bool = False,
            sox_augment: bool = True
    ):
        self.df = pd.read_csv(manifest_filepath).sample(frac=0.01)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.sample_rate = sample_rate
        self.validation = validation
        sox_augmentation, spec_augmentation = None, None

        if sox_augment and not validation:
            sox_augmentation = SoxAugment(sample_rate=16000)
        if spec_augment and not validation:
            spec_augmentation = SpecAugment(specaug_rate, freq_mask)

        mel_spec = LogMelSpec(sample_rate=sample_rate)

        self.audio_transforms = torch.nn.Sequential(
            *[augment for augment in [sox_augmentation, mel_spec, spec_augmentation] if augment is not None]
        )

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        audio_path, transcript_path = sample.audio, sample.text
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        return spect, transcript

    def parse_audio(self, audio_path):
        waveform = load_audio(audio_path)
        return self.audio_transforms(waveform)  # (channel, feature, time)

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return len(self.df)


def _collate_fn(batch):
    def max_by_sample_len(p):
        return p[0].size(2)

    batch = sorted(batch, key=lambda sample: sample[0].size(2), reverse=True)
    longest_sample = max(batch, key=max_by_sample_len)[0]
    freq_size = longest_sample.size(1)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(2)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0].squeeze(0)
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs, targets, input_percentages, target_sizes


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio
