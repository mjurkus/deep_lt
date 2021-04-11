import logging
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import TimeMasking, FrequencyMasking

logger = logging.getLogger(__name__)


def load_audio(path):
    sound, sample_rate = torchaudio.load(path, normalization=True)
    return sound


class SoxAugment(nn.Module):
    """
        effects = [
            ['gain', '-n'],  # normalises to 0dB
            ['pitch', pitch],  # pitch shift, valid rage -500..500
            ['speed', speed], # 0.8..1.2
            # ["echos", "0.8", "0.9", f"{int(self.rng.uniform(1, 150))}", "0.4"],
        ]
    """

    def __init__(self, rate, sample_rate=16000, speed_range: float = 0.2, pitch_range: int = 400):
        super(SoxAugment, self).__init__()
        self.rate = rate
        self.sample_rate = sample_rate
        self.speed_range = speed_range
        self.pitch_range = pitch_range
        self.rng = np.random.RandomState(42)

    def forward(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.augment(x)

        return x

    def augment(self, x):
        speed = self.rng.uniform(1 - self.speed_range, 1 + self.speed_range)
        pitch = int(self.rng.uniform(-self.pitch_range, self.pitch_range))
        effects = [
            ['gain', '-n'],
            ['pitch', f"{pitch}"],
            ['speed', f'{speed:.2f}'],
        ]

        x, _ = apply_effects_tensor(x, self.sample_rate, effects, channels_first=True)

        return x


class SpecAugment(nn.Module):

    def __init__(self, rate, freq_mask=15, time_mask=0.1):
        super(SpecAugment, self).__init__()
        self.rng = np.random.RandomState(42)
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.rate = rate

    def forward(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug(x)

        return x

    def specaug(self, x):
        seq = nn.Sequential(
            FrequencyMasking(freq_mask_param=self.freq_mask),
            TimeMasking(time_mask_param=x.shape[2] * self.time_mask),
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


class SpectrogramDataset(Dataset):
    def __init__(
            self,
            manifest_filepath: str,
            labels: List[str],
            spec_aug_rate=0,
            freq_mask=15,
            time_mask=0.1,
            sox_aug_rate=0,
            sox_speed_range=0,
            sox_pitch_range=0,
            validation: bool = False,
            sample_rate=16000,
            **kwargs,
    ):
        self.df = pd.read_csv(manifest_filepath)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.sample_rate = sample_rate
        self.validation = validation
        sox_augmentation, spec_augmentation = None, None

        if sox_aug_rate and not validation:
            sox_augmentation = SoxAugment(rate=sox_aug_rate, sample_rate=16000)
            logger.info(f"Enabled SOX augmentation at rate of {sox_aug_rate}. "
                        f"Speed range: {sox_speed_range}; Pitch range: {sox_pitch_range};")
        else:
            logger.info(f"SOX augmentation disabled. Validation: {validation}")

        if spec_aug_rate and not validation:
            spec_augmentation = SpecAugment(spec_aug_rate, freq_mask=freq_mask, time_mask=time_mask)
            logger.info(f"Enabled spectrogram augmentation at rate of {spec_aug_rate}. "
                        f"Time mask: {time_mask}, Freq mask: {freq_mask}")
        else:
            logger.info(f"Spectrogram augmentation disabled. Validation: {validation}")

        mel_spec = LogMelSpec(sample_rate=sample_rate)

        self.audio_transforms = torch.nn.Sequential(
            *[augment for augment in [sox_augmentation, mel_spec, spec_augmentation] if augment is not None]
        )

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        audio_path, transcript = sample.path, sample.text
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript)
        return spect, transcript

    def parse_audio(self, audio_path):
        waveform = load_audio(audio_path)
        return self.audio_transforms(waveform)  # (channel, feature, time)

    def parse_transcript(self, transcript):
        transcript = transcript.replace('\n', '')
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
