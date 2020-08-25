import fnmatch
import io
import os
import subprocess

import pandas as pd
from tqdm import tqdm


def create_manifest(data_path, output_name, manifest_path, min_duration=None, max_duration=None):
    file_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(data_path)
                  for f in fnmatch.filter(files, '*.wav')]
    file_paths = order_and_prune_files(file_paths, min_duration, max_duration)

    os.makedirs(manifest_path, exist_ok=True)
    manifest_file = manifest_path + output_name
    with io.FileIO(manifest_file, "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample.encode('utf-8'))

    print(f"Manifest created at {manifest_file}")
    return manifest_file


def create_dataframe(data_path, output_name, manifest_path):
    file_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(data_path)
                  for f in fnmatch.filter(files, '*.wav')]
    file_paths = gather_file_durations(file_paths)

    os.makedirs(manifest_path, exist_ok=True)

    data = []
    for wav_path, duration in tqdm(file_paths, total=len(file_paths)):
        transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
        audio_file_path = os.path.abspath(wav_path)
        transcript_file_path = os.path.abspath(transcript_path)
        data.append((duration, audio_file_path, transcript_file_path))

    df = pd.DataFrame(data, columns=["duration", "audio", "transcript"])
    df.to_csv(manifest_path + output_name, index=False)
    print(f"Created pd.DataFrame at {manifest_path + output_name}")


def order_and_prune_files(file_paths, min_duration, max_duration):
    duration_file_paths = gather_file_durations(file_paths)

    if min_duration:
        print(f"Pruning manifest shorter than {min_duration}s")
        n_before = len(duration_file_paths)
        duration_file_paths = [(path, duration) for path, duration in duration_file_paths if
                               min_duration <= duration]
        print(f"Removed {n_before - len(duration_file_paths)} files shorter than {min_duration}s")

    if max_duration:
        print(f"Pruning manifest longer than {max_duration}s")
        n_before = len(duration_file_paths)
        duration_file_paths = [(path, duration) for path, duration in duration_file_paths if
                               duration <= max_duration]
        print(f"Removed {n_before - len(duration_file_paths)} files longer than {max_duration}s")

    def func(element):
        return element[1]

    duration_file_paths.sort(key=func)
    return [x[0] for x in duration_file_paths]  # Remove durations


def gather_file_durations(file_paths):
    """
    Returns: List of Tuple[path, duration]
    """

    print("Collecting audio file durations...")
    duration_file_paths = []
    for path in tqdm(file_paths, total=len(file_paths)):
        duration_file_paths.append(
            (path, float(subprocess.check_output(['soxi -D \"%s\"' % path.strip()], shell=True))))

    return duration_file_paths
