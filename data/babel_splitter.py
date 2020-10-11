import argparse
import os
import re
from pathlib import Path
from typing import List

from pydub import AudioSegment
from tqdm import tqdm


class Utterance:
    def __init__(self, file: str, start: float, text: str, end: float):
        self.end = end
        self.text = text
        self.start = start
        self.file = file

    def __str__(self):
        return f"{self.start} :: '{self.text}' :: {self.end}"

    @property
    def is_silence(self) -> bool:
        return len(self.text) == 0 or self.text == "<no-speech>" or self.end is None


def run(arg):
    utterances = split_txt_files(Path(arg.source_dir) / "txt", arg.destination_dir)
    split_audio_files(arg.source_dir, arg.destination_dir, utterances)


def split_audio_files(source_dir: str, audio_destination_dir, utterances: List[List[Utterance]]):
    audio_dir = Path(source_dir) / "wav"

    for ul in tqdm(utterances, total=len(utterances)):
        audio_file = str(audio_dir / Path(ul[0].file).stem) + ".wav"
        input_audio = AudioSegment.from_wav(audio_file)
        output_audio_dir = Path(audio_destination_dir) / 'wav' / Path(ul[0].file).stem
        output_audio_dir.mkdir(parents=True, exist_ok=True)

        for idx, u in enumerate(ul):
            try:
                split_audio_by_start_end(
                    input_audio=input_audio,
                    start=u.start * 1000,
                    end=u.end * 1000,
                    output_audio_dir=output_audio_dir,
                    fname=str(idx)
                )
            except Exception as e:
                print(u.file, u)
                raise e


def split_audio_by_start_end(input_audio, start, end, output_audio_dir, fname):
    output = input_audio[start:end]
    output.export(os.path.join(output_audio_dir, fname + ".wav"), format='wav')


def split_txt_files(text_source_dir, text_destination_dir) -> List[List[Utterance]]:
    all_files = [os.path.join(dir_path, f) for dir_path, _, files in os.walk(text_source_dir) for f in files]

    result: List[List[Utterance]] = list()

    for path in tqdm(all_files, total=len(all_files)):
        utterances = parse_single_file(path)
        utterances = list(filter(lambda u: not u.is_silence, utterances))

        for idx, utterance in enumerate(utterances):
            file_dir = Path(text_destination_dir) / 'txt' / Path(utterance.file).stem
            file_dir.mkdir(parents=True, exist_ok=True)

            with open(file_dir / f"{idx}.txt", 'w') as f:
                sanitized = sanitize_text(utterance.text)
                f.write(sanitized)

        if len(utterances) > 0:
            result.append(utterances)

    return result


def parse_single_file(file) -> List[Utterance]:
    with open(file) as f:
        lines = f.readlines()

    result: List[Utterance] = list()

    for x in range(0, len(lines) - 1, 2):
        start, text, end = lines[x:x + 3]
        u = Utterance(
            file=file,
            start=float(re.findall("\\d+.\\d{1,3}", start.replace("\n", ""))[0]),
            text=sanitize_text(text),
            end=float(re.findall("\\d+.\\d{1,3}", end.replace("\n", ""))[0]),
        )

        if len(result) > 0:
            result[-1].end = u.start

        result.append(u)

    return result


sanitize_rules = [
    ("\\(\\(\\)\\)", ""),
    ("\\<\\S+\\>", ""),
    ("\\s{2,}", " "),
    ("[^a-ząčęėįįšųūž\\s]", ""),
    ("^\\s", "")
]


def sanitize_text(text: str) -> str:
    text = text.lower()
    for rule in sanitize_rules:
        text = re.sub(rule[0], rule[1], text)

    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parses and splits BABEL texts')
    parser.add_argument("--source-dir", default=None, type=str, help="Directory where dataset is stored")
    parser.add_argument("--destination-dir", default=None, type=str, help="Directory where outputs should be stored")
    args = parser.parse_args()
    run(args)
