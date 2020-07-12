import argparse
import os

from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest

parser = argparse.ArgumentParser(description='Processes local dataset and creates manifest.')
parser = add_data_opts(parser)
parser.add_argument("--source-dir", default=None, type=str, help="Directory where dataset is stored")
parser.add_argument("--manifest_name", default='local_manifest', type=str, help="Manifest name")
args = parser.parse_args()


def _preprocess_transcript(phrase):
    return phrase.strip().upper()


def main():
    source_dir = args.source_dir

    if not os.path.exists(source_dir):
        raise NotADirectoryError(f"Directory does not exist: {source_dir}")

    create_manifest(data_path=source_dir, output_name=f"{args.manifest_name}.csv", manifest_path=args.manifest_dir)


if __name__ == "__main__":
    main()
