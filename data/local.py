import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest, create_dataframe

parser = argparse.ArgumentParser(description='Processes local dataset and creates manifest.')
parser = add_data_opts(parser)
parser.add_argument("--source-dir", default=None, type=str, help="Directory where dataset is stored")
parser.add_argument("--manifest-name", default=None, type=str, help="Manifest name")
parser.add_argument("--split", nargs=3, help="Data split ratios train, val, test")
parser.add_argument("--dataframe", action="store_true", help="Creates manifest dataframe with files and durations")
args = parser.parse_args()


def main():
    source_dir = args.source_dir

    if not os.path.exists(source_dir):
        raise NotADirectoryError(f"Directory does not exist: {source_dir}")

    if args.dataframe:
        print("Creating DataFrame")
        create_dataframe(
            data_path=source_dir,
            output_name=f"{args.manifest_name}.csv",
            manifest_path=args.manifest_dir,
        )
    else:
        print("Creating manifest")
        manifest = create_manifest(
            data_path=source_dir,
            output_name=f"{args.manifest_name}.csv",
            manifest_path=args.manifest_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )

        if args.split is not None:
            train, val, test = [float(a) for a in args.split]
            df = pd.read_csv(manifest)
            manifest_dir = Path(args.manifest_dir)
            print(f"Total size of manifest is {len(df)} records.")
            test_df = None  # make IDE happy
            if test > 0:
                train_df, test_df = train_test_split(df, test_size=test)
                train_df, val_df = train_test_split(train_df, test_size=val)
                test_df.to_csv(manifest_dir / f"test_{args.manifest_name}.csv", index=False)
            else:
                train_df, val_df = train_test_split(df, test_size=val)

            train_df.to_csv(manifest_dir / f"train_{args.manifest_name}.csv", index=False)
            val_df.to_csv(manifest_dir / f"val_{args.manifest_name}.csv", index=False)

            print(f"Train size is {len(train_df)}")
            print(f"Val size is {len(val_df)}")
            print(f"Test size is {0 if test_df is None else len(test_df)}")


if __name__ == "__main__":
    main()
