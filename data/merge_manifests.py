import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Processes local dataset and creates manifest.')
parser.add_argument("--source-dir", default=None, type=str, help="Directory where dataset is stored")
parser.add_argument("--split", nargs=3, help="Data split ratios train, val, test")
parser.add_argument("--manifest_name", default='manifest', type=str, help="Manifest name")
parser.add_argument('--manifest-dir', default='./manifests/', type=str, help='Output directory for manifests')
args = parser.parse_args()


def main():
    source_dir = args.source_dir

    if not os.path.exists(source_dir):
        raise NotADirectoryError(f"Directory does not exist: {source_dir}")

    manifests = [f for f in os.listdir(source_dir) if f.endswith("csv")]

    print(manifests)

    manifest = Path(source_dir) / 'merged_manifest.csv'

    with open(manifest, "w") as manifest_file:
        for m in manifests:
            with open(Path(source_dir) / m) as f:
                manifest_file.writelines(f.readlines())

    if args.split is not None:
        train, val, test = [float(a) for a in args.split]
        df = pd.read_csv(manifest)
        print(f"Total size of manifest is {len(df)} records.")
        test_df = None  # make IDE happy
        if test > 0:
            train_df, test_df = train_test_split(df, test_size=test)
            train_df, val_df = train_test_split(train_df, test_size=val)
            test_df.to_csv(args.manifest_dir + f"test_{args.manifest_name}.csv", index=False, header=None)
        else:
            train_df, val_df = train_test_split(df, test_size=val)

        train_df.to_csv(args.manifest_dir + f"train_{args.manifest_name}.csv", index=False, header=None)
        val_df.to_csv(args.manifest_dir + f"val_{args.manifest_name}.csv", index=False, header=None)

        print(f"Train size is {len(train_df)}")
        print(f"Val size is {len(val_df)}")
        print(f"Train size is {0 if test_df is None else len(test_df)}")


if __name__ == "__main__":
    main()
