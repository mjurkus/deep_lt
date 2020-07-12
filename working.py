import pandas as pd
from sklearn.model_selection import train_test_split

ds = pd.read_csv('manifests/liepa_manifest.csv', header=None, names=['colA', 'colB'])

train_ds, val_ds = train_test_split(ds, test_size=0.2)

train_ds.to_csv('manifests/train_manifest.csv', index=False, header=None)
val_ds.to_csv('manifests/val_manifest.csv', index=False, header=None)
