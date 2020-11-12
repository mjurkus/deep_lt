import pytorch_lightning as pl

from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.loader.utils import to_absolute_path


class DeepSpeechDataModule(pl.LightningDataModule):

    def __init__(
            self,
            labels,
            params,
    ):
        super().__init__()

        self.train_path = to_absolute_path(params.train_manifest)
        self.val_path = to_absolute_path(params.val_manifest)
        self.labels = labels
        self.params = params

    def train_dataloader(self):
        train_dataset = SpectrogramDataset(
            manifest_filepath=self.train_path,
            labels=self.labels,
            **vars(self.params)
        )

        train_loader = AudioDataLoader(
            dataset=train_dataset,
            num_workers=self.params.num_workers,
            batch_size=self.params.batch_size,
            shuffle=True,
            pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = SpectrogramDataset(
            manifest_filepath=self.val_path,
            labels=self.labels,
            validation=True,
        )
        val_loader = AudioDataLoader(
            dataset=val_dataset,
            num_workers=self.params.num_workers,
            batch_size=self.params.batch_size,
            shuffle=False,
            pin_memory=True
        )
        return val_loader
