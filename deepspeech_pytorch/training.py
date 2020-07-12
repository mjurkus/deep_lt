import json

import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger

from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.model import DeepSpeech, supported_rnns, DeepSpeechModule


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(cfg):
    # Set seeds for determinism
    pl.trainer.seed_everything(cfg.training.seed)

    print(cfg)

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)

    model = DeepSpeech(
        rnn_hidden_size=cfg.model.hidden_size,
        nb_layers=cfg.model.hidden_layers,
        labels=labels,
        rnn_type=supported_rnns[cfg.model.rnn_type.value],
        audio_conf=cfg.data.spect,
        bidirectional=True
    )
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    # Data setup
    decoder = GreedyDecoder(model.labels)  # Decoder used for validation
    train_dataset = SpectrogramDataset(
        audio_conf=model.audio_conf,
        manifest_filepath=to_absolute_path(cfg.data.train_manifest),
        labels=model.labels,
        normalize=True,
        augmentation_conf=cfg.data.augmentation,
    )

    val_dataset = SpectrogramDataset(
        audio_conf=model.audio_conf,
        manifest_filepath=to_absolute_path(cfg.data.val_manifest),
        labels=model.labels,
        normalize=True,
    )

    train_loader = AudioDataLoader(
        dataset=train_dataset,
        num_workers=cfg.data.num_workers,
        batch_size=cfg.data.batch_size,
        shuffle=True,
    )

    val_loader = AudioDataLoader(
        dataset=val_dataset,
        num_workers=cfg.data.num_workers,
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )

    module = DeepSpeechModule(
        model=model,
        decoder=decoder,
        cfg=cfg
    )

    comet_logger = CometLogger(
        api_key=cfg.comet.api_key,
        project_name=cfg.comet.project_name,
        workspace=cfg.comet.workspace,
        disabled=cfg.comet.disabled or len(cfg.comet.api_key) == 0
    )

    callbacks = [
        LearningRateLogger(),
    ]

    model_checkpoint_callback = ModelCheckpoint(
        filepath='models/epoch_{epoch}-{val_loss:.2f}-{wer:.2f}-{cer:.2f}',
        save_weights_only=True,
        save_top_k=True,
        mode='min',
        monitor='wer',
        verbose=True
    )

    early_stopping = EarlyStopping('val_loss', patience=3, verbose=True)

    trainer = Trainer(
        logger=comet_logger,
        callbacks=callbacks,
        max_epochs=cfg.training.epochs,
        gpus=1,
        fast_dev_run=cfg.training.fast_dev_run,
        early_stop_callback=early_stopping,
        checkpoint_callback=model_checkpoint_callback,
        gradient_clip_val=1.0
    )

    trainer.fit(module, train_dataloader=train_loader, val_dataloaders=val_loader)
