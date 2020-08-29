import datetime
import logging
import os
from argparse import ArgumentParser

date_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
logging.basicConfig(
    filename=f'logs/training_{date_tag}.log',
    level=os.environ.get('LOG_LEVEL', 'INFO'),
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger

from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.model import DeepSpeech
from pathlib import Path
import yaml
import json


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_model_path(checkpoints: dict) -> str:
    if checkpoints['checkpoint']:
        return checkpoints['checkpoint']

    models_path = to_absolute_path(checkpoints['checkpoint_path'])

    best_wer = 100.1
    best_model_path = None
    for model in os.listdir(models_path):
        model = Path(model)
        for part in model.name.split("-"):
            if part.startswith("wer"):
                wer = float(part.split("=")[1])
                if best_wer > wer:
                    best_wer = wer
                    best_model_path = model

    if best_model_path is None:
        raise ValueError("Best model not found")

    logger.info(f"Best model found {best_model_path}")

    return str(to_absolute_path(str(models_path / best_model_path)))


def train(params):
    # Set seeds for determinism
    pl.trainer.seed_everything(42)

    with open(to_absolute_path('hparams.yaml')) as f:
        hparams = yaml.safe_load(f)

    hparams: dict = {**hparams, **vars(params)}

    logger.info(json.dumps(hparams, sort_keys=True, indent=2))

    with open(to_absolute_path(hparams['labels_path'])) as label_file:
        labels = json.load(label_file)

    decoder = GreedyDecoder(labels)  # Decoder used for validation

    hparams['num_classes'] = len(labels)

    if not hparams['checkpoint']['enabled']:
        model = DeepSpeech(
            hparams=hparams,
            decoder=decoder,
        )
    else:
        model = DeepSpeech.load_from_checkpoint(
            checkpoint_path=get_model_path(hparams['checkpoint']),
            hparams_file='hparams.yaml',
            hparam_overrides=hparams,
            decoder=decoder,
        )

    logger.info("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    # Data setup
    train_dataset = SpectrogramDataset(
        audio_conf=model.audio_conf,
        manifest_filepath=to_absolute_path(hparams['train_manifest']),
        labels=labels,
        normalize=True,
        augmentation_conf=hparams['augment_config'],
    )

    val_dataset = SpectrogramDataset(
        audio_conf=model.audio_conf,
        manifest_filepath=to_absolute_path(hparams['train_manifest']),
        labels=labels,
        normalize=True,
    )

    train_loader = AudioDataLoader(
        dataset=train_dataset,
        num_workers=hparams['num_workers'],
        batch_size=hparams['batch_size'],
        shuffle=True,
    )

    val_loader = AudioDataLoader(
        dataset=val_dataset,
        num_workers=hparams['num_workers'],
        batch_size=hparams['batch_size'],
        shuffle=False,
    )

    comet_ml_experiment_key = os.environ.get('COMET_EXPERIMENT_KEY', None)
    comet_logger = CometLogger(
        save_dir='comet',
        project_name='deep-lt',
        workspace='mjurkus',
        offline=params.comet_offline,
        experiment_key=None if not comet_ml_experiment_key else comet_ml_experiment_key,
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

    early_stopping = EarlyStopping('val_loss', patience=5, verbose=True)

    trainer = Trainer(
        logger=comet_logger,
        callbacks=callbacks,
        max_epochs=hparams['epochs'],
        gpus=1,
        fast_dev_run=hparams['fast_dev_run'],
        early_stop_callback=early_stopping,
        checkpoint_callback=model_checkpoint_callback,
        precision=16
    )

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


def to_absolute_path(path: str) -> str:
    path = Path(path)

    if path.is_absolute():
        return path

    base = Path(os.getcwd())
    return str(base / path)


def add_trainer_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--fast_dev_run", action='store_true')
    parser.add_argument("--comet_offline", default=False, action='store_true')
    parser.add_argument("--verbose", default=False, action='store_true')

    return parser
