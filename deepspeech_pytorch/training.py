import datetime
import logging
import os
from argparse import ArgumentParser

from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.loader.utils import to_absolute_path

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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger

from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.model import DeepSpeech
from pathlib import Path
import json


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_model_path(checkpoint: str, checkpoint_path: str) -> str:
    if checkpoint:
        return checkpoint

    models_path = to_absolute_path(checkpoint_path)

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

    logger.info(json.dumps({**vars(params)}, sort_keys=True, indent=2))

    with open(to_absolute_path(params.labels_path)) as label_file:
        labels = json.load(label_file)

    decoder = GreedyDecoder(labels)  # Decoder used for validation

    if params.num_classes != len(labels):
        raise ValueError(f"Label count and num_classes do not match. {len(labels)} != {params.num_classes}")

    if not params.load_checkpoint:
        model = DeepSpeech(
            hparams=params,
            decoder=decoder,
        )
    else:
        model = DeepSpeech.load_from_checkpoint(
            checkpoint_path=get_model_path(params.checkpoint, params.checkpoint_path),
            hparam_overrides=params,
            decoder=decoder,
        )

    logger.info("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    comet_ml_experiment_key = os.environ.get('COMET_EXPERIMENT_KEY', None)
    comet_logger = CometLogger(
        save_dir='comet',
        project_name='deep-lt',
        workspace='mjurkus',
        offline=params.comet_offline,
        experiment_key=None if not comet_ml_experiment_key else comet_ml_experiment_key,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping('wer', patience=3, verbose=True)
    ]

    model_checkpoint_callback = ModelCheckpoint(
        filepath='models/{epoch}-{loss:.2f}-{wer:.2f}-{cer:.2f}',
        save_weights_only=True,
        save_top_k=True,
        mode='min',
        monitor='wer',
        verbose=True
    )

    trainer = Trainer(
        logger=comet_logger,
        callbacks=callbacks,
        max_epochs=params.epochs,
        gpus=1,
        fast_dev_run=params.fast_dev_run,
        checkpoint_callback=model_checkpoint_callback,
        gradient_clip_val=400,  # TODO move to config
        precision=16,
        auto_lr_find=True,
        profiler="simple"
    )

    data_module = DeepSpeechDataModule(labels=labels, params=params)

    logger.info("Starting model tuning")
    trainer.tune(model, datamodule=data_module)
    logger.info("Finished tuning.")

    trainer.fit(model, datamodule=data_module)


def add_trainer_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--fast-dev-run", default=False, action='store_true')
    parser.add_argument("--comet-offline", default=False, action='store_true')
    parser.add_argument("--verbose", default=False, action='store_true')
    parser.add_argument(
        "--train-manifest", default="manifests/train_manifest.csv", type=str, metavar="PATH", help="training data"
    )
    parser.add_argument(
        "--val-manifest", default="manifests/val_manifest.csv", type=str, metavar="PATH", help="validation data"
    )
    parser.add_argument(
        "--labels-path", default="labels.json", type=str, metavar="PATH", help="path to label json file"
    )
    parser.add_argument(
        "--num-classes", default=37, type=int, metavar="N", help="class count"
    )
    parser.add_argument(
        "--hidden-size", default=1024, type=int, metavar="N", help="hidden size of RNN layers"
    )
    parser.add_argument(
        "--hidden-layers", default=5, type=int, metavar="N", help="Hidden RNN layer count"
    )
    parser.add_argument(
        "--batch-size", default=12, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-5, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument("--eps", metavar="EPS", type=float, default=1e-8)
    parser.add_argument(
        "--epochs", default=50, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "--num-workers", default=16, type=int, metavar="N", help="number of workers used in data loading"
    )
    parser.add_argument(
        "--window-size", default=.02, type=float, metavar="W", help="Window size for spectrogram generation (seconds)"
    )
    parser.add_argument(
        "--window-stride",
        default=.01,
        type=float,
        metavar="W",
        help="Window stride for spectrogram generation (seconds)"
    )
    parser.add_argument(
        "--spec-aug-rate",
        default=0,
        type=float,
        metavar="N",
        help="spectrogram augmentation rate"
    )
    parser.add_argument(
        "--time-mask",
        default=0,
        type=float,
        metavar="N",
        help="maximal width ratio of time mask",
    )
    parser.add_argument(
        "--freq-mask",
        default=0,
        type=int,
        metavar="SOX",
        help="maximal width of frequency mask",
    )
    parser.add_argument(
        "--sox-aug-rate",
        default=0.3,
        type=float,
        metavar="SOX",
        help="Enable audio augmentation with sox"
    )
    parser.add_argument(
        "--sox-speed-range",
        default=0.2,
        type=float,
        metavar="SOX",
        help="Range to augment audio speed. [1-N..1+N]"
    )
    parser.add_argument(
        "--sox-pitch-range",
        default=0,
        type=int,
        metavar="N",
        help="Range to augment audio pitch [-N..N]"
    )
    parser.add_argument(
        "--load-checkpoint",
        default=False,
        type=bool,
        metavar="B",
        help="Load best checkpoint"
    )
    parser.add_argument(
        "--checkpoint-path",
        default="models",
        type=str,
        metavar="PATH",
        help="where checkpoints should be stored",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="to the specific checkpoint",
    )
    return parser
