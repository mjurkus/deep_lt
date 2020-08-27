from pathlib import Path

from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.model import DeepSpeech
import json
import os

def load_model(
        device,
        model_path: str,  # TODO use path
):
    with open('labels.json') as label_file:
        labels = json.load(label_file)

    hparams = {
        "model": {
            "hidden_size": 1024,
            "hidden_layers": 5,
        },
        "audio_conf": {
            "sample_rate": 16000,
            "window_size": .02,
            "window_stride": .01,
        },
        "num_classes": len(labels)
    }

    print(hparams['model'])

    model = DeepSpeech.load_from_checkpoint(
        checkpoint_path=to_absolute_path('models/epoch_epoch=12-val_loss=457.15-wer=3.13-cer=0.44.ckpt'),
        hparams=hparams,
        decoder=None,
    )
    model.to(device)
    model.eval()
    return model


def load_decoder(decoder_type,
                 labels,
                 lm_path,
                 alpha,
                 beta,
                 cutoff_top_n,
                 cutoff_prob,
                 beam_width,
                 lm_workers):
    if decoder_type == "beam":
        from deepspeech_pytorch.decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels=labels,
                                 lm_path=lm_path,
                                 alpha=alpha,
                                 beta=beta,
                                 cutoff_top_n=cutoff_top_n,
                                 cutoff_prob=cutoff_prob,
                                 beam_width=beam_width,
                                 num_processes=lm_workers)
    else:
        decoder = GreedyDecoder(labels=labels)
    return decoder


def remove_parallel_wrapper(model):
    """
    Return the model or extract the model out of the parallel wrapper
    :param model: The training model
    :return: The model without parallel wrapper
    """
    # Take care of distributed/data-parallel wrapper
    model_no_wrapper = model.module if hasattr(model, "module") else model
    return model_no_wrapper


def to_absolute_path(path: str) -> str:
    path = Path(path)

    if path.is_absolute():
        return path

    base = Path(os.getcwd())
    return str(base / path)
