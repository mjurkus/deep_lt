import json
import os
from pathlib import Path

from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.model import DeepSpeech


def load_model(
        device,
        model_path: str,
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
            "window": "hamming",
        },
        "num_classes": len(labels)
    }

    model = DeepSpeech.load_from_checkpoint(
        checkpoint_path=to_absolute_path(model_path),
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


def to_absolute_path(path: str) -> str:
    path = Path(path)

    if path.is_absolute():
        return path

    base = Path(os.getcwd())
    return str(base / path)
