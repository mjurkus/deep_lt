from dataclasses import dataclass

from deepspeech_pytorch.enums import DistributedBackend, SpectrogramWindow, RNNType

defaults = [
    {"model": "bidirectional"},
]


@dataclass
class TrainingConfig:
    no_cuda: bool = False  # Enable CPU only training
    finetune: bool = False  # Fine-tune the model from checkpoint "continue_from"
    seed: int = 123456  # Seed for generators
    dist_backend: DistributedBackend = DistributedBackend.nccl  # If using distribution, the backend to be used
    epochs: int = 70  # Number of Training Epochs
    fast_dev_run: bool = True


@dataclass
class CheckpointConfig:
    enabled: bool = True
    checkpoint_path: str = 'models/'


@dataclass
class SpectConfig:
    sample_rate: int = 16000  # The sample rate for the data/model features
    window_size: float = .02  # Window size for spectrogram generation (seconds)
    window_stride: float = .01  # Window stride for spectrogram generation (seconds)
    window: SpectrogramWindow = SpectrogramWindow.hamming  # Window type for spectrogram generation


@dataclass
class AugmentationConfig:
    speed_volume_perturb: bool = False  # Use random tempo and gain perturbations.
    spec_augment: bool = False  # Use simple spectral augmentation on mel spectograms.
    noise_dir: str = ''  # Directory to inject noise into audio. If default, noise Inject not added
    noise_prob: float = 0.4  # Probability of noise being added per sample
    noise_min: float = 0.0  # Minimum noise level to sample from. (1.0 means all noise, not original signal)
    noise_max: float = 0.5  # Maximum noise levels to sample from. Maximum 1.0


@dataclass
class DataConfig:
    train_manifest: str = 'manifests/train_manifest.csv'
    val_manifest: str = 'manifests/val_manifest.csv'
    batch_size: int = 20  # Batch size for training
    num_workers: int = 4  # Number of workers used in data-loading
    labels_path: str = 'labels.json'  # Contains tokens for model output
    spect: SpectConfig = SpectConfig()
    augmentation: AugmentationConfig = AugmentationConfig()


@dataclass
class BiDirectionalConfig:
    rnn_type: RNNType = RNNType.lstm  # Type of RNN to use in model
    hidden_size: int = 1024  # Hidden size of RNN Layer
    hidden_layers: int = 5  # Number of RNN layers


@dataclass
class UniDirectionalConfig(BiDirectionalConfig):
    lookahead_context: int = 20  # The lookahead context for convolution after RNN layers


@dataclass
class OptimConfig:
    learning_rate: float = 3e-4  # Initial Learning Rate
    weight_decay: float = 1e-5  # Initial Weight Decay


@dataclass
class AdamConfig(OptimConfig):
    eps: float = 1e-8  # Adam eps
    betas: tuple = (0.9, 0.999)  # Adam betas


@dataclass
class ApexConfig:
    opt_level: str = 'O1'  # Apex optimization level, check https://nvidia.github.io/apex/amp.html for more information
    loss_scale: int = 1  # Loss scaling used by Apex. Default is 1 due to warp-ctc not supporting scaling of gradients


@dataclass
class CometMLConfig:
    api_key: str = ''
    project_name: str = "deep-lt"
    workspace: str = "mjurkus"
    disabled: bool = True
    experiment_key: str = ''


@dataclass
class DeepSpeechConfig:
    optim: AdamConfig = AdamConfig()
    model: BiDirectionalConfig = BiDirectionalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    apex: ApexConfig = ApexConfig()
    comet: CometMLConfig = CometMLConfig()
