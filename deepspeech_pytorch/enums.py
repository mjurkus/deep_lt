from enum import Enum


class DistributedBackend(Enum):
    gloo = 'gloo'
    mpi = 'mpi'
    nccl = 'nccl'


class SpectrogramWindow(Enum):
    hamming = 'hamming'
    hann = 'hann'
    blackman = 'blackman'
    bartlett = 'bartlett'
