import torch
from pesq import pesq
from joblib import Parallel, delayed
import numpy as np
import torch.nn as nn


def power_compress(x):
    # x[b, f_size, num_f, 2]
    real = x[..., 0]    # [ b, f_size, num_f]
    imag = x[..., 1]    # [ b, f_size, num_f]
    spec = torch.complex(real, imag)    # [ b, f_size, num_f]
    mag = torch.abs(spec)    # [ b, f_size, num_f]
    phase = torch.angle(spec)    # [ b, f_size, num_f]
    mag = mag**0.3    # [ b, f_size, num_f]
    real_compress = mag * torch.cos(phase)    # [ b, f_size, num_f]
    imag_compress = mag * torch.sin(phase)    # [ b, f_size, num_f]
    return torch.stack([real_compress, imag_compress], 1)    # [ b, 2, f_size, num_f]


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


def get_spec(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to('cuda')


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
