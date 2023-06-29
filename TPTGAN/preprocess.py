# import librosa
import os

import numpy as np
import scipy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToTensor(object):
    r"""Convert ndarrays in sample to Tensors."""

    def __call__(self, x):
        return torch.from_numpy(x).float()


class SignalToFrames(object):
    r"""Chunks a signal into frames
         required input shape is [1, 1, -1]
         input params:    (frame_size: window_size,  frame_shift: overlap(samples))
         output:   [1, 1, num_frames, frame_size]
    """

    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        # frame_size = self.frame_size
        # frame_shift = self.frame_shift
        sig_len = in_sig.shape[-1]
        nframes = math.ceil((sig_len - self.frame_size) / self.frame_shift + 1)
        # nframes = (sig_len // (self.frame_size - self.frame_shift))
        a = np.zeros(list(in_sig.shape[:-1]) + [nframes, self.frame_size])
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a


class TorchSignalToFrames(object):
    """
    it is for torch tensor
    """

    def __init__(self, frame_size=512, frame_shift=256):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        nframes = math.ceil((sig_len - self.frame_size) / self.frame_shift + 1)
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a


class OLA:
    r"""Performs overlap-and-add
        required input is ndarray
        performs frames into signal
    """

    def __init__(self, frame_shift=256):
        self.frame_shift = frame_shift

    def __call__(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = np.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype)
        ones = np.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class TorchOLA(nn.Module):
    r"""Overlap and add on gpu using torch tensor
        required input is tensor
        perform frames into signal
        used in the output of network
    """

    # Expects signal at last dimension
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift

    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device,
                          requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class STFT:
    r"""Computes STFT of a signal
    input is ndarray
    required input shape is [1, 1, -1]
    """

    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.win = scipy.hamming(frame_size)
        # self.win = np.hamming(frame_size)
        self.get_frames = SignalToFrames(self.frame_size, self.frame_shift)

    def __call__(self, signal):
        frames = self.get_frames(signal)
        frames = frames * self.win
        feature = np.fft.fft(frames)[..., 0:(self.frame_size // 2 + 1)]
        # feature = np.fft.fft(frames)
        feat_R = np.real(feature)
        feat_I = np.imag(feature)
        feature = np.stack([feat_R, feat_I], axis=0)
        # 在此将feature变为[2,T,F]
        feature = np.squeeze(feature, axis=(1, 2))
        return feature


class ISTFT:
    r"""Computes inverse STFT
    input:ndarray[B, 2, N, F]
    output:tensor[]
    """

    # includes overlap-and-add
    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.win = np.hamming(frame_size)
        self.ola = OLA(self.frame_shift)

    def __call__(self, stft):
        R = stft[0:1, ...]
        I = stft[1:2, ...]
        cstft = R + 1j * I
        fullFFT = np.concatenate((cstft, np.conj(cstft[..., -2:0:-1])), axis=-1)
        T = np.fft.ifft(fullFFT)
        T = np.real(T)
        T = T / self.win
        signal = self.ola(T)
        # return signal.astype(np.float32)
        return torch.tensor(signal)


class TorchISTFT:
    r"""Computes inverse STFT"""

    # includes overlap-and-add
    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.win = torch.hamming_window(frame_size).cuda()
        self.ola = TorchOLA(self.frame_shift)

    def __call__(self, stft):
        R = stft[0:1, ...]
        I = stft[1:2, ...]
        cstft = R + 1j * I
        # fullFFT = torch.cat((cstft, torch.conj(cstft[..., -2:0:-1])), dim=-1)
        fullFFT = torch.cat((cstft, torch.conj(torch.flip(cstft, dims=[3])[..., :-2])), dim=-1)
        T = torch.fft.ifft(fullFFT)
        T = torch.real(T)
        T = T / self.win
        signal = self.ola(T)
        return signal


'''
# test stft and istft
path = os.path.join(os.getcwd(), 'IC0936W0001_babble.wav')
wav, sr = librosa.load(path, sr=16000)
wav = np.reshape(wav, [1, 1, -1])
print(wav.shape)

get_sfft = STFT(frame_size=512, frame_shift=256)
returned = get_sfft(wav)
get_istft = ISTFT(frame_size=512, frame_shift=256)
returned1 = get_istft(returned)
print(returned1.shape)
'''

'''
# test the overlap function 
path = os.path.join(os.getcwd(), 'IC0936W0001_babble.wav')
wav, sr = librosa.load(path, sr=16000)
wav = np.reshape(wav, [1, 1, -1])
print(wav.shape)

window_size = 512
hop = int(window_size * 0.5)
get_frames = SignalToFrames(frame_size=window_size, frame_shift=hop)
returned = get_frames(wav)
print(returned.shape)

ola = OLA(hop)
returned1 = ola(returned)
print(returned1.shape)
'''


class SliceSig:
    '''
    input 已经是读到的audio numpy值 -- ndarray
    返回的是 frame list
    input: frame_size, hop
    '''

    def __init__(self, frame_size, hop):
        self.frame_size = frame_size
        self.frame_shift = hop

    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        # num_frames = math.ceil((sig_len - self.frame_shift) / (self.frame_size - self.frame_shift))
        # exp_len = num_frames * self.frame_size - (num_frames - 1) * self.frame_shift
        # pad_0 = np.zeros(exp_len - sig_len, dtype='float32')
        # pad_sig = np.concatenate((in_sig, pad_0))
        slices = []
        for end_idx in range(self.frame_size, sig_len, self.frame_size - self.frame_shift):
            start_idx = end_idx - self.frame_size
            slice_sig = in_sig[start_idx:end_idx]
            slices.append(slice_sig)
            final_idx = end_idx
        slices.append(in_sig[final_idx:])

        return slices


class ISTFTtensor(torch.nn.Module):
    def __init__(self, filter_length=512, hop_length=103, window='hamming', center=True):
        super(ISTFTtensor, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.center = center

        win_cof = scipy.signal.get_window(window, filter_length)
        self.inv_win = self.inverse_stft_window(win_cof, hop_length)

        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        inverse_basis = torch.FloatTensor(self.inv_win * \
                                          np.linalg.pinv(fourier_basis).T[:, None, :])

        self.register_buffer('inverse_basis', inverse_basis.float())

    # Use equation 8 from Griffin, Lim.
    # Paper: "Signal Estimation from Modified Short-Time Fourier Transform"
    # Reference implementation: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/signal/spectral_ops.py
    # librosa use equation 6 from paper: https://github.com/librosa/librosa/blob/0dcd53f462db124ed3f54edf2334f28738d2ecc6/librosa/core/spectrum.py#L302-L311
    def inverse_stft_window(self, window, hop_length):
        window_length = len(window)
        denom = window ** 2
        overlaps = -(-window_length // hop_length)  # Ceiling division.
        denom = np.pad(denom, (0, overlaps * hop_length - window_length), 'constant')
        denom = np.reshape(denom, (overlaps, hop_length)).sum(0)
        denom = np.tile(denom, (overlaps, 1)).reshape(overlaps * hop_length)
        return window / denom[:window_length]

    def forward(self, real_imag_part, length=None):
        # Note: the size of real_image_part is (B, 2, T, F)
        real_imag_part = torch.cat((real_imag_part[:, 0, :, :], real_imag_part[:, 1, :, :]), dim=-1).permute(0, 2, 1)

        inverse_transform = F.conv_transpose1d(real_imag_part,
                                               self.inverse_basis.to(real_imag_part.device),
                                               stride=self.hop_length,
                                               padding=0)

        padded = int(self.filter_length // 2)
        if length is None:
            if self.center:
                inverse_transform = inverse_transform[:, :, padded:-padded]
        else:
            if self.center:
                inverse_transform = inverse_transform[:, :, padded:]
            inverse_transform = inverse_transform[:, :, :length]

        return inverse_transform
