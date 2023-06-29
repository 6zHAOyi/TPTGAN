import torch
from torch.utils.data import Dataset
from preprocess import ToTensor, STFT

import numpy as np
import random
import h5py


class TrainingDataset(Dataset):
    r"""Training dataset."""

    def __init__(self, file_path, frame_size=512, frame_shift=256, nsamples=16000*2):
        # file_path is the path of training dataset
        # option1: '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/timit_mix/trainset/two_data'
        # option2 : .txt file format  file_path='/data/KaiWang/pytorch_learn/pytorch_for_speech/DDAEC/train_file_list'

        # self.file_list = glob.glob(os.path.join(file_path, '*'))

        with open(file_path, 'r') as train_file_list:
            self.file_list = [line.strip() for line in train_file_list.readlines()]

        self.nsamples = nsamples
        # self.STFT = STFT(frame_size=frame_size,
        #                  frame_shift=frame_shift)
        self.to_tensor = ToTensor()
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __len__(self):
        # print(len(self.file_list))
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')
        feature = reader['noisy_raw'][:]
        label = reader['clean_raw'][:]
        reader.close()

        size = feature.shape[0]
        if size >= self.nsamples:
            start = random.randint(0, max(0, size - self.nsamples))
            noisy = feature[start:start + self.nsamples]
            clean = label[start:start + self.nsamples]
        else:
            # noisy = np.zeros(self.nsamples, dtype="float32")
            # clean = np.zeros(self.nsamples, dtype="float32")
            # noisy[0:size] = feature
            # clean[0:size] = label
            units = self.nsamples // size
            clean = []
            noisy = []
            for i in range(units):
                clean.append(label)
                noisy.append(feature)
            clean.append(label[: self.nsamples % size])
            noisy.append(feature[: self.nsamples % size])
            clean = np.concatenate(clean, axis=-1)
            noisy = np.concatenate(noisy, axis=-1)

        # noisy = np.reshape(noisy, [1, -1])  # [1, sig_len]
        # clean = np.reshape(clean, [1, -1])  # [1, sig_len]

        noisy = self.to_tensor(noisy)  # [sig_len,]
        # noisy = torch.stft(input=noisy, n_fft=self.frame_size, hop_length=self.frame_shift, window=torch.hamming_window(self.frame_size), onesided=True)  # [1, F, N, 2]
        clean = self.to_tensor(clean)  # [sig_len,]

        return noisy, clean


class EvalDataset(Dataset):
    r"""Evaluation dataset."""

    def __init__(self, file_path, frame_size=512, frame_shift=256):
        with open(file_path, 'r') as validation_file_list:
            self.file_list = [line.strip() for line in validation_file_list.readlines()]

        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')

        feature = reader['noisy_raw'][:]
        label = reader['clean_raw'][:]

        # feature = np.reshape(feature, [1, -1])  # [1, sig_len]

        feature = self.to_tensor(feature)  # [sig_len, ]
        # feature = torch.stft(input=feature, n_fft=self.frame_size, hop_length=self.frame_shift,
        #                      window=torch.hamming_window(self.frame_size), return_complex=True)  # [1, F, N]
        label = self.to_tensor(label)  # [sig_len, ]

        return feature, label


class Company_EvalDataset(Dataset):
    r"""Evaluation dataset."""

    def __init__(self, file_path, frame_size=512, frame_shift=256):
        # self.filename = filename
        with open(file_path, 'r') as validation_file_list:
            self.file_list = [line.strip() for line in validation_file_list.readlines()]

        self.STFT = STFT(frame_size=frame_size,
                                         frame_shift=frame_shift)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')

        feature = reader['noisy_raw'][:]
        label = reader['clean_raw'][:]

        # feature = np.reshape(feature, [1, 1, -1])  # [1, 1, sig_len]

        # feature = self.get_frames(feature)  # [1, 1, num_frames, frame_size]

        # feature = self.to_tensor(feature)  # [sig_len, ]
        label = self.to_tensor(label)  # [sig_len, ]

        return feature, label


# testing 中 clean 和 noisy分不同的noisy和dB
class TestDataset(Dataset):
    r"""Evaluation dataset."""

    def __init__(self, file_path, frame_size=512, frame_shift=256):
        # self.filename = filename
        with open(file_path, 'r') as validation_file_list:
            self.file_list = [line.strip() for line in validation_file_list.readlines()]

        self.STFT = STFT(frame_size=frame_size,
                                         frame_shift=frame_shift)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')

        feature = reader['noisy_raw'][:]
        feature_copy = feature

        feature = np.reshape(feature, [1, 1, -1])  # [1, 1, sig_len]

        feature = self.STFT(feature)  # [1, 1, num_frames, frame_size]

        feature = self.to_tensor(feature)  # [1, 1, num_frames, frame_size]
        feature_copy = self.to_tensor(feature_copy)

        return feature, feature_copy


class TrainCollate_real(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            # bacth[i][0]=feature bacth[i][1]=label
            feat_dim = batch[0][0].shape[1]  # frame_size
            label_dim = batch[0][1].shape[-1]  # sig_len

            feat_nchannels = batch[0][0].shape[0]  # 1
            label_nchannels = batch[0][1].shape[0]  # 1

            # sorted by sig_len for label
            sorted_batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)
            # (num_frames, sig_len)
            # lengths = list(map(lambda x: (x[0].shape[1], x[1].shape[1]), sorted_batch))
            lengths = list(map(lambda x: (x[0].shape[2], x[1].shape[1]), sorted_batch))

            padded_feature_batch = torch.zeros((len(lengths), feat_nchannels, feat_dim, lengths[0][0], 2))
            # padded_feature_batch = torch.zeros((len(lengths), lengths[0][0]))
            padded_label_batch = torch.zeros((len(lengths), label_nchannels, lengths[0][1]))
            lengths1 = torch.zeros((len(lengths),), dtype=torch.int32)

            for i in range(len(lengths)):
                padded_feature_batch[i, :, :, 0:lengths[i][0], :] = sorted_batch[i][0]
                # padded_label_batch[i, :, 0:lengths[i][1]] = sorted_batch[i][1]
                padded_label_batch[i, :, 0:lengths[i][1]] = sorted_batch[i][1]
                lengths1[i] = lengths[i][1]

            return padded_feature_batch, padded_label_batch, lengths1   # padded_feature[B, 2, frame_nums, frame_size] padded_label[B, 1, sig_len] lengths1:sig_len
        else:
            raise TypeError('`batch` should be a list.')


class TrainCollate_complex(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            # bacth[i][0]=feature bacth[i][1]=label
            feat_dim = batch[0][0].shape[1]  # frame_size
            label_dim = batch[0][1].shape[-1]  # sig_len

            feat_nchannels = batch[0][0].shape[0]  # 1
            label_nchannels = batch[0][1].shape[0]  # 1

            # sorted by sig_len for label
            sorted_batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)
            # (num_frames, sig_len)
            # lengths = list(map(lambda x: (x[0].shape[1], x[1].shape[1]), sorted_batch))
            lengths = list(map(lambda x: (x[0].shape[-1], x[1].shape[1]), sorted_batch))

            padded_feature_batch = torch.zeros((len(lengths), feat_nchannels, feat_dim, lengths[0][0]))
            # padded_feature_batch = torch.zeros((len(lengths), lengths[0][0]))
            padded_label_batch = torch.zeros((len(lengths), label_nchannels, lengths[0][1]))
            lengths1 = torch.zeros((len(lengths),), dtype=torch.int32)

            for i in range(len(lengths)):
                padded_feature_batch[i, :, :, 0:lengths[i][0]] = sorted_batch[i][0]
                # padded_label_batch[i, :, 0:lengths[i][1]] = sorted_batch[i][1]
                padded_label_batch[i, :, 0:lengths[i][1]] = sorted_batch[i][1]
                lengths1[i] = lengths[i][1]

            return padded_feature_batch, padded_label_batch, lengths1   # padded_feature[B, 2, frame_nums, frame_size] padded_label[B, 1, sig_len] lengths1:sig_len
        else:
            raise TypeError('`batch` should be a list.')

class EvalCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            return batch[0][0], batch[0][1]
        else:
            raise TypeError('`batch` should be a list.')


class Company_EvalCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            return batch[0][0], batch[0][1]
        else:
            raise TypeError('`batch` should be a list.')


class TestCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            # testdataloder 中的batch_size = 1; 因此就返回仅有的一个(feature, label)
            shape = batch[0][1].shape[-1]
            # del batch[0][1]
            return batch[0][0], shape
        else:
            raise TypeError('`batch` should be a list.')
