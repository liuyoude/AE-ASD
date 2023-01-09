import re
import torch
from torch.utils.data import Dataset
from audiomentations import TimeStretch, PitchShift, AddGaussianSNR, FrequencyMask, TimeMask, Reverse, Gain, Resample, \
    PolarityInversion, ApplyImpulseResponse
import torchaudio
import librosa
import numpy as np

import utils


class ASDDataset(Dataset):
    def __init__(self, dirs: list, args):
        self.filename_list = []
        self.sr = args.sr
        self.n_mels = args.n_mels
        self.frames = args.frames
        for dir in dirs:
            self.filename_list.extend(utils.get_filename_list(dir))
        self.wav2mel = utils.Wave2Mel(sr=args.sr, power=args.power,
                                      n_fft=args.n_fft, n_mels=args.n_mels,
                                      win_length=args.win_length, hop_length=args.hop_length)

    def __getitem__(self, item):
        filename = self.filename_list[item]
        return self.transform(filename)

    def transform(self, filename):
        # label, one_hot = utils.get_label('/'.join(filename.split('/')[-3:]), self.att2idx, self.file_att_2_idx)
        (x, _) = librosa.core.load(filename, sr=self.sr, mono=True)
        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        dims = self.n_mels * self.frames
        vector_size = x_mel.shape[1] - self.frames + 1
        mel_vec = torch.zeros((vector_size, dims))
        for t in range(self.frames):
            mel_vec[:, t * self.n_mels: (t + 1) * self.n_mels] = x_mel[:, t: t + vector_size].T
        return mel_vec

    def __len__(self):
        return len(self.filename_list)


if __name__ == '__main__':
    import numpy as np

    a = np.random.random((16000*2,))
    xs = [a[np.newaxis, :], a[np.newaxis, :]]
    xs = np.concatenate(xs, axis=0)
    print(xs.shape)
    # x_wavs = augment_list[2](xs, sample_rate=16000).copy()
    # print(x_wavs.shape)
    label = torch.tensor([3] * 5)
    print(label)
    #
    # part_Aug = PartAugmentation(sr=16000, augs_list=normal_augs, secs=5.0)
    # b = part_Aug(a)
