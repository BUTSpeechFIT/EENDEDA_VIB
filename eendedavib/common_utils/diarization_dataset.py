#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

import common_utils.features as features
import common_utils.kaldi_data as kaldi_data
import numpy as np
import torch
import os
import logging
import pickle 

from typing import Tuple
import random 

def _count_frames(init: int, data_len: int, size: int, step: int) -> int:
    # no padding at edges, last remaining samples are ignored
    return int((init + data_len - size + step) / step)


def _gen_frame_indices(
    init_frame: int,
    data_length: int,
    size: int,
    step: int,
    use_last_samples: bool,
    min_length: int,
) -> None:
    i = -1
    for i in range(_count_frames(init_frame, data_length, size, step)):
        yield init_frame + (i * step), init_frame + (i * step) + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (init_frame + (i + 1) * step) > min_length:
            yield init_frame + (i + 1) * step, data_length


class KaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        chunk_size: int,
        context_size: int,
        feature_dim: int,
        frame_shift: int,
        frame_size: int,
        input_transform: str,
        n_speakers: int,
        sampling_rate: int,
        shuffle: bool,
        subsampling: int,
        use_last_samples: bool,
        min_length: int,
        dtype: type = np.float32,
    ):
        self.data_dir = data_dir
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.feature_dim = feature_dim
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.sampling_rate = sampling_rate
        self.chunk_indices = []

        self.data = kaldi_data.KaldiData(self.data_dir)

        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            data_len = int(
                self.data.reco2dur[rec] * sampling_rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            if self.data.uem:
                init_frame = (self.data.uem[rec][0] * sampling_rate / frame_shift)
                init_frame = int(init_frame / self.subsampling)
                data_len = (self.data.uem[rec][1] * sampling_rate / frame_shift)
                data_len = int(data_len / self.subsampling)
            else:
                init_frame = 0
            if chunk_size > 0:
                for st, ed in _gen_frame_indices(
                        init_frame,
                        data_len,
                        chunk_size,
                        chunk_size,
                        use_last_samples,
                        min_length
                ):
                    self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling))
            else:
                self.chunk_indices.append(
                    (rec, 0, data_len * self.subsampling))
        logging.info(f"#files: {len(self.data.wavs)}, "
                     "#chunks: {len(self.chunk_indices)}")

        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.chunk_indices)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        rec, st, ed = self.chunk_indices[i]
        Y, T = features.get_labeledSTFT(
            self.data,
            rec,
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            self.n_speakers
        )
        Y = features.transform(
            Y, self.sampling_rate, self.feature_dim, self.input_transform)
        Y_spliced = features.splice(Y, self.context_size)
        Y_ss, T_ss = features.subsample(Y_spliced, T, self.subsampling)

        # If the sample contains more than "self.n_speakers" speakers,
        #  extract top-(self.n_speakers) speakers
        if self.n_speakers and T_ss.shape[1] > self.n_speakers:
            selected_spkrs = np.argsort(
                T_ss.sum(axis=0))[::-1][:self.n_speakers]
            T_ss = T_ss[:, selected_spkrs]

        return torch.from_numpy(np.copy(Y_ss)), torch.from_numpy(
            np.copy(T_ss)), rec, st, ed 

class PrecomputedDiarizationFeatures(torch.utils.data.Dataset):
    def __init__(
        self,
        features_dir: str,
        chunk_size: int,
        shuffle: bool,
        use_last_samples: bool,
        min_length: int,
        dtype: type = np.float32,
    ):
        self.features_dir = features_dir
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.chunk_indices = []

        self.saved = None  # used in case of empty sequence

        self.filenames = [os.path.join(self.features_dir, f) for f in
                          os.listdir(self.features_dir) if
                          os.path.isfile(os.path.join(self.features_dir, f))]
        if shuffle:
            random.shuffle(self.filenames)

        # make chunk indices: filename, recording_name, start_frame, end_frame
        for idx, fn in enumerate(self.filenames):
            with open(fn, 'rb') as f:
                d = pickle.load(f)
#                zipped = list(zip(d['xs'], d['ts'], d['names'], d['beg'],
#                                  d['end'], d['spk_ids']))
            
            data_len = d['xs'][0].shape[0]
            rec = d['names'][0]
            init_frame = 0
            if chunk_size > 0:
                for st, ed in _gen_frame_indices(
                        init_frame,
                        data_len,
                        chunk_size,
                        chunk_size,
                        use_last_samples,
                        min_length
                ):
                    self.chunk_indices.append((fn, rec, st, ed))
            else:
                self.chunk_indices.append((fn, rec, 0, data_len))
        logging.info(f"#files: {len(self.filenames)}, "
                     "#chunks: {len(self.chunk_indices)}")

        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.chunk_indices)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        fn, rec, st, ed = self.chunk_indices[i]
        with open(fn, 'rb') as f:
            d = pickle.load(f)

        Y = d['xs'][0][st:ed]
        T = d['ts'][0][st:ed]
        speaker_ids = d['spk_ids'][0][st:ed]
        if Y.shape[0] == 0:
            print(f"{rec, st, ed} is empty: {Y.shape}, replacing with saved sequence")
            Y, T, speaker_ids = self.saved
        else:
            self.saved = (Y, T, speaker_ids)

        return torch.from_numpy(np.copy(Y)), torch.from_numpy(
            np.copy(T)), rec, st, ed

