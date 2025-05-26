# denoise_bayes_vae/dataset.py
# -*- encoding: utf-8 -*-

import os
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.audio_wav import load_audio_librosa_resample
from utils.logger import Logger

logger = Logger().get_logger()

class ChunkedSpeechDataset(Dataset):
    """
    Dataset that loads fixed-length chunks from noisy and clean audio files.
    """
    def __init__(self, noisy_paths, clean_paths, chunk_len=16000, target_sr=16000):
        self.chunk_len = chunk_len
        self.target_sr = target_sr
        self.index = []  # List of tuples: (noisy_path, clean_path, start_sec)

        for n_path, c_path in zip(noisy_paths, clean_paths):
            info = torchaudio.info(n_path)
            total_samples = info.num_frames
            sample_rate = info.sample_rate
            total_duration_sec = total_samples / sample_rate
            chunk_duration_sec = self.chunk_len / self.target_sr

            # calculate total chunks (each chunk's max length is 1 secs)
            num_full_chunks = int(total_duration_sec // chunk_duration_sec)
            residual_sec = total_duration_sec % chunk_duration_sec

            # if residual chunk length is more than 0.2 sec, it is appended
            if residual_sec >= 0.2:
                num_chunks = num_full_chunks + 1
            else:
                num_chunks = num_full_chunks

            for i in range(num_chunks):
                start_sec = i * chunk_duration_sec
                self.index.append((n_path, c_path, start_sec))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Load a chunk from each file
        :param idx:
        :return:
        """
        n_path, c_path, offset_sec = self.index[idx]
        duration_sec = self.chunk_len / self.target_sr  # 1sec

        noisy, n_sr = load_audio_librosa_resample(n_path, duration=duration_sec, offset=offset_sec)
        clean, c_sr = load_audio_librosa_resample(c_path, duration=duration_sec, offset=offset_sec)

        # Pad if too short
        if noisy.shape[1] < self.chunk_len:
            pad_len = self.chunk_len - noisy.shape[1]
            noisy = F.pad(noisy, (0, pad_len))
            clean = F.pad(clean, (0, pad_len))

        return noisy.squeeze(0), clean.squeeze(0)


class SpeechDatasetLoader:
    """
    Speech Denoise Dataset Loader Class to distribute train and validation dataset
    """
    def __init__(self, noisy_dir:str, clean_dir: str,
                 train_val_ratio:float = 0.8,
                 batch_size:int = 16, extension: str = '.wav'):
        """
        Speech Denoise Dataset Loader
        :param noisy_dir: (str) noisy data file directory for training and validation
        :param clean_dir: (str) noisy clean file directory for training and validation
        :param train_val_ratio: (float) train/validation split ratio (default: 0.8) for 80% train, 20% validation
        :param batch_size: (int) batch size
        :param extension: (str) only support wav format
        """
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.extension = extension
        self.train_val_ratio = train_val_ratio
        self.batch_size = batch_size
        self.train_noisy = []
        self.train_clean = []
        self.val_noisy = []
        self.val_clean = []

        logger.info('Start indexing input data.')

        if train_val_ratio >= 1:
            error = f'train_val_ratio must be less than 1.'
            logger.error(error)
            raise ValueError(error)

        self.__split_train_test_files()
        self.train_dataset = ChunkedSpeechDataset(self.train_noisy, self.train_clean)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True,
                                       persistent_workers=True)

        self.val_dataset = ChunkedSpeechDataset(self.val_noisy, self.val_clean)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     persistent_workers=True)

        logger.info(f'Indexed input dataset \n'
                    f'Total (clean, noisy) files  = ({len(self.train_clean) + len(self.val_clean)}, '
                    f'{len(self.train_noisy) + len(self.val_noisy)})\n'
                    f'Train (clean, noisy) files = ({len(self.train_clean)}, {len(self.train_noisy)})\n'
                    f'Validation (clean, noisy) files = ({len(self.val_clean)}, {len(self.val_noisy)})\n'
                    f'Total train chunks: {len(self.train_dataset)}\n'
                    f'Total validation chunks: {len(self.val_dataset)}\n')

    def get_train_dataloader(self):
        """
        get train data loader
        :return: (DataLoader)
        """
        return self.train_loader

    def get_val_dataloader(self):
        """
        get validation data loader
        :return: (DataLoader)
        """
        return self.val_loader

    def __split_train_test_files(self):
        """
        split train, test files
        :return:
        """
        noisy_files = sorted([f for f in os.listdir(self.noisy_dir) if f.endswith(self.extension)])
        clean_files = sorted([f for f in os.listdir(self.clean_dir) if f.endswith(self.extension)])
        noisy_paths = []
        clean_paths = []

        if not len(noisy_files):
            error = f'Not found noisy file in {self.noisy_dir}.'
            logger.error(error)
            raise ValueError()

        if not len(clean_files):
            error = f'Not found clean file in {self.clean_dir}.'
            logger.error(error)
            raise ValueError(error)

        if len(noisy_files) != len(clean_files):
            error = f'The number of noise files and clean files is different. ' \
                    f'noisy files:{len(noisy_files)}, clean files:{len(clean_files)}'
            logger.error(error)
            raise ValueError(error)

        for noisy_file in noisy_files:
            if noisy_file not in clean_files:
                error = f'Not found clean file. file={noisy_file}'
                logger.error(error)
                raise ValueError(error)

            noisy_paths.append(os.path.join(self.noisy_dir, noisy_file))
            clean_paths.append(os.path.join(self.clean_dir, noisy_file))

        # split train, test files according to train_val_ratio
        test_size = 1 - self.train_val_ratio
        self.train_noisy, self.val_noisy, self.train_clean, self.val_clean = \
            train_test_split(noisy_paths, clean_paths, test_size=test_size, shuffle=True, random_state=42)
