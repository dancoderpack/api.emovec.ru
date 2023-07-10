import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class MusicDataset(Dataset):
    """
        Dataset class for music data.

        Attributes:
            mfcc_dir (str): Directory path of the MFCC files.
            annot_df (DataFrame): DataFrame containing the annotations.
            music_name (list): List of music names. """

    def __init__(self, annot_dir, mfcc_dir):
        self.mfcc_dir = mfcc_dir
        self.annot_df = pd.read_csv(annot_dir)
        self.music_name = list(self.annot_df['song_id'].values)

    def __len__(self):
        return len(self.music_name)

    def normalize_mfccs(self, mfcc):
        """
                Normalizes the MFCCs.

                Args:
                    mfcc (torch.Tensor): MFCC tensor.

                Returns:
                    torch.Tensor: Normalized MFCC tensor.
                """
        mean = torch.tensor(10.8789, dtype=torch.float)
        std = torch.tensor(61.9204, dtype=torch.float)
        return (mfcc - mean) / std

    def __getitem__(self, idx):
        """
               Retrieves an item from the dataset based on the index.

               Args:
                   idx (int): Index of the item.

               Returns:
                   tuple: A tuple containing the MFCCs, arousal, and valence tensors.
               """
        series = self.annot_df.iloc[idx]
        mfcc = self.normalize_mfccs(
            torch.tensor(np.load(os.path.join(self.mfcc_dir, f"{series['song_id']}.npy")), dtype=torch.float))
        arousal = torch.tensor(series['arousal'], dtype=torch.float)
        valence = torch.tensor(series['valence'], dtype=torch.float)
        return mfcc, arousal, valence
