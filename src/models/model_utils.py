from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
import csv
from src.features.dataloader import MusicDataset


def split_annotation():
    """
        Splits the annotation data into train and validation sets.

        Returns:
            None
        """
    df = pd.read_csv('data/processed/annotation.csv')
    x_train, x_valid = train_test_split(df, test_size=0.25,
                                        random_state=42)
    x_train.to_csv('data/processed/train_annotation.csv', index=False)
    x_valid.to_csv('data/processed/valid_annotation.csv', index=False)


def get_tarin_valid_data():
    """
        Retrieves the training and validation dataloaders.

        Returns:
            tuple: A tuple containing the training data DataLoader and the validation data DataLoader.
        """
    split_annotation()
    train_dir = 'data/processed/train_annotation.csv'
    valid_dir = 'data/processed/valid_annotation.csv'
    mfcc_dir = 'data/processed/mfcc'
    train_dataset = MusicDataset(train_dir, mfcc_dir)
    valid_dataset = MusicDataset(valid_dir, mfcc_dir)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    return train_data, valid_data


def make_history_file(directory):
    """
        Creates a training history file in the specified directory.

        Args:
            directory (str): Directory path to create the history file.

        Returns:
            None
        """
    with open(f'{directory}/history.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows([['epoch', 'train_loss', 'train_metric', 'valid_loss', 'valid_metric']])


def print_results(epoch, history):
    """
       Prints the results (loss and metric) for a given epoch.

       Args:
           epoch (int): Epoch number.
           history (dict): Dictionary containing the history of losses and metrics.

       Returns:
           None
       """
    print(f'epoch: {epoch}\n'
          f'train: loss {history["train_losses"][-1]:.4f}\n'
          f'train: metric {history["train_metrics"][-1]:.4f}\n'
          f'valid: loss {history["valid_losses"][-1]:.4f}\n'
          f'valid: metric {history["valid_metrics"][-1]:.4f}')
    print(f'{"-" * 35}')
    print()


def get_shorter_music_length(music, sr, sec):
    """
       Retrieves a shorter length of music from the beginning.

       Args:
           music (np.ndarray): Input audio waveform.
           sr (int): Sampling rate of the audio.
           sec (int): Duration in seconds to retrieve from the beginning.

       Returns:
           np.ndarray: Shorter length of music from the beginning.
       """
    duration = sr * sec
    music = music[:duration]
    return music


def get_pad(valence, arousal, sec):
    """
       Pads the valence and arousal arrays with zeros to match a specified duration.

       Args:
           valence (list): Array of valence values.
           arousal (list): Array of arousal values.
           sec (int): Duration in seconds to match.

       Returns:
           tuple: A tuple containing the padded valence and arousal arrays.
       """
    if len(valence) < sec:
        length = sec - len(valence)
        valence = valence + [np.array(0, dtype=float)] * length
        arousal = arousal + [np.array(0, dtype=float)] * length
    return valence, arousal


def count_sec_mean(valence, arousal, sec):
    """
        Calculates the mean valence and arousal values for each specified duration.

        Args:
            valence (list): Array of valence values.
            arousal (list): Array of arousal values.
            sec (int): Duration in seconds.

        Returns:
            tuple: A tuple containing the mean valence and arousal values for each duration.
        """
    valence_sec = []
    arousal_sec = []
    for elem in range(sec, len(valence) + 1, sec):
        valence_sec.append(round(np.mean(valence[elem - sec:elem]), 4))
        arousal_sec.append(round(np.mean(arousal[elem - sec:elem]), 4))
    return valence_sec, arousal_sec
