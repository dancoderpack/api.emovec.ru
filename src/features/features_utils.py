import numpy as np
import os
import shutil
import librosa
import torch


def create_folder(folder_path):
    """
      Create a new folder at the specified path.

      If a folder already exists at the given path, it will be deleted first and then recreated.

      Args:
          folder_path (str): The path of the folder to create.

      Returns:
          None
      """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    else:
        os.mkdir(folder_path)


def get_mfcc(music, sr):
    """
      Calculate the Mel-frequency cepstral coefficients (MFCC) of an audio signal.

      Args:
          music (ndarray): Audio signal.
          sr (int): Sampling rate of the audio signal.

      Returns:
          ndarray: MFCC of the audio signal.
      """
    sr = sr / 1000
    n_fft = int(30 * sr)
    hop_length = int(10 * sr)
    mfcc = librosa.feature.mfcc(y=music, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    return mfcc


def get_quadrants(arousal, valence):
    """
    Determine the quadrant based on arousal and valence values.

    Args:
        arousal (float): Arousal value.
        valence (float): Valence value.

    Returns:
        int: The quadrant number (1, 2, 3, or 4).
    """
    mid_val = 0.5
    if valence > mid_val:
        return 4 if arousal < mid_val else 1
    else:
        return 3 if arousal < mid_val else 2


def normalize_mfccs(mfcc):
    """
        Normalizes the MFCCs.

        Args:
            mfcc (torch.Tensor): MFCC tensor.

        Returns:
            torch.Tensor: Normalized MFCC tensor.
        """
    mean = torch.tensor(13.7398, dtype=torch.float)
    std = torch.tensor(54.0514, dtype=torch.float)
    return (mfcc - mean) / std


def make_mono_sound(song):
    """
        Converts a stereo sound to mono.

        Args:
            song (np.ndarray): Input audio waveform.

        Returns:
            song (np.ndarray): Mono audio waveform.
        """
    if len(song.shape) > 1:
        if song.shape[1] > 1:
            song = librosa.to_mono(song.T)
    return song


def preprocess_song(song, sr):
    """
        Preprocesses a song by calculating MFCC and normalizing it.

        Args:
            song (np.ndarray): Input audio waveform.
            sr (int): Sampling rate of the audio.

        Returns:
            torch.Tensor: Preprocessed MFCC tensor.
        """
    mfcc = get_mfcc(song, sr)
    mfcc = torch.tensor(mfcc, dtype=torch.float)
    mfcc = normalize_mfccs(mfcc)
    mfcc = torch.unsqueeze(mfcc, 0)
    return mfcc


def cut_music(music, sr):
    """
        Cuts the music into frames of a specified duration (1 sec).

        Args:
            music (np.ndarray): Input audio waveform.
            sr (int): Sampling rate of the audio.

        Returns:
            list: Music frames.
        """
    frame_duration = 1
    frame_length = int(frame_duration * sr)
    music = librosa.util.frame(music, frame_length=frame_length, hop_length=frame_length, axis=0)
    return music
