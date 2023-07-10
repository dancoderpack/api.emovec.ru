import soundfile as sf
import librosa
import gdown
import pandas as pd
import csv
import os
import shutil
from src.features.features_utils import make_mono_sound


def save_csv(music, music_name, dir_, sr, arousal_sec, valence_sec, counter):
    """
        Saves a music file and its corresponding annotation to a CSV file.

        Args:
            music (numpy.ndarray): The audio data of the music.
            music_name (int): The name of the music.
            dir_ (str): The directory path where the files will be saved (pememo or deam).
            sr (int): The sample rate of the music.
            arousal_sec (float): The arousal value in seconds.
            valence_sec (float): The valence value in seconds.
            counter (int): The counter value used for naming the files.

        Returns:
            None
        """
    music = music[counter]
    sf.write(f'{dir_}/music/{music_name}_{counter}.wav', music, sr)
    with open(f'{dir_}/annotation.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerows([[f'{music_name}_{counter}', arousal_sec, valence_sec]])


def create_csv(dir_):
    """
        Creates a new CSV file for storing song annotations.

        Args:
            dir_ (str): The directory path where the CSV file will be created.

        Returns:
            None
        """
    with open(f'{dir_}/annotation.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows([['song_id', 'arousal', 'valence']])


def cut_music(song_dir):
    """
        Cuts a music file starting from 15 seconds and divides it into 1 sec frames.

        Args:
            song_dir (str): The directory path of the music file.

        Returns:
            tuple: A tuple containing the cut music and its sample rate (music, sr).
        """
    music, sr = sf.read(song_dir)
    music = make_mono_sound(music)
    if sr != 44100:
        music = librosa.resample(music, orig_sr=sr, target_sr=44100)
    sr = 44100
    start_time = librosa.time_to_samples(15, sr=sr)
    music = music[start_time:]
    music_length = music.shape[0] // sr
    if music_length == 0:
        return
    frame_duration = 1
    frame_length = int(frame_duration * sr)
    music = librosa.util.frame(music, frame_length=frame_length, hop_length=frame_length, axis=0)
    return music, sr


def download_folder_from_google_drive(url):
    """
            Download datasets from google drive

            Args:
                url (str): The URL of the Google Drive folder.

            Returns:
                None
            """
    gdown.download_folder(url, quiet=True, use_cookies=False, remaining_ok=True)


def download_file_from_doodle_drive(url, output_path):
    """
        Downloads a file from Doodle Drive using the provided URL.

        Args:
            url (str): URL of the file to download from Doodle Drive.
            output_path (str): Path to save the downloaded file.

        Returns:
            None
        """
    gdown.download(url, output_path, quiet=False, fuzzy=True)