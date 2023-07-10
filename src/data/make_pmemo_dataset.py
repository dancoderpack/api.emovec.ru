import os
import pandas as pd

from src.data.data_utils import create_csv, cut_music, save_csv


def make_pmemo_dataset(song_dir, annotaion_dir):
    """
        Cut PMEmo dataset b

        Args:
            song_dir (str): Path to the directory containing the songs.
            annotation_dir (str): Path to the directory containing the annotations.

        Returns:
            None
        """
    dir_ = 'data/interim'
    create_csv(dir_)
    df = pd.read_csv(annotaion_dir)

    for elem in sorted(os.listdir(song_dir), key=lambda x: int(x[:x.index('.')])):
        results = cut_music(f'{song_dir}/{elem}')
        if results is None:
            continue
        music, sr = results
        music_name = int(elem.split('.')[0])

        arousal = list(df[df['musicId'] == music_name]['Arousal(mean)'].values)
        valence = list(df[df['musicId'] == music_name]['Valence(mean)'].values)

        for counter, index in enumerate(range(0, len(arousal), 2)):
            if (counter == len(music)) or (index + 1 == len(arousal)):
                continue
            arousal_sec = (arousal[index] + arousal[index + 1]) / 2
            valence_sec = (valence[index] + valence[index + 1]) / 2
            save_csv(music, music_name, dir_, sr, arousal_sec, valence_sec, counter)
