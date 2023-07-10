import librosa
import numpy as np
import os
import pandas as pd
from collections import Counter
import random
from src.features.features_utils import create_folder, get_mfcc, get_quadrants


def make_mfcc_dataset(music_dir, annot_dir):
    """
      Creates an MFCC representation for each audio file and saves it as a NumPy array in the
      'data/processed/mfcc' directory.

      Args:
          music_dir (str): The directory containing the audio files.
          annot_dir (str): The path to the annotation CSV file.

      Returns:
          None
      """
    create_folder('data/processed/mfcc')
    annot_df = pd.read_csv(annot_dir)
    for song in list(annot_df['song_id'].values):
        music, sr = librosa.load(os.path.join(music_dir, song + '.wav'), sr=None, mono=True)
        mfcc = get_mfcc(music, sr)
        if mfcc is None:
            continue
        np.save(f'data/processed/mfcc/{song}.npy', mfcc)


class DataPreprocessor():
    """
        Create annotation for balanced dataset
    """
    def __init__(self, df_dir):
        self.__df = pd.read_csv(df_dir)
        self.__make_new_dataset()

    def __add_quadrants_columns(self, quadrants):
        """
        Add a 'quadrant' column to the DataFrame.

        Args:
            quadrants (list): List of quadrant numbers.

        Returns:
            pandas.DataFrame: Updated DataFrame with the 'quadrant' column.
        """
        qudrants_df = pd.DataFrame(quadrants, columns=['quadrant'])
        self.__df = pd.concat([self.__df, qudrants_df], axis=1)
        return self.__df

    def __count_quadrants(self):
        """
        Count the number of samples in each quadrant.

        Returns:
        tuple: A tuple containing a list of quadrants and a Counter object with quadrant counts.
        """
        quadrants = []
        for row in range(self.__df.shape[0]):
            arousal = self.__df.iloc[row]['arousal']
            valence = self.__df.iloc[row]['valence']
            quadrants.append(get_quadrants(arousal, valence))
        quadrants_counter = Counter(quadrants)
        for key in quadrants_counter.keys():
            print(f'Quadrant {key} : {quadrants_counter[key]} samples')
        return quadrants, quadrants_counter

    def __save_csv(self):
        """
        Save the DataFrame as a CSV file.

        Returns:
        None
        """
        self.__df.to_csv('data/processed/annotation.csv', index=False)

    def __make_new_dataset(self):
        """
        Create a new dataset with a specified size by balancing the samples in each quadrant.

        Args:
        size (int): The desired size of each quadrants.

        Returns:
        None
        """
        quadrants, quadrants_counter = self.__count_quadrants()
        self.__df = self.__add_quadrants_columns(quadrants)
        size = quadrants_counter[4]
        for key in quadrants_counter.keys():
            quadrant_size = quadrants_counter[key]
            indexes = list(self.__df[self.__df['quadrant'] == key].index)

            if quadrant_size >= size:
                del_size = quadrants_counter[key] - size
                del_indexes = random.sample(indexes, del_size)
                self.__df.drop(index=del_indexes, inplace=True)

            else:
                added_size = size - quadrant_size
                added_indexes = random.sample(indexes, added_size)
                self.__df = pd.concat([self.__df, self.__df.loc[added_indexes]], axis=0).reset_index(drop=True)
        self.__save_csv()
