import pandas as pd
import random
import os
import subprocess

from config import get_config


def get_library():
    domain = get_config().web_domain
    host = f"https://{domain}"

    print(subprocess.call(['pwd']))
    df = pd.read_csv('webapi/songs.csv')
    tracks = list(df['id'].values)
    random_tracks = random.sample(tracks, 10)
    tracks_df = df[df['id'].isin(random_tracks)]
    artworks = os.listdir('public/artworks')
    random_artworks = random.sample(artworks, 10)
    tracks = [{'id': tracks_df.iloc[index]['id'], 'artist': tracks_df.iloc[index]['artist_name'],
               'title': tracks_df.iloc[index]['track_name'],
               'artwork': f'{host}/public/artworks/' + random_artworks[index],
               'file': f'{host}/public/music/' + tracks_df.iloc[index]['id'] + '.mp3'} for index in
              range(tracks_df.shape[0])]
    return tracks
