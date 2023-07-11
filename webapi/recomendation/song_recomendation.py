import pandas as pd
from response.GetRecommendationResponse import GetRecommendationResponse
from sklearn.metrics.pairwise import cosine_similarity
from config import get_config
import os
import random


def get_recommendation(ids):
    domain = get_config().web_domain
    host = f"https://{domain}"

    features_df = pd.read_csv('webapi/recomendation/rec_sys.csv')
    vector = features_df[features_df['id'].isin(ids)].drop(columns=['id']).mean(axis=0)
    all_songs = features_df[~features_df['id'].isin(ids)].drop(columns=['id'])

    try:
        res = cosine_similarity(all_songs.values, vector.values.reshape(1, -1))
    except ValueError:
        return
    recommendation = list(
        pd.DataFrame(res, index=features_df[~features_df['id'].isin(ids)]['id'])[0].nlargest(10).index
    )

    df = pd.read_csv('webapi/songs.csv')
    tracks_df = df[df['id'].isin(recommendation)]

    artworks = os.listdir('public/artworks')
    random_artworks = random.sample(artworks, 10)
    tracks = [{'id': tracks_df.iloc[index]['id'], 'artist': tracks_df.iloc[index]['artist_name'],
               'title': tracks_df.iloc[index]['track_name'],
               'artwork': f'{host}/public/artworks/' + random_artworks[index],
               'file': f'{host}/public/music/' + tracks_df.iloc[index]['id'] + '.mp3'} for index in
              range(tracks_df.shape[0])]

    return GetRecommendationResponse(songs=tracks)
