from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def genre_preprocess(df):
    df['genres'] = df['genres'].apply(lambda x: x.split(", "))
    df['genres'] = df['genres'].apply(lambda x: [elem.strip('_') for elem in x])
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['genres'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = [i for i in tfidf.get_feature_names_out()]
    genre_df.drop(columns='unknown')
    genre_df.reset_index(drop=True, inplace=True)
    genre_df.drop(columns=['_hip_hop'], inplace=True)
    return genre_df


def get_sentiment_analisys(df):
    df['subjectivity'] = df['track_name'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['polarity'] = df['track_name'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df


def normalize(df):
    float_df = df[['acousticness', 'danceability', 'year', "artist_popularity", "song_popularity",
                   'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness',
                   'tempo', 'valence', 'key', 'mode', 'subjectivity', 'polarity']]
    scaler = MinMaxScaler()
    scaled_float_df = pd.DataFrame(scaler.fit_transform(float_df), columns=float_df.columns) * 0.5
    return scaled_float_df
