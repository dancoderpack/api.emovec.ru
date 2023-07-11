import numpy as np
from io import BytesIO
from response.GetPredictionResponse import GetPredictionResponse, Stats, AVDVector
from src.models.predict_model import Predictor


class ContinuousEmotionRepresentation:
    """
    Represents continuous emotion representation for a song.

    Attributes:
        __predictor (Predictor): The predictor object used for model prediction.
        valence (list): Array of valence values for the song.
        arousal (list): Array of arousal values for the song.
    """

    def __init__(self, song, song_length=None, period=1):
        """
               Initializes a ContinuousEmotionRepresentation object.

               Args:
                   song (str or file-like object): The path to the song file or a file-like object containing the song data.
                   song_length (float, optional): The length of the song in seconds. Defaults to None.
                   period (int, optional): The period used for prediction. Defaults to 1.
               """
        self.song = song
        self.song_length = song_length
        self.period = period
        self.__predictor = Predictor()
        self.valence, self.arousal = self.__get_model_prediction()

    def __get_mean(self):
        """
                Calculates the mean valence and arousal values.

                Returns:
                    tuple: A tuple containing the mean valence and arousal values.
                """
        return np.mean(self.valence), np.mean(self.arousal)

    def __get_std(self):
        """
                Calculates the standard deviation of valence and arousal values.

                Returns:
                    tuple: A tuple containing the standard deviation of valence and arousal values.
                """
        return np.std(self.valence), np.std(self.arousal)

    def __get_qc(self):
        """
                Calculates the number of quick changes in valence and arousal.

                Returns:
                    tuple: A tuple containing the number of quick changes in valence and arousal.
                """
        arousal_qc = sum(1 for i in range(1, len(self.arousal)) if self.arousal[i] * self.arousal[i - 1] < 0)
        valence_qc = sum(1 for i in range(1, len(self.valence)) if self.valence[i] * self.valence[i - 1] < 0)
        return valence_qc, arousal_qc

    def __get_distance(self):
        """
                Calculates the Euclidean distance between valence and arousal values.

                Returns:
                    list: A list containing the Euclidean distances between valence and arousal values.
                """
        distance = [np.sqrt(v * v + a * a) for v, a in zip(self.valence, self.arousal)]
        return distance

    def __get_model_prediction(self):
        """
                Obtains the valence and arousal predictions for the song using the predictor model.

                Returns:
                    tuple: A tuple containing the valence and arousal predictions for the song.
                """
        if isinstance(self.song, str):
            with open(self.song, 'rb') as f:
                music = BytesIO(f.read())
        else:
            music = BytesIO(self.song.stream.read())
        valence, arousal = self.__predictor.predict(music, self.song_length, self.period)
        return valence, arousal

    def get_static_prediction_result(self):
        """
                Returns a prediction result object containing various statistics about the song.

                Returns:
                    GetPredictionResponse: A prediction result object containing statistics.
                """
        mean_valence, mean_arousal = self.__get_mean()
        std_valence, std_arousal = self.__get_std()
        qc_valence, qc_arousal = self.__get_qc()
        distance = self.__get_distance()
        prediction_response = GetPredictionResponse(
            stats=Stats(
                mean_valence=round(mean_valence, 4),
                mean_arousal=round(mean_arousal, 4),
                std_valence=round(std_valence, 4),
                std_arousal=round(std_arousal, 4),
                qc_arousal=round(qc_arousal, 4),
                qc_valence=round(qc_valence, 4),
            ),
            avd_vector=AVDVector(
                arousal=self.arousal,
                valence=self.valence,
                distance=distance
            )
        )
        return prediction_response
