import torch
import soundfile as sf
import librosa

from models.model import EmotionModel
from src.models.model_utils import get_shorter_music_length, get_pad, count_sec_mean
from src.features.features_utils import make_mono_sound, cut_music, preprocess_song


class Predictor:
    """
        Predictor class for predicting emotion values of a song.

        Attributes:
            model_arousal (EmotionModel): Emotion model for arousal prediction.
            model_valence (EmotionModel): Emotion model for valence prediction.
            device (str): Device for running the models (e.g., 'cuda' or 'cpu'). """

    def __init__(self):
        self.model_arousal = EmotionModel()
        self.model_valence = EmotionModel()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_arousal.load_state_dict(
            torch.load('models/weights/arousal/best_model.pt', map_location=self.device))
        self.model_valence.load_state_dict(
            torch.load('models/weights/valence/best_model.pt', map_location=self.device))
        self.model_arousal, self.model_valence = self.model_arousal.to(self.device), self.model_valence.to(self.device)
        self.model_arousal.eval()
        self.model_valence.eval()


    def predict(self, song, sec=None, mean_sec=1):
        """
                Predicts emotion values for a given song.

                Args:
                    song (io.BytesIO): music.
                    sec (int): Duration in seconds to consider from the song.
                    mean_sec (int): Duration in seconds for calculating mean values.

                Returns:
                    tuple: A tuple containing the predicted valence and arousal values.
                """
        data, sr = sf.read(song)
        data = make_mono_sound(data)

        if sr != 44100:
            data = librosa.resample(data, orig_sr=sr, target_sr=44100)
        sr = 44100
        if sec is not None:
            data = get_shorter_music_length(data, sr, sec)
        data = cut_music(data, sr)

        arousal = []
        valence = []
        for elem in data:
            mfcc = preprocess_song(elem, sr)
            valence_sec = self.model_valence(mfcc.to(self.device)).squeeze()
            arousal_sec = self.model_arousal(mfcc.to(self.device)).squeeze()
            valence.append(valence_sec.cpu().detach().numpy())
            arousal.append(arousal_sec.cpu().detach().numpy())
        if sec is not None:
            valence, arousal = get_pad(valence, arousal, sec)
        valence = [round(float(v), 2) for v in valence]
        arousal = [round(float(a), 2) for a in arousal]
        if mean_sec > 1:
            valence, arousal = count_sec_mean(valence, arousal, mean_sec)
        return valence, arousal
