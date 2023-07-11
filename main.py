import dataclasses

import soundfile
from flask import Flask, request, send_from_directory

from flask_cors import CORS

from webapi.repository import get_library
from music_preprocess.ContinuousEmotionRepresentation import ContinuousEmotionRepresentation
from music_preprocess.gpt import get_description
from webapi.recomendation.song_recomendation import get_recommendation

import soundfile as sf

print(sf.read("public/music/661y1Uk3VKNR4b9QWC7ziV.mp3"))

app = Flask("song-emotion-demo")

CORS(app, resources={r"/*": {"origins": "[https://emovec.ru, https://www.emovec.ru]"}})


@app.route("/public/<path:filename>")
def serve_static(filename):
    return send_from_directory("public", filename)


@app.route('/predictValuesDemo', methods=['POST'])
def get_values():
    song_id = request.get_json()['id']
    filename = f'public/music/{song_id}.mp3'
    print(soundfile.read(filename))
    emotion_representation = ContinuousEmotionRepresentation(song=soundfile.read(filename))
    return dataclasses.asdict(emotion_representation.get_static_prediction_result())


@app.route('/predict', methods=['POST'])
def get_prediction():
    song_length = int(request.form['song_length'])
    period = int(request.form['period'])
    song = request.files['song']
    print(type(song_length))

    emotion_representation = ContinuousEmotionRepresentation(song, song_length, period)
    return dataclasses.asdict(emotion_representation.get_static_prediction_result())


@app.route('/gpt_description', methods=['GET'])
def get_gpt_description():
    arousal = request.get_json()['arousal']
    valence = request.get_json()['valence']
    arousal = float(arousal)
    valence = float(valence)
    return dataclasses.asdict(get_description(arousal, valence))


@app.route('/library', methods=['GET'])
def get_library_api():
    return get_library()


@app.route('/recommendation', methods=['POST'])
def get_songs_recommendation():
    ids = request.get_json()['ids']
    return dataclasses.asdict(get_recommendation(ids))
