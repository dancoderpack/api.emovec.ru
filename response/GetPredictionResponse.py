from dataclasses import dataclass


@dataclass
class AVDVector:
    arousal: list
    valence: list
    distance: list


@dataclass
class Stats:
    mean_valence: float
    mean_arousal: float
    std_valence: float
    std_arousal: float
    qc_valence: int
    qc_arousal: int


@dataclass
class GetPredictionResponse:
    avd_vector: AVDVector
    stats: Stats

