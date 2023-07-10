import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd
import json
from models.model import EmotionModel
from src.models.model_utils import get_tarin_valid_data


class Visualizer():
    def __init__(self):
        self.model_arousal = EmotionModel()
        self.model_valence = EmotionModel()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_arousal.load_state_dict(
            torch.load('models/weights/arousal/best_model.pt', map_location=self.device))
        self.model_valence.load_state_dict(
            torch.load('models/weights/valence/best_model.pt', map_location=self.device))
        _, self.valid_data = get_tarin_valid_data()
        self.model_arousal, self.model_valence = self.model_arousal.to(self.device), self.model_valence.to(self.device)
        self.model_arousal.eval()
        self.model_valence.eval()
        self.arousal_history_dir = 'models/weights/arousal/history.csv'
        self.arousal_best_res_dir = 'models/weights/arousal/best_res.json'
        self.valence_history_dir = 'models/weights/valence/history.csv'
        self.valence_best_res_dir = 'models/weights/valence/best_res.json'
        sns.set_style('whitegrid')
        sns.set_palette('husl')


    def get_model_results(self):
        true_val_arousal = []
        pred_val_arousal = []
        true_val_valence = []
        pred_val_valence = []
        for music, arousal, valence in self.valid_data:
          out_arousal = self.model_arousal(music.to(self.device))
          out_valence = self.model_valence(music.to(self.device))
          true_val_arousal+=list(arousal.cpu().detach().numpy().reshape(1, -1)[0])
          true_val_valence+=list(valence.cpu().detach().numpy().reshape(1, -1)[0])
          pred_val_arousal+=list(out_arousal.cpu().detach().numpy().reshape(1, -1)[0])
          pred_val_valence+=list(out_valence.cpu().detach().numpy().reshape(1, -1)[0])
        return true_val_arousal, pred_val_arousal, true_val_valence, pred_val_valence


    def get_arousal_valence_plot(self, axis1, axis2):
        arousal_true, arousal_pred, valence_true, valence_pred = self.get_model_results()
        axis1.set_xlim(0, 1)
        axis1.set_ylim(0, 1)
        axis1.axvline(x=0.5, color='grey', linestyle='--')
        axis1.axhline(y=0.5, color='grey', linestyle='--')
        axis1.scatter(valence_pred, arousal_pred)
        axis1.set_xlabel('Valence')
        axis1.set_ylabel('Arousal')
        axis1.set_title('Дж. Рассел: arousal и valence')

        axis2.set_xlim(0, 1)
        axis2.set_ylim(0, 1)
        axis2.axvline(x=0.5, color='grey', linestyle='--')
        axis2.axhline(y=0.5, color='grey', linestyle='--')
        axis2.scatter(valence_true, arousal_true)
        axis2.set_xlabel('Valence')
        axis2.set_ylabel('Arousal')
        axis2.set_title('Дж. Рассел: arousal и valence')
        plt.show()


    def get_quadrants(self, arousal, valence):
        mid_val = 0.5
        if valence > mid_val:
            return 4 if arousal < mid_val else 1
        else:
            return 3 if arousal < mid_val else 2


    def get_quadrants_res(self):
        true_val_arosual, pred_val_arosual, true_val_valence, pred_val_valence = self.get_model_results()
        true = []
        pred = []
        for arousal, valence in zip(pred_val_arosual, pred_val_valence):
            pred.append(self.get_quadrants(arousal, valence))
        for arousal, valence in zip(true_val_arosual, true_val_valence):
            true.append(self.get_quadrants(arousal, valence))
        quadrants = {1: [], 2: [], 3: [], 4: []}
        for index, quadrant in enumerate(true):
            quadrants[quadrant].append(1) if pred[index] == quadrant else quadrants[quadrant].append(0)
        df = pd.DataFrame({'Quadrant 1': [np.mean(quadrants[1])],
                           'Quadrant 2': [np.mean(quadrants[2])],
                           'Quadrant 3': [np.mean(quadrants[3])],
                           'Quadrant 4': [np.mean(quadrants[4])]})
        return df


    def get_range(self, target='arousal'):
        true_val_arosual, pred_val_arosual, true_val_valence, pred_val_valence = self.get_model_results()
        pred_val = pred_val_arosual if target == 'arousal' else pred_val_valence
        min_val = np.min(pred_val)
        max_val = np.max(pred_val)
        df = pd.DataFrame({'min_val': min_val,
                           'max_val': max_val}, index=[target])
        return df


    def get_best_res(self, target='arousal'):
        dir = self.arousal_best_res_dir if target == 'arousal' else self.valence_best_res_dir
        with open(dir) as f:
            data = json.load(f)
        df = pd.DataFrame({'epoch': [data['epoch']],
                           'valid_loss': [data['valid_loss']],
                           'valid_metric': [data['valid_metric']]}, index=[target])
        return df


    def get_history_plot(self, axis1, axis2, target='arousal'):
        if target == 'arousal':
            df = pd.read_csv(self.arousal_history_dir)
        elif target == 'valence':
            df = pd.read_csv(self.valence_history_dir)
        axis1.plot(df['epoch'], df['train_loss'], label='train')
        axis1.plot(df['epoch'], df['valid_loss'], label='valid')
        axis1.legend()
        axis1.set_xlabel('epochs')
        axis1.set_ylabel('loss')
        axis1.set_title(f'{target} loss function')
        axis2.plot(df['epoch'], df['train_metric'], label='train')
        axis2.plot(df['epoch'], df['valid_metric'], label='valid')
        axis2.legend()
        axis2.set_xlabel('epochs')
        axis2.set_ylabel('metric')
        axis2.set_title(f'{target} metrics')
        plt.tight_layout()


    def get_true_pred_plot(self, axis, target='arousal'):
        true_val_arosual, pred_val_arosual, true_val_valence, pred_val_valence = self.get_model_results()
        axis.set_aspect('equal')
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        if target == 'arousal':
            axis.set_xticks([])
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_position('center')
            axis.text(0.40, 1.1, 'Arousal', verticalalignment='top')
            x = np.linspace(0, 1, len(true_val_arosual), endpoint=True)
            idx = np.argsort(pred_val_arosual)
            true_sorted_arousal = np.array(true_val_arosual)[idx.astype(int)]
            pred_sorted_arousal = np.array(pred_val_arosual)[idx.astype(int)]
            axis.plot(x, pred_sorted_arousal, lw=2, color='purple')
            axis.scatter(x, true_sorted_arousal, s=1)
        if target == 'valence':
            axis.set_yticks([])
            axis.spines['left'].set_visible(False)
            axis.spines['bottom'].set_position('center')
            axis.text(1.01, 0.515, 'Valence', horizontalalignment='right')
            y = np.linspace(0, 1, len(true_val_valence), endpoint=True)
            idx = np.argsort(pred_val_valence)
            true_sorted_arousal = np.array(true_val_valence)[idx.astype(int)]
            pred_sorted_arousal = np.array(pred_val_valence)[idx.astype(int)]
            axis.plot(pred_sorted_arousal, y, lw=2, color='purple')
            axis.scatter(true_sorted_arousal, y, s=1)


    def get_full_visualisation(self):
        fig, axes = plt.subplots(4, 2, figsize=(7, 12))
        self.get_history_plot(axes[0, 0], axes[0, 1], target='arousal')
        self.get_history_plot(axes[1, 0], axes[1, 1], target='valence')
        self.get_true_pred_plot(axes[2, 0], target='arousal')
        self.get_true_pred_plot(axes[2, 1], target='valence')
        self.get_arousal_valence_plot(axes[3, 0], axes[3, 1])
        plt.tight_layout()
        plt.show()

        arousal_metrics_df = self.get_best_res('arousal')
        arousal_range_df = self.get_range(target='arousal')
        arousal_df = pd.concat([arousal_metrics_df,arousal_range_df], axis=1)
        valence_metrics_df = self.get_best_res('valence')
        valence_range_df = self.get_range(target='valence')
        valence_df = pd.concat([valence_metrics_df,valence_range_df], axis=1)
        df = pd.concat([arousal_df, valence_df])
        quadrant_df = self.get_quadrants_res()
        print(df)
        print(quadrant_df)