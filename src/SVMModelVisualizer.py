import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.inspection import permutation_importance

from scipy.interpolate import griddata
import mne

class SVMModelVisualizer:
    def __init__(self, model, X_test, y_test, electrode_path, kernel='linear'):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.kernel = kernel
        self.electrode_path = electrode_path
        self.y_pred = model.predict(X_test)
        self.y_score = model.decision_function(X_test)

        if self.kernel == 'linear':
            self.weights = model.coef_.flatten()
        else:
            result = permutation_importance(model, X_test, y_test, n_repeats=10, n_jobs=-1)
            self.weights = result.importances_mean

        self.electrode_coords = pd.read_csv(electrode_path, header=None).values
        self.electrode_coords_scaled = self.electrode_coords / np.max(np.abs(self.electrode_coords))
        self.W_x = self.weights[::2]
        self.W_y = self.weights[1::2]
        self.W_mag = np.sqrt(self.W_x**2 + self.W_y**2)

    def report_metrics(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        roc = roc_auc_score(self.y_test, self.y_score)
        report = classification_report(self.y_test, self.y_pred)
        print("Accuracy:", acc)
        print("ROC AUC:", roc)
        print("\nClassification Report:\n", report)
        return acc, roc, report

    def plot_weight_stem(self, ax=None, title="SVM Weights per Channel", xlabel="Channel Index", ylabel="Weight"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        ax.stem(self.weights)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return fig, ax

    def plot_topomap(self, ax=None, title="Topographic Electrode Weights"):
        electrode_names = [f"e{i}" for i in range(1, 103)]
        sfreq = 100
        info = mne.create_info(ch_names=electrode_names, sfreq=sfreq, ch_types='eeg')
        electrode_coords_3d = np.hstack([self.electrode_coords_scaled, np.zeros((self.electrode_coords.shape[0], 1))])
        pos_dict = dict(zip(electrode_names, electrode_coords_3d))
        montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame='head')
        info.set_montage(montage)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        else:
            fig = ax.figure

        ax.set_title(title)
        im, _ = mne.viz.plot_topomap(
            self.W_mag,
            self.electrode_coords_scaled,
            axes=ax,
            show=False,
            cmap='plasma',
            outlines='head',
            sphere=(0, 0, 0, 1.1),
            extrapolate='box',
            mask_params=dict(radius=0.1)
        )
        return fig, ax, im

    def plot_topomap_interpolated(self, axs=None, titles=("Interpolated Heatmap", "Topographic Map")):
        grid_x, grid_y = np.mgrid[-1:1:300j, -1:1:300j]
        grid_z = griddata(self.electrode_coords_scaled, self.W_mag, (grid_x, grid_y), method='cubic')
        mask = np.sqrt(grid_x**2 + grid_y**2) <= 1
        grid_z[~mask] = np.nan

        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        else:
            fig = axs[0].figure

        im1 = axs[0].imshow(grid_z.T, extent=(-1, 1, -1, 1), origin='lower', cmap='plasma', alpha=0.8)
        axs[0].set_title(titles[0])
        axs[0].axis('off')

        im2, _ = mne.viz.plot_topomap(
            self.W_mag,
            self.electrode_coords_scaled,
            axes=axs[1],
            show=False,
            cmap='plasma',
            outlines='head',
            contours=0,
            sphere=(0, 0, 0, 1.1),
            extrapolate='box'
        )
        axs[1].set_title(titles[1])

        return fig, axs, (im1, im2)

    def plot_confusion_matrix(self, ax=None, cmap='Blues', title='Confusion Matrix'):
        cm = confusion_matrix(self.y_test, self.y_pred)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                    xticklabels=self.model.classes_, yticklabels=self.model.classes_, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        return fig, ax

    def plot_roc_curve(self, ax=None, title='ROC Curve'):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_score)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_score(self.y_test, self.y_score):.3f}')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title(title)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid(True, linestyle=':')
        return fig, ax

    def plot_decision_statistic_kde(self, ax=None, title='Distribution of Decision Statistic Values'):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        sns.kdeplot(self.y_score[self.y_test == 0], fill=True, label='Class 0', color='blue', alpha=0.5, ax=ax)
        sns.kdeplot(self.y_score[self.y_test == 1], fill=True, label='Class 1', color='green', alpha=0.5, ax=ax)
        ax.axvline(x=0, color='red', linestyle='--', label='Decision Boundary')
        ax.set_title(title)
        ax.set_xlabel('Decision Statistic Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)
        return fig, ax

