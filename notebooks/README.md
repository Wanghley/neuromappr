# 🧠 BCI Movement Classification – Notebooks

This folder contains modular Jupyter Notebooks used in the development and evaluation of a Support Vector Machine (SVM)-based classifier for Brain-Computer Interface (BCI) movement decoding.

The classification task distinguishes between EEG signals corresponding to **overt** and **imagined** arm movements. Each notebook reflects a logical step in the machine learning pipeline, supporting reproducibility, interpretability, and alignment with the ECE 580 mini-project report structure.

## 📁 Notebook Overview

| Notebook                          | Description |
|----------------------------------|-------------|
| `data_exploration.ipynb`         | Loads the EEG dataset and performs initial inspection of data structure, signal distribution, and basic statistics. |
| `feature_engineering.ipynb`      | Applies signal preprocessing (filtering, normalization) and extracts relevant features for classification. |
| `linear_svm.ipynb`               | Implements a baseline linear SVM model (no kernel) for within-condition training and initial evaluation. |
| `cross_validation.ipynb`         | Executes two-level cross-validation to tune the regularization parameter and assess model performance (ROC, accuracy). |
| `kernel_experiments.ipynb`       | Explores non-linear SVMs using kernels such as RBF and polynomial, comparing performance to the linear baseline. |
| `cross_train_analysis.ipynb`     | Analyzes model generalization by training on overt and testing on imagined data (and vice versa), highlighting domain shifts. |
| `visualizations.ipynb`           | Generates detailed visualizations including ROC curves, brain surface maps, and stem plots of channel weights. |
| `report_plots_and_figures.ipynb` | Prepares high-resolution, publication-ready plots and tables for inclusion in the slidedoc project report. |

## 🔁 How to Use

Each notebook can be run independently after ensuring the data is preprocessed and relevant scripts/functions are available. For reproducibility:
- Use a virtual environment (e.g., `venv` or `conda`)
- Track package versions in `requirements.txt` or `environment.yml`
- Output plots should be saved in `/reports/figures/` for final reporting

---

**Project Root Structure Suggestion:**
```
bci-movement-decoding/
├── notebooks/
├── data/
├── src/
├── reports/
├── references/
└── README.md
```

---

## 👩‍💻 Author
Wanghley Soares Martins  
Duke University – ECE 580, Spring 2025