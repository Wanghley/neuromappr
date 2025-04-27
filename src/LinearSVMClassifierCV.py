from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

class LinearSVMClassifierCV:
    def __init__(self, alphas, kernel='linear', outer_folds=6, inner_folds=5, verbose=False):
        self.alphas = alphas
        self.kernel = kernel
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.verbose = verbose
        self.model = None
        self.best_alpha = None
        self.best_C = None

    def cross_validate_svm_alpha(self, X, y):
        outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True)
        best_alphas = []

        def evaluate_alpha(alpha, X_train_outer, y_train_outer):
            inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=True)
            C = 1.0 / alpha
            val_accuracies = []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y_train_outer):
                X_train_inner = X_train_outer[inner_train_idx]
                y_train_inner = y_train_outer[inner_train_idx]
                X_val_inner = X_train_outer[inner_val_idx]
                y_val_inner = y_train_outer[inner_val_idx]

                clf = SVC(kernel=self.kernel, C=C)
                clf.fit(X_train_inner, y_train_inner)
                val_acc = clf.score(X_val_inner, y_val_inner)
                val_accuracies.append(val_acc)

            return alpha, np.mean(val_accuracies)

        for outer_fold_idx, (train_idx_outer, _) in enumerate(
            tqdm(outer_cv.split(X, y), total=self.outer_folds, desc="🔁 Outer CV")
        ):
            X_train_outer = X[train_idx_outer]
            y_train_outer = y[train_idx_outer]

            alpha_results = Parallel(n_jobs=-1)(
                delayed(evaluate_alpha)(alpha, X_train_outer, y_train_outer)
                for alpha in tqdm(self.alphas, desc=f"    🔍 Alphas for Fold {outer_fold_idx+1}", leave=False)
            )

            best_alpha, best_val_acc = max(alpha_results, key=lambda x: x[1])
            best_alphas.append(best_alpha)

            if self.verbose:
                print(f"✅ Outer Fold {outer_fold_idx + 1}: Best α = {best_alpha}, Val Acc = {best_val_acc:.4f}")

        return float(np.mean(best_alphas)), best_alphas

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test):
        self.best_alpha, _ = self.cross_validate_svm_alpha(X_train, y_train)
        self.best_C = 1.0 / self.best_alpha

        self.model = SVC(kernel=self.kernel, C=self.best_C, probability=True)
        self.model.fit(X_train, y_train)

        y_scores = self.model.decision_function(X_test)
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        cls_report = classification_report(y_test, y_pred, output_dict=True)
        conf_mat = confusion_matrix(y_test, y_pred)

        metrics = {
            'accuracy': acc,
            'roc_auc': roc_auc,
            'classification_report': cls_report,
            'confusion_matrix': conf_mat
        }

        roc_info = {
            'fpr': fpr,
            'tpr': tpr
        }

        return self.model, metrics, roc_info, self.best_alpha
