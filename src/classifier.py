"""Model training, evaluation, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Classifier:
    """Train a calibrated linear SVM and a Random Forest."""

    def __init__(self, workers: int = -1):
        self.workers = workers
        self.svm_pipeline = None
        self.rf_pipeline = None
        self.thresholds = {"svm": 0.5, "rf": 0.5}

    def train(self, X_train: np.ndarray, y_train: np.ndarray, tune: bool = False) -> None:
        print("Training calibrated Linear SVM")
        self.svm_pipeline = self._train_svm(X_train, y_train, tune)

        print("\nTraining Random Forest")
        self.rf_pipeline = self._train_rf(X_train, y_train, tune)

    def _train_svm(self, X: np.ndarray, y: np.ndarray, tune: bool):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        base = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            C=1.0,
                            class_weight="balanced",
                            dual="auto",
                            max_iter=20000,
                            random_state=42,
                        ),
                        cv=3,
                        n_jobs=1,
                    ),
                ),
            ]
        )

        if tune:
            grid = GridSearchCV(
                base,
                {
                    "svm__estimator__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                },
                cv=cv,
                scoring="roc_auc",
                n_jobs=self.workers,
                verbose=1,
            )
            grid.fit(X, y)
            print(f"  best SVM params: {grid.best_params_}")
            print(f"  best SVM CV AUC: {grid.best_score_:.4f}")
            return grid.best_estimator_

        scores = cross_val_score(base, X, y, cv=cv, scoring="roc_auc", n_jobs=self.workers)
        print(f"  SVM CV AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
        base.fit(X, y)
        return base

    def _train_rf(self, X: np.ndarray, y: np.ndarray, tune: bool):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        if tune:
            rf = RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced")
            grid = GridSearchCV(
                rf,
                {
                    "n_estimators": [300, 600],
                    "max_features": ["sqrt", 0.35],
                    "min_samples_leaf": [1, 2, 4],
                },
                cv=cv,
                scoring="roc_auc",
                n_jobs=self.workers,
                verbose=1,
            )
            grid.fit(X, y)
            print(f"  best RF params: {grid.best_params_}")
            print(f"  best RF CV AUC: {grid.best_score_:.4f}")
            best_params = grid.best_params_
            best_rf = RandomForestClassifier(
                random_state=42,
                n_jobs=1,
                class_weight="balanced",
                n_estimators=best_params["n_estimators"],
                max_features=best_params["max_features"],
                min_samples_leaf=best_params["min_samples_leaf"],
            )
            best = CalibratedClassifierCV(best_rf, cv=3, method="isotonic", n_jobs=self.workers)
            best.fit(X, y)
            return best

        rf_base = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        scores = cross_val_score(rf_base, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        print(f"  RF CV AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
        rf = CalibratedClassifierCV(rf_base, cv=3, method="isotonic", n_jobs=self.workers)
        rf.fit(X, y)
        return rf

    def predict(self, X: np.ndarray, model: str = "svm") -> np.ndarray:
        proba = self.predict_proba(X, model)
        return (proba >= self.thresholds.get(self._model_key(model), 0.5)).astype(np.int32)

    def predict_proba(self, X: np.ndarray, model: str = "svm") -> np.ndarray:
        return self._model(model).predict_proba(X)[:, 1]

    def calibrate_thresholds(self, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
        for model_key in ["svm", "rf"]:
            proba = self.predict_proba(X_val, model_key)
            candidates = np.unique(np.concatenate([[0.05, 0.5, 0.95], proba]))
            best_threshold = 0.5
            best_score = -1.0
            for threshold in candidates:
                pred = (proba >= threshold).astype(np.int32)
                score = f1_score(y_val, pred, zero_division=0)
                if score > best_score or (
                    score == best_score and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
                ):
                    best_score = score
                    best_threshold = float(threshold)
            self.thresholds[model_key] = best_threshold
            print(f"  {model_key} threshold: {best_threshold:.4f} (val F1={best_score:.4f})")
        return self.thresholds

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, results_dir: str = "results") -> dict:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics = {}

        for name, model_key in [("SVM", "svm"), ("RandomForest", "rf")]:
            y_pred = self.predict(X_test, model_key)
            y_proba = self.predict_proba(X_test, model_key)
            metrics[name] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "auc_roc": float(roc_auc_score(y_test, y_proba)),
                "threshold": float(self.thresholds[model_key]),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }

        self._print_metrics(metrics)
        self._plot_confusion_matrices(metrics, out)
        self._plot_roc_curves(X_test, y_test, out)
        self._plot_feature_importance(out)
        with (out / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def save_models(self, model_dir: str = "models") -> None:
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.svm_pipeline, out / "svm_model.pkl")
        joblib.dump(self.rf_pipeline, out / "rf_model.pkl")
        with (out / "thresholds.json").open("w", encoding="utf-8") as f:
            json.dump(self.thresholds, f, indent=2)
        print(f"Models saved to {out}")

    def load_models(self, model_dir: str = "models") -> None:
        root = Path(model_dir)
        self.svm_pipeline = joblib.load(root / "svm_model.pkl")
        self.rf_pipeline = joblib.load(root / "rf_model.pkl")
        thresholds_path = root / "thresholds.json"
        if thresholds_path.exists():
            with thresholds_path.open(encoding="utf-8") as f:
                loaded = json.load(f)
            self.thresholds = {"svm": float(loaded.get("svm", 0.5)), "rf": float(loaded.get("rf", 0.5))}

    def _model(self, model: str):
        model = self._model_key(model)
        if model == "svm":
            clf = self.svm_pipeline
        elif model == "rf":
            clf = self.rf_pipeline
        else:
            raise ValueError(f"Unknown model: {model}")
        if clf is None:
            raise RuntimeError("Model is not trained or loaded")
        return clf

    @staticmethod
    def _model_key(model: str) -> str:
        if model == "svm":
            return "svm"
        if model in {"rf", "random_forest", "RandomForest"}:
            return "rf"
        raise ValueError(f"Unknown model: {model}")

    @staticmethod
    def _print_metrics(metrics: dict) -> None:
        print("\nEvaluation")
        print(f"{'metric':<12} {'SVM':>10} {'RandomForest':>14}")
        for key in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
            print(f"{key:<12} {metrics['SVM'][key]:>10.4f} {metrics['RandomForest'][key]:>14.4f}")

    @staticmethod
    def _plot_confusion_matrices(metrics: dict, out: Path) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (name, values) in zip(axes, metrics.items()):
            cm = np.asarray(values["confusion_matrix"])
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks([0, 1], labels=["Clean", "Stego"])
            ax.set_yticks([0, 1], labels=["Clean", "Stego"])
            threshold = cm.max() / 2 if cm.size else 0
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > threshold else "black",
                    )
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / "confusion_matrices.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_roc_curves(self, X_test: np.ndarray, y_test: np.ndarray, out: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, model_key in [("SVM", "svm"), ("RandomForest", "rf")]:
            proba = self._model(model_key).predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "roc_curves.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_feature_importance(self, out: Path) -> None:
        rf = self.rf_pipeline
        if isinstance(rf, CalibratedClassifierCV):
            estimators = [
                calibrated.estimator
                for calibrated in rf.calibrated_classifiers_
                if hasattr(calibrated.estimator, "feature_importances_")
            ]
            if not estimators:
                return
            importances = np.mean([est.feature_importances_ for est in estimators], axis=0)
        elif rf is not None and hasattr(rf, "feature_importances_"):
            importances = rf.feature_importances_
        else:
            return
        top = np.argsort(importances)[-25:][::-1]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(top)), importances[top])
        ax.set_title("Top Random Forest feature importances")
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Importance")
        ax.set_xticks(range(len(top)), labels=[str(i) for i in top], rotation=45)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
