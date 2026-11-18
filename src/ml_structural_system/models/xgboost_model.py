"""Define the XGBoost pipeline.

Shared CV, final and test logic lives in ``models/base.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm
from xgboost.callback import TrainingCallback

from ..gpu import XGBClassifierCPUPredict, xgboost_device
from ..progress import bar_disabled
from .base import BaseModelPipeline


class _TqdmXGBCallback(TrainingCallback):
    """Show a tqdm progress bar over XGBoost boosting rounds.

    The bar closes when training ends, including on early stopping.
    """

    def __init__(self, n_estimators: int) -> None:
        """Initialize the callback.

        Args:
            n_estimators: Maximum number of boosting rounds, used as the bar total.
        """
        super().__init__()
        self._n_estimators = n_estimators
        self._pbar: tqdm | None = None

    def before_training(self, model: Any) -> Any:
        """Open the progress bar before training starts.

        Args:
            model: Booster being trained.

        Returns:
            The unchanged ``model``.
        """
        self._pbar = tqdm(
            total=self._n_estimators,
            desc="  xgb",
            leave=False,
            unit="round",
            dynamic_ncols=True,
            disable=bar_disabled(),
        )
        return model

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        """Advance the progress bar after each boosting round.

        Args:
            model: Booster being trained.
            epoch: Index of the finished round.
            evals_log: Evaluation history so far.

        Returns:
            False, so this callback never stops training.
        """
        if self._pbar is not None:
            self._pbar.update(1)
        return False  # False = do not stop early (XGBoost handles that itself)

    def after_training(self, model: Any) -> Any:
        """Close the progress bar once training completes.

        Args:
            model: Trained booster.

        Returns:
            The unchanged ``model``.
        """
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        return model


class XGBoostPipeline(BaseModelPipeline):
    """Pipeline for XGBoost's ``XGBClassifier``.

    CV fits use early stopping on the validation fold. The final refit uses
    the mean best iteration from the winning CV run. Training runs on a CUDA
    GPU when available.

    Attributes:
        model_type: Identifier "XGBoost".
        param_grid: Grid over tree depth, learning rate, minimum child weight
            and row and column subsampling.
    """

    model_type = "XGBoost"

    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [1, 5],  # higher = fewer splits on rare classes (regularization)
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    def _objective_kwargs(self) -> dict[str, Any]:
        """Choose the XGBoost objective and evaluation metric for this dataset.

        Returns:
            Softmax objective with multiclass log loss for more than two
            classes, otherwise logistic objective with log loss.
        """
        if (self.cfg.num_classes or 0) > 2:
            return {
                "objective": "multi:softprob",
                "num_class": self.cfg.num_classes,
                "eval_metric": "mlogloss",
            }
        return {"objective": "binary:logistic", "eval_metric": "logloss"}

    def build_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted XGBClassifier for CV with early stopping.

        Args:
            params: Hyperparameters from ``param_grid``.
            class_weight: Unused. Class weights are passed through
                ``sample_weight_for``.

        Returns:
            Unfitted ``XGBClassifierCPUPredict`` with up to 1000 rounds.
        """
        return XGBClassifierCPUPredict(
            **self._objective_kwargs(),
            n_estimators=1000,
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=self.cfg.random_state,
            device=xgboost_device(),
            callbacks=[_TqdmXGBCallback(1000)],
            **params,
        )

    def fit_params(self, x_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Build extra keyword arguments passed to ``fit()`` during CV.

        Args:
            x_val: Validation features used for early stopping.
            y_val: Validation labels used for early stopping.

        Returns:
            ``eval_set`` and ``verbose`` keyword arguments.
        """
        return {"eval_set": [(x_val, y_val)], "verbose": False}

    def build_final_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted XGBClassifier for the final refit.

        Uses the mean best iteration from CV as the number of rounds, or 1000
        when it is not available. No early stopping.

        Args:
            params: Best hyperparameters from CV.
            class_weight: Unused. Class weights are passed through
                ``sample_weight_for``.

        Returns:
            Unfitted ``XGBClassifierCPUPredict``.
        """
        n = getattr(self, "_final_n_estimators", None) or 1000
        return XGBClassifierCPUPredict(
            **self._objective_kwargs(),
            n_estimators=n,
            n_jobs=-1,
            random_state=self.cfg.random_state,
            device=xgboost_device(),
            callbacks=[_TqdmXGBCallback(n)],
            **params,
        )

    def sample_weight_for(self, y_train: pd.Series, class_weight: dict | None = None) -> np.ndarray | None:
        """Compute per-sample weights from class weights.

        Args:
            y_train: Training labels.
            class_weight: Optional ``{class_code: weight}`` mapping. If None,
                inverse-frequency weights are computed from ``y_train``.

        Returns:
            Array with the weight of each sample's class.
        """
        weights = class_weight if class_weight is not None else self.class_weights(y_train)
        return y_train.map(weights).to_numpy()
