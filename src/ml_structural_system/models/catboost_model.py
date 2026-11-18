"""Define the CatBoost pipeline.

Shared CV, final and test logic lives in ``models/base.py``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm

from ..gpu import catboost_gpu_kwargs, catboost_task_type
from ..progress import bar_disabled
from .base import BaseModelPipeline


class _TqdmCatBoostCallback:
    """Show a tqdm progress bar over CatBoost iterations.

    Passed to ``fit(callbacks=[...])``.
    """

    def __init__(self, n_iterations: int) -> None:
        """Initialize the callback.

        Args:
            n_iterations: Maximum number of iterations, used as the bar total.
        """
        self._n_iterations = n_iterations
        self._pbar: tqdm | None = None

    def after_iteration(self, info: Any) -> bool:
        """Advance the progress bar after each CatBoost iteration.

        Opens the bar on the first call.

        Args:
            info: Iteration info passed by CatBoost.

        Returns:
            True, so this callback never stops training.
        """
        if self._pbar is None:
            self._pbar = tqdm(
                total=self._n_iterations,
                desc="  catboost",
                leave=False,
                unit="iter",
                dynamic_ncols=True,
                disable=bar_disabled(),
            )
        self._pbar.update(1)
        return True  # True = continue training

    def __del__(self) -> None:
        """Close the progress bar on garbage collection, if still open."""
        if self._pbar is not None:
            self._pbar.close()


class CatBoostPipeline(BaseModelPipeline):
    """Pipeline for ``CatBoostClassifier`` with a multiclass loss.

    CV fits use early stopping on the validation fold. The final refit uses
    the mean best iteration from the winning CV run. Training runs on a GPU
    when available.

    Attributes:
        model_type: Identifier "CatBoost".
        param_grid: Grid over tree depth, learning rate and L2 leaf
            regularization.
    """

    model_type = "CatBoost"

    param_grid = {
        "depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "l2_leaf_reg": [1, 10],
    }

    def _class_weights_list(self, class_weight: dict | None) -> list[float] | None:
        """Convert a ``{class_code: weight}`` dict into the list CatBoost expects.

        The dict follows the convention of ``BaseModelPipeline.class_weights``
        and ``cost_matrix.class_weights``. Missing classes get weight 1.0.

        Args:
            class_weight: Optional ``{class_code: weight}`` mapping.

        Returns:
            Weights ordered by class code, or None if ``class_weight`` is None.
        """
        if class_weight is None:
            return None
        n_classes = len(self.cfg.class_names) if self.cfg.class_names else max(class_weight) + 1
        return [class_weight.get(i, 1.0) for i in range(n_classes)]

    def build_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted CatBoostClassifier for CV with early stopping.

        Args:
            params: Hyperparameters from ``param_grid``.
            class_weight: Optional ``{class_code: weight}`` mapping.

        Returns:
            Unfitted ``CatBoostClassifier`` with up to 1000 iterations.
        """
        return CatBoostClassifier(
            iterations=1000,
            early_stopping_rounds=50,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_state=self.cfg.random_state,
            class_weights=self._class_weights_list(class_weight),
            verbose=False,
            task_type=catboost_task_type(),
            **catboost_gpu_kwargs(),
            **params,
        )

    def fit_params(self, x_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Build extra keyword arguments passed to ``fit()`` during CV.

        Adds a progress bar callback when training on CPU.

        Args:
            x_val: Validation features used for early stopping.
            y_val: Validation labels used for early stopping.

        Returns:
            ``eval_set``, ``verbose`` and optionally ``callbacks`` keyword arguments.
        """
        params: dict[str, Any] = {"eval_set": (x_val, y_val), "verbose": False}
        if catboost_task_type() != "GPU":
            params["callbacks"] = [_TqdmCatBoostCallback(1000)]
        return params

    def build_final_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted CatBoostClassifier for the final refit.

        Uses the mean best iteration from CV as the number of iterations, or
        1000 when it is not available. No early stopping.

        Args:
            params: Best hyperparameters from CV.
            class_weight: Optional ``{class_code: weight}`` mapping.

        Returns:
            Unfitted ``CatBoostClassifier``.
        """
        n = getattr(self, "_final_n_estimators", None) or 1000
        return CatBoostClassifier(
            iterations=n,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_state=self.cfg.random_state,
            class_weights=self._class_weights_list(class_weight),
            verbose=False,
            task_type=catboost_task_type(),
            **catboost_gpu_kwargs(),
            **params,
        )
