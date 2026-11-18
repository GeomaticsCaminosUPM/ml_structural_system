"""Define the balanced random forest pipeline.

Shared CV, final and test logic lives in ``models/base.py``.
"""

from __future__ import annotations

from typing import Any

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import BaseEstimator

from .base import BaseModelPipeline


class RandomForestPipeline(BaseModelPipeline):
    """Pipeline for imbalanced-learn's ``BalancedRandomForestClassifier``.

    Each tree is fit on a class-balanced bootstrap sample, so SMOTE
    strategies are not compared.

    Attributes:
        model_type: Identifier "RandomForest".
        cv_strategies: Only "baseline" and "class_weight" (plus
            "cost_sensitive" when a cost matrix is configured).
        param_grid: Grid over tree depth, minimum split size and features
            per split.
    """

    model_type = "RandomForest"

    # BalancedRandomForest already undersamples per tree, so no SMOTE strategies.
    cv_strategies = {
        "baseline":     {"resampler": None, "use_class_weight": False},
        "class_weight": {"resampler": None, "use_class_weight": True},
        # cost_sensitive added dynamically by __init__ when cost_matrix is set
    }

    param_grid = {
        # n_estimators fixed at 200 during CV (500 for final refit via build_final_model)
        "max_depth": [6, 12, None],
        "min_samples_split": [2, 8],
        "max_features": ["sqrt", 0.5],  # sqrt≈6 features/split vs 50% of 35
    }

    def build_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted 200-tree BalancedRandomForestClassifier for CV.

        Args:
            params: Hyperparameters from ``param_grid``.
            class_weight: Optional ``{class_code: weight}`` mapping.

        Returns:
            Unfitted ``BalancedRandomForestClassifier``.
        """
        return BalancedRandomForestClassifier(
            n_estimators=200,
            random_state=self.cfg.random_state,
            n_jobs=-1,
            class_weight=class_weight,
            **params,
        )

    def build_final_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted 500-tree BalancedRandomForestClassifier for the refit.

        Args:
            params: Best hyperparameters from CV.
            class_weight: Optional ``{class_code: weight}`` mapping.

        Returns:
            Unfitted ``BalancedRandomForestClassifier``.
        """
        return BalancedRandomForestClassifier(
            n_estimators=500,
            random_state=self.cfg.random_state,
            n_jobs=-1,
            class_weight=class_weight,
            **params,
        )
