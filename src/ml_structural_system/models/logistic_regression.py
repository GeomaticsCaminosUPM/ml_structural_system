"""Define the logistic regression pipeline.

Shared CV, final and test logic lives in ``models/base.py``.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from .base import BaseModelPipeline


class LogisticRegressionPipeline(BaseModelPipeline):
    """Pipeline for L2-regularized multinomial logistic regression.

    Attributes:
        model_type: Identifier "LogisticRegression".
        param_grid: Grid over the inverse regularization strength ``C``.
    """

    model_type = "LogisticRegression"

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
    }

    def build_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted L2-regularized LogisticRegression.

        Args:
            params: Hyperparameters from ``param_grid``.
            class_weight: Optional ``{class_code: weight}`` mapping.

        Returns:
            Unfitted ``LogisticRegression``.
        """
        return LogisticRegression(
            max_iter=2000,
            random_state=self.cfg.random_state,
            class_weight=class_weight,
            l1_ratio=0,  # pure L2
            **params,
        )
