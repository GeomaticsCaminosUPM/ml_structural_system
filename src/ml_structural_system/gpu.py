"""Detect a CUDA GPU and choose the XGBoost and CatBoost device settings.

Falls back to CPU when no usable GPU is found.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Check whether XGBoost and CatBoost can train on a CUDA GPU.

    Returns:
        True if CatBoost sees at least one GPU and the installed XGBoost was
        built with CUDA support, False otherwise (including on any error).
    """
    try:
        import xgboost
        from catboost.utils import get_gpu_device_count

        return get_gpu_device_count() > 0 and bool(xgboost.build_info().get("USE_CUDA"))
    except Exception:  # noqa: BLE001 - any failure means no usable GPU
        return False


def xgboost_device() -> str:
    """Return the ``device`` argument for XGBoost's ``XGBClassifier``.

    Returns:
        "cuda" if a GPU is available, else "cpu".
    """
    return "cuda" if cuda_available() else "cpu"


def catboost_task_type() -> str:
    """Return the ``task_type`` argument for ``CatBoostClassifier``.

    Returns:
        "GPU" if a GPU is available, else "CPU".
    """
    return "GPU" if cuda_available() else "CPU"


def catboost_gpu_kwargs() -> dict[str, float]:
    """Return extra CatBoost arguments that cap its GPU memory, if requested.

    CatBoost reserves 95% of the free GPU memory by default. Setting the
    environment variable ``CATBOOST_GPU_RAM_PART`` (a fraction between 0 and 1)
    caps that share, e.g. to keep memory for other programs. It does not change
    the trained models.

    Returns:
        ``{"gpu_ram_part": fraction}`` when training on the GPU with the
        variable set, otherwise an empty dict.
    """
    value = os.environ.get("CATBOOST_GPU_RAM_PART")
    if value and catboost_task_type() == "GPU":
        return {"gpu_ram_part": float(value)}
    return {}


from xgboost import XGBClassifier  # noqa: E402, already a hard dependency of this package


class XGBClassifierCPUPredict(XGBClassifier):
    """Train XGBoost on the configured device and predict on CPU.

    Drop-in replacement for ``XGBClassifier``. It trains on the ``device`` it
    was built with (usually ``xgboost_device()``) and switches to CPU after
    ``fit()``, so later predictions on CPU data do not need the GPU.
    """

    def fit(self, *args: Any, **kwargs: Any) -> XGBClassifierCPUPredict:
        """Fit the model, then switch it to CPU if it was trained on CUDA.

        Args:
            *args: Positional arguments for ``XGBClassifier.fit``.
            **kwargs: Keyword arguments for ``XGBClassifier.fit``.

        Returns:
            The fitted estimator.
        """
        super().fit(*args, **kwargs)
        if self.get_params().get("device") == "cuda":
            self.set_params(device="cpu")
        return self
