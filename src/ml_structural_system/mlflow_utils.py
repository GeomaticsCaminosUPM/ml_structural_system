"""Provide helpers to query MLflow runs and read back their parameters."""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd


def safe_cast_param(value: Any) -> Any:
    """Cast an MLflow param string back to int, float or None when possible.

    MLflow stores params as strings. ``"None"`` becomes None, integer strings
    become int and other numeric strings become float. Anything else, including
    non-string values, is returned unchanged.

    Args:
        value: Param value read from MLflow.

    Returns:
        The cast value, or ``value`` itself if it cannot be cast.
    """
    if not isinstance(value, str):
        return value
    if value == "None":
        return None
    try:
        if "." not in value and value.lstrip("-").isdigit():
            return int(value)
        return float(value)
    except ValueError:
        return value


def cast_params(params: dict[str, Any]) -> dict[str, Any]:
    """Apply :func:`safe_cast_param` to every value in a dict.

    Args:
        params: Param names mapped to their MLflow string values.

    Returns:
        A new dict with the same keys and cast values.
    """
    return {key: safe_cast_param(value) for key, value in params.items()}


def get_best_run(experiment_name: str, model_type: str, metric: str = "mean_f1_macro") -> pd.Series:
    """Return the best finished cross-validation (CV) run for a model type.

    Args:
        experiment_name: MLflow experiment to search.
        model_type: Value of the ``model_type`` param to filter runs by.
        metric: Metric name to rank runs by, highest first. The first column
            ending with this name is used.

    Returns:
        The best run as a row of :func:`mlflow.search_runs` output.

    Raises:
        RuntimeError: If no finished run exists for ``model_type`` or the
            metric is not logged.
    """
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"params.model_type = '{model_type}' and attributes.status = 'FINISHED'",
        output_format="pandas",
    )
    if len(runs) == 0:
        raise RuntimeError(
            f"No completed '{model_type}' runs found in experiment '{experiment_name}'. "
            "Check that run_cv() ran successfully for this model."
        )

    metric_col = next((c for c in runs.columns if c.endswith(metric)), None)
    if metric_col is None:
        raise RuntimeError(
            f"No '{metric}' metric found among runs for '{model_type}'. "
            "Check the metric name logged in run_cv()."
        )

    return runs.sort_values(metric_col, ascending=False).iloc[0]


def extract_params(run: pd.Series, allowed_keys: set[str]) -> dict[str, Any]:
    """Pull the ``params.*`` columns off an MLflow run and keep only allowed ones.

    Args:
        run: One run row from :func:`mlflow.search_runs`.
        allowed_keys: Param names to keep, without the ``params.`` prefix.

    Returns:
        Allowed param names mapped to their non-None values, still as strings.
    """
    all_params = {
        key.replace("params.", ""): value
        for key, value in run.items()
        if key.startswith("params.")
    }
    return {key: value for key, value in all_params.items() if key in allowed_keys and value is not None}
