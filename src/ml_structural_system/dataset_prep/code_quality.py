"""Map construction ``year`` to ``code_quality`` and back, per national seismic code.

Each country has a table of years when its seismic design codes came into force.
A building gets the label of the latest code in force in its construction year.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

CODE_YEAR_TABLES: dict[str, dict[int, str]] = {
    "costa_rica": {1974: "low_code", 1986: "medium_code", 2003: "high_code"},
    "guatemala": {1996: "low_code"},
    "dominican_republic": {1979: "low_code", 2011: "medium_code"},
}


def available_code_types() -> list[str]:
    """List the ``code_type`` keys with a configured year table.

    Returns:
        Sorted ``code_type`` names, e.g. ``"costa_rica"``.
    """
    return sorted(CODE_YEAR_TABLES)


def _thresholds(code_type: str) -> list[tuple[int, str]]:
    """Return the sorted (year, label) thresholds for ``code_type``.

    Raises:
        ValueError: If ``code_type`` has no configured year table.
    """
    if code_type not in CODE_YEAR_TABLES:
        raise ValueError(f"Unknown code_type '{code_type}'. Known types: {available_code_types()}")
    return sorted(CODE_YEAR_TABLES[code_type].items())


def _year_to_code_quality(year: float, thresholds: list[tuple[int, str]]) -> str | None:
    """Map one construction year to its ``code_quality`` label.

    Years before the first threshold map to ``"pre_code"``. A missing year maps
    to ``None``.
    """
    if pd.isna(year):
        return None
    quality = "pre_code"
    for threshold_year, label in thresholds:
        if year >= threshold_year:
            quality = label
    return quality


def year_series_to_code_quality(years: pd.Series, code_type: str) -> pd.Series:
    """Map each construction year to a ``code_quality`` label.

    Args:
        years: Construction years. Missing values map to ``None``.
        code_type: Key of the country year table to use.

    Returns:
        Labels such as ``"pre_code"``, ``"low_code"`` or ``"high_code"``.

    Raises:
        ValueError: If ``code_type`` is unknown.
    """
    thresholds = _thresholds(code_type)
    return years.map(lambda y: _year_to_code_quality(y, thresholds))


PRE_CODE_YEARS_BEFORE_FIRST = 20


def code_quality_to_year_series(qualities: pd.Series, code_type: str) -> pd.Series:
    """Map each ``code_quality`` label back to the first year of its era.

    ``"pre_code"`` maps to ``PRE_CODE_YEARS_BEFORE_FIRST`` years before the
    first threshold. Unknown labels map to NaN.

    Args:
        qualities: ``code_quality`` labels.
        code_type: Key of the country year table to use.

    Returns:
        A representative construction year for each label.

    Raises:
        ValueError: If ``code_type`` is unknown.
    """
    thresholds = _thresholds(code_type)
    label_to_year = {label: year for year, label in thresholds}
    if thresholds:
        label_to_year.setdefault("pre_code", thresholds[0][0] - PRE_CODE_YEARS_BEFORE_FIRST)
    return qualities.map(label_to_year)


def apply_code_quality_mapping(
    df: pd.DataFrame,
    feature_columns: list[str],
    code_type: str | None,
    source_label: str = "",
) -> pd.DataFrame:
    """Fill in ``code_quality`` from ``year``, or ``year`` from ``code_quality``.

    Only acts when one of the two is a needed feature, is missing from ``df``,
    and the other column is present. Logs a warning and returns ``df``
    unchanged if ``code_type`` is ``None``.

    Args:
        df: Building data.
        feature_columns: Feature columns the pipeline needs.
        code_type: Key of the country year table, or ``None`` if not configured.
        source_label: Source name used as a prefix in log messages.

    Returns:
        ``df``, or a copy of it with the derived column added.
    """
    prefix = f"[{source_label}] " if source_label else ""
    needs_code_quality = "code_quality" in feature_columns and "code_quality" not in df.columns
    needs_year = "year" in feature_columns and "year" not in df.columns

    if needs_code_quality and "year" in df.columns:
        if code_type is None:
            logger.warning(
                "%s'code_quality' is a needed feature and a 'year' column is present, "
                "but no code_type is configured for this source, skipping the mapping.",
                prefix,
            )
            return df
        df = df.copy()
        df["code_quality"] = year_series_to_code_quality(df["year"], code_type)
        n_mapped = int(df["code_quality"].notna().sum())
        logger.info(
            "%sMapped 'year' -> 'code_quality' (code_type='%s') for %d building(s).",
            prefix, code_type, n_mapped,
        )
    elif needs_year and "code_quality" in df.columns:
        if code_type is None:
            logger.warning(
                "%s'year' is a needed feature and a 'code_quality' column is present, "
                "but no code_type is configured for this source, skipping the mapping.",
                prefix,
            )
            return df
        df = df.copy()
        df["year"] = code_quality_to_year_series(df["code_quality"], code_type)
        n_mapped = int(df["year"].notna().sum())
        logger.info(
            "%sMapped 'code_quality' -> 'year' (code_type='%s') for %d building(s).",
            prefix, code_type, n_mapped,
        )

    return df
