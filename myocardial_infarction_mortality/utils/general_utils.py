from __future__ import annotations

import pandas as pd


def subset_features_by_time_slot(
    df: pd.DataFrame,
    selected_time_slot: str,
    exclude_features_by_slot: dict[str, list[str]],
    warn_missing: bool = True,
) -> pd.DataFrame:
    """
    Validate a prediction time slot and subset dataframe columns accordingly.

    The dataset protocol defines multiple time moments (e.g., admission, day1, day2, day3).
    At each moment, some features are not available and must be excluded from the analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe. Patient ID is assumed to be the index (already set when loading).
    selected_time_slot : str
        Selected time slot identifier. Must be a key of `exclude_features_by_slot`.
    exclude_features_by_slot : dict[str, list[str]]
        Mapping time slot identifier -> list of feature names to exclude for that slot.
    warn_missing : bool, default True
        If True, prints a warning for excluded features that are not present in `df.columns`.

    Returns
    -------
    df_out : pandas.DataFrame
        Subset dataframe containing only columns usable for the selected time slot.

    Raises
    ------
    ValueError
        If `selected_time_slot` is not found in `exclude_features_by_slot`.

    Notes
    -----
    - Excluded columns not present in `df.columns` are ignored.
    - The returned dataframe is a copy.

    Examples
    --------
    >>> df_analysis = subset_features_by_time_slot(
    ...     df=df,
    ...     selected_time_slot="admission",
    ...     exclude_features_by_slot={"admission": ["R_AB_1_n", "R_AB_2_n", ...], "day1": ["R_AB_2_n", "R_AB_3_n", ...], ... }
    ... )
    """
    if selected_time_slot not in exclude_features_by_slot:
        allowed = sorted(exclude_features_by_slot.keys())
        raise ValueError(f"Invalid selected_time_slot={selected_time_slot!r}. Allowed: {allowed}")

    excluded = list(exclude_features_by_slot[selected_time_slot])

    if warn_missing:
        missing = sorted([c for c in excluded if c not in df.columns])
        if missing:
            print("[WARN] Excluded features not found in df.columns:")
            for c in missing:
                print(f"  - {c}")

    cols_to_drop = [c for c in excluded if c in df.columns]
    return df.drop(columns=cols_to_drop).copy()
