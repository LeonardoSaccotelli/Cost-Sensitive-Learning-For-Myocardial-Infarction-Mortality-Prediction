from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def load_dataset_with_schema(
    file_path: Union[str, Path],
    delimiter: str,
    header: Optional[int],
    index_col: Optional[Union[int, str]],
    float_features: list[str],
    int_features: list[str],
    categorical_nominal_features: list[str],
    categorical_ordinal_features: dict[str, list[int]],
) -> pd.DataFrame:
    """
    Load a CSV dataset and enforce dtypes according to a predefined schema.

    The function reads the dataset using `pandas.read_csv` and then enforces:
    - float features -> `numpy.float64`
    - integer features -> pandas nullable integer `"Int64"`
    - nominal categorical features -> pandas `"category"` (unordered)
    - ordinal categorical features -> pandas ordered `"category"` with explicit levels

    Only columns that exist in the loaded DataFrame are cast; missing columns are
    silently skipped.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the CSV file.
    delimiter : str
        Column delimiter passed to `pandas.read_csv` (`sep`).
    header : int or None
        Row number to use as column names. Use None if the file has no header row.
    index_col : int, str, or None
        Column to use as the row index (e.g., Patient ID). Use None to keep the
        default RangeIndex.
    float_features : list[str]
        Column names to cast to float64.
    int_features : list[str]
        Column names to cast to pandas nullable integer `"Int64"`.
    categorical_nominal_features : list[str]
        Column names to cast to pandas `"category"` dtype (unordered).
    categorical_ordinal_features : dict[str, list[int]]
        Mapping: ordinal column name -> ordered list of integer categories.

    Returns
    -------
    df : pandas.DataFrame
        Loaded DataFrame with schema casts applied where possible.

    Examples
    --------
    >>> df = load_dataset_with_schema(
    ...     file_path="data/raw/dataset.csv",
    ...     delimiter=",",
    ...     header=0,
    ...     index_col=0,
    ...     float_features=["K_BLOOD"],
    ...     int_features=["AGE"],
    ...     categorical_nominal_features=["SEX"],
    ...     categorical_ordinal_features={"FK_STENOK": [0, 1, 2, 3, 4]},
    ... )
    """
    df: pd.DataFrame = pd.read_csv(
        Path(file_path),
        sep=delimiter,
        header=header,
        index_col=index_col,
    )

    # Floats
    for col in float_features:
        if col in df.columns:
            df[col] = df[col].astype(np.float64)

    # Integers (nullable, supports NA)
    for col in int_features:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    # Nominal categorical (unordered)
    for col in categorical_nominal_features:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Ordinal categorical (ordered)
    for col, levels in categorical_ordinal_features.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=levels, ordered=True)

    return df


def save_dict_json(
    data: Mapping[str, Any],
    path: PathLike,
    *,
    mode: Literal["w", "a"] = "w",
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> None:
    """
    Serialize a mapping to JSON on disk (overwrite or JSON Lines append).

    In write mode (``mode='w'``), this function writes a single pretty-printed JSON
    object. In append mode (``mode='a'``), it appends one compact JSON object per
    line (NDJSON/JSON Lines). Parent directories are not created.

    Parameters
    ----------
    data : Mapping[str, Any]
        Dictionary-like object to serialize.
    path : str or pathlib.Path
        Destination file path. The parent directory must already exist.
    mode : {'w', 'a'}, default 'w'
        Output mode. If ``'w'``, overwrite with a single pretty-printed JSON object.
        If ``'a'``, append one JSON object per line (JSON Lines / NDJSON).
    ensure_ascii : bool, default False
        If ``False``, write UTF-8 characters as-is. If ``True``, escape non-ASCII
        characters.
    sort_keys : bool, default False
        If ``True``, sort dictionary keys in the output.

    Returns
    -------
    None
        This function writes to disk and returns nothing.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``{'w', 'a'}``.
    FileNotFoundError
        If the parent directory of ``path`` does not exist.

    Notes
    -----
    - Append mode writes JSON Lines (NDJSON), which is not a single valid JSON
      document. Use tools/readers that support line-delimited JSON.
    - This function does not create parent directories.

    Examples
    --------
    >>> save_dict_json({"run_id": 1, "score": 0.92}, "reports/metrics/run_1.json")

    >>> save_dict_json({"fold": 0, "ap": 0.88}, "reports/metrics/log.jsonl", mode="a")
    """

    if mode not in {"w", "a"}:
        raise ValueError('mode must be either "w" (overwrite) or "a" (append).')

    if mode == "w":
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, sort_keys=sort_keys, indent=2)
            f.write("\n")
    else:  # "a" → JSON Lines
        with path.open("a", encoding="utf-8") as f:
            line = json.dumps(
                data,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                separators=(",", ":"),  # compact for one-line records
            )
            f.write(line + "\n")
