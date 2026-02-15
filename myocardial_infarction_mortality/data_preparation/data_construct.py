from __future__ import annotations

from typing import Any, Literal

import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


def map_zsn_a_logic(X: Any) -> np.ndarray:
    """
    Map ZSN_A stage codes into two engineered ordinal-like signals.

    This function expects a single-column input representing ``ZSN_A`` codes and
    maps each code to a 2D representation:

    - ``HF_right_line``
    - ``HF_left_line``

    Mapping
    -------
    0 -> [0, 0]
    1 -> [1, 1]
    2 -> [2, 1]
    3 -> [1, 2]
    4 -> [3, 3]
    other/unknown -> [NaN, NaN]

    Parameters
    ----------
    X : Any
        Array-like input of shape (n_samples, 1). It may be a pandas DataFrame/Series
        or a NumPy array. Only the first column is used.

    Returns
    -------
    mapped : numpy.ndarray of shape (n_samples, 2)
        Mapped representation with two engineered features. Unknown values are
        mapped to NaNs.

    Notes
    -----
    This function is designed to be used inside a scikit-learn ``FunctionTransformer``.
    It intentionally accepts "Any" because sklearn may pass a pandas object or a NumPy
    array depending on the upstream transformer.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0], [2], [4], [99]])
    >>> map_zsn_a_logic(X)
    array([[ 0.,  0.],
           [ 2.,  1.],
           [ 3.,  3.],
           [nan, nan]])
    """
    # Domain mapping: ZSN_A code -> (HF_right_line, HF_left_line)
    mapping: dict[int, tuple[float, float]] = {
        0: (0.0, 0.0),
        1: (1.0, 1.0),
        2: (2.0, 1.0),
        3: (1.0, 2.0),
        4: (3.0, 3.0),
    }

    # Support pandas DataFrame-like input or numpy input
    if hasattr(X, "iloc"):
        # pandas DataFrame/Series: take first column
        data = X.iloc[:, 0].to_numpy()
    else:
        # numpy: flatten to 1D
        data = np.asarray(X).ravel()

    # Build mapped output row by row
    out: list[tuple[float, float]] = []
    for val in data:
        # Handle NaN/None safely
        if val is None or (isinstance(val, float) and np.isnan(val)):
            out.append((np.nan, np.nan))
            continue

        # Convert to int if possible; if it fails -> unknown
        try:
            key = int(val)
        except (TypeError, ValueError):
            out.append((np.nan, np.nan))
            continue

        out.append(mapping.get(key, (np.nan, np.nan)))

    return np.asarray(out, dtype=float)


def get_preprocessing_pipeline_numerical() -> tuple[Pipeline, Pipeline]:
    """
    Build numerical preprocessing pipelines.

    Two pipelines are returned:
    1) Median imputation + log1p + standard scaling
    2) Median imputation + standard scaling

    Returns
    -------
    pipeline_num_log_std : imblearn.pipeline.Pipeline
        Pipeline with ``SimpleImputer(strategy="median")`` ->
        ``FunctionTransformer(np.log1p)`` -> ``StandardScaler()``.
    pipeline_num_std : imblearn.pipeline.Pipeline
        Pipeline with ``SimpleImputer(strategy="median")`` -> ``StandardScaler()``.

    Notes
    -----
    Use the log1p pipeline only for non-negative features where log1p is meaningful.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({"x": [0.0, 1.0, 10.0, None]})
    >>> p_log, p_std = get_preprocessing_pipeline_numerical()
    >>> Xt = p_log.fit_transform(X[["x"]])
    >>> Xt.shape
    (4, 1)
    """
    # Pipeline: impute -> log1p -> standardize
    pipeline_num_log_std = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline: impute -> standardize
    pipeline_num_std = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return pipeline_num_log_std, pipeline_num_std


def get_preprocessing_pipeline_categorical(
    config_cols: dict[str, Any],
) -> tuple[Pipeline, Pipeline, Pipeline, Pipeline]:
    """
    Build categorical preprocessing pipelines.

    Parameters
    ----------
    config_cols : dict[str, Any]
        Configuration dictionary. Expected keys:
        - ``"cat_nominal"`` : list[str]
        - ``"cat_ordinal"`` : list[str]
        - ``"cat_ordinal_order"`` : list[list[int]]
        - ``"cat_partial_ordinal"`` : list[str]
        - ``"cat_binary"`` : list[str]

    Returns
    -------
    pipeline_cat_nominal : imblearn.pipeline.Pipeline
        Most-frequent imputation + one-hot encoding.
    pipeline_cat_ordinal : imblearn.pipeline.Pipeline
        Most-frequent imputation + ordinal encoding with configured categories.
    pipeline_cat_partial_ordinal : imblearn.pipeline.Pipeline
        Most-frequent imputation + custom mapping via ``map_zsn_a_logic``.
    pipeline_cat_binary : imblearn.pipeline.Pipeline
        Most-frequent imputation only (keeps binary values as-is).

    Raises
    ------
    KeyError
        If required keys are missing from ``config_cols``.
    ValueError
        If ``len(cat_ordinal) != len(cat_ordinal_order)``.

    Notes
    -----
    - ``OneHotEncoder(handle_unknown="ignore")`` prevents errors at inference when unseen
      categories appear.
    - The partial-ordinal transformer expands one input column into two output columns.

    Examples
    --------
    >>> import pandas as pd
    >>> config = {
    ...     "cat_nominal": ["nom"],
    ...     "cat_ordinal": ["ord"],
    ...     "cat_ordinal_order": [[0, 1, 2]],
    ...     "cat_partial_ordinal": ["ZSN_A"],
    ...     "cat_binary": ["bin"],
    ... }
    >>> p_nom, p_ord, p_part, p_bin = get_preprocessing_pipeline_categorical(config)
    >>> df = pd.DataFrame({"nom": ["a", "b"], "ord": [0, 2], "ZSN_A": [1, 4], "bin": [1, 0]})
    >>> p_nom.fit_transform(df[["nom"]]).shape
    (2, 2)
    >>> p_part.fit_transform(df[["ZSN_A"]]).shape
    (2, 2)
    """
    # Validate required keys exist
    required_keys = {
        "cat_nominal",
        "cat_ordinal",
        "cat_ordinal_order",
        "cat_partial_ordinal",
        "cat_binary",
    }
    missing = sorted(required_keys.difference(config_cols.keys()))
    if missing:
        raise KeyError(f"config_cols is missing required keys: {missing}")

    cat_ordinal = config_cols["cat_ordinal"]
    cat_ordinal_order = config_cols["cat_ordinal_order"]

    # Validate ordinal configuration consistency
    if len(cat_ordinal) != len(cat_ordinal_order):
        raise ValueError(
            "cat_ordinal_order must have the same length as cat_ordinal "
            f"(got {len(cat_ordinal_order)} vs {len(cat_ordinal)})."
        )

    # 1) Nominal categorical: impute -> one-hot encode
    pipeline_cat_nominal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 2) Ordinal categorical: impute -> ordinal encode with explicit ordering
    pipeline_cat_ordinal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(categories=cat_ordinal_order)),
        ]
    )

    # 3) Partial ordinal (e.g., ZSN_A): impute -> map into two engineered columns
    pipeline_cat_partial_ordinal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "partial_map",
                FunctionTransformer(
                    func=map_zsn_a_logic,
                    validate=False,
                    feature_names_out=lambda transformer, input_features: [
                        "HF_right_line",
                        "HF_left_line",
                    ],
                ),
            ),
        ]
    )

    # 4) Binary categorical: impute only (keeps values as-is)
    pipeline_cat_binary = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return (
        pipeline_cat_nominal,
        pipeline_cat_ordinal,
        pipeline_cat_partial_ordinal,
        pipeline_cat_binary,
    )


def get_preprocessing_pipeline(
    config_cols: dict[str, Any],
    *,
    remainder: Literal["drop", "passthrough"] = "drop",
) -> ColumnTransformer:
    """
    Build the full preprocessing ``ColumnTransformer`` from a configuration dictionary.

    Parameters
    ----------
    config_cols : dict[str, Any]
        Configuration dictionary. Expected keys:
        - ``"num_log1p_standard_scaler"`` : list[str]
        - ``"num_standard_scaler"`` : list[str]
        - ``"cat_nominal"`` : list[str]
        - ``"cat_ordinal"`` : list[str]
        - ``"cat_ordinal_order"`` : list[list[int]]
        - ``"cat_partial_ordinal"`` : list[str]
        - ``"cat_binary"`` : list[str]
    remainder : {'drop', 'passthrough'}, default 'drop'
        What to do with columns not assigned to any transformer.

    Returns
    -------
    preprocessor : sklearn.compose.ColumnTransformer
        A transformer that applies the configured preprocessing steps to each column group.

    Raises
    ------
    KeyError
        If required keys are missing from ``config_cols``.

    Notes
    -----
    - The ``cat_partial_ordinal`` transformer can increase dimensionality (e.g., 1 column -> 2 columns).
    - Depending on scikit-learn version and transformers, you may call
      ``preprocessor.get_feature_names_out()`` after fitting.

    Examples
    --------
    >>> import pandas as pd
    >>> config = {
    ...     "num_log1p_standard_scaler": ["x"],
    ...     "num_standard_scaler": ["age"],
    ...     "cat_nominal": ["nom"],
    ...     "cat_ordinal": ["ord"],
    ...     "cat_ordinal_order": [[0, 1, 2]],
    ...     "cat_partial_ordinal": ["ZSN_A"],
    ...     "cat_binary": ["bin"],
    ... }
    >>> pre = get_preprocessing_pipeline(config)
    >>> df = pd.DataFrame({
    ...     "x": [0.0, 10.0, None],
    ...     "age": [60, 70, 80],
    ...     "nom": ["a", "b", "a"],
    ...     "ord": [0, 2, 1],
    ...     "ZSN_A": [1, 4, 0],
    ...     "bin": [1, 0, 1],
    ... })
    >>> Xt = pre.fit_transform(df)
    >>> Xt.shape[0]
    3
    """
    # Validate required keys exist
    required_keys = {
        "num_log1p_standard_scaler",
        "num_standard_scaler",
        "cat_nominal",
        "cat_ordinal",
        "cat_ordinal_order",
        "cat_partial_ordinal",
        "cat_binary",
    }
    missing = sorted(required_keys.difference(config_cols.keys()))
    if missing:
        raise KeyError(f"config_cols is missing required keys: {missing}")

    # Build numerical pipelines
    pipeline_num_log_std, pipeline_num_std = get_preprocessing_pipeline_numerical()

    # Build categorical pipelines
    (
        pipeline_cat_nominal,
        pipeline_cat_ordinal,
        pipeline_cat_partial_ordinal,
        pipeline_cat_binary,
    ) = get_preprocessing_pipeline_categorical(config_cols)

    # Route each feature group to the correct pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num_log1p_standard_scaler",
                pipeline_num_log_std,
                config_cols["num_log1p_standard_scaler"],
            ),
            (
                "num_standard_scaler",
                pipeline_num_std,
                config_cols["num_standard_scaler"],
            ),
            ("cat_nominal", pipeline_cat_nominal, config_cols["cat_nominal"]),
            ("cat_ordinal", pipeline_cat_ordinal, config_cols["cat_ordinal"]),
            ("cat_partial_ordinal", pipeline_cat_partial_ordinal, config_cols["cat_partial_ordinal"]),
            ("cat_binary", pipeline_cat_binary, config_cols["cat_binary"]),
        ],
        remainder=remainder,
    )

    return preprocessor