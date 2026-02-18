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


def cast_to_float(X: Any) -> Any:
    """
    Cast input data to ``float64`` to ensure scikit-learn compatibility with missing values.

    This helper is primarily meant for Pandas nullable integer dtypes (e.g., ``Int64``),
    where missing values are represented as ``<NA>``. Many scikit-learn transformers expect
    numeric arrays where missing values are encoded as ``np.nan``. Casting to ``float64``
    ensures that nullables can be represented as ``np.nan`` and prevents downstream steps
    (imputers/scalers/encoders) from encountering dtype-related issues.

    Parameters
    ----------
    X : Any
        Array-like input (e.g., pandas DataFrame/Series or NumPy array) that supports
        ``X.astype(np.float64)``.

    Returns
    -------
    X_float : Any
        Same structure as ``X`` (depending on the input type) cast to ``float64``.
        Missing values are represented as ``np.nan`` where applicable.

    Raises
    ------
    TypeError
        If ``X`` does not implement ``astype`` or cannot be cast to ``np.float64``.
    ValueError
        If casting fails due to incompatible values.

    Notes
    -----
    This function is commonly used inside a scikit-learn ``FunctionTransformer`` as an early
    pipeline step, before imputation and scaling.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([1, None, 3], dtype="Int64")
    >>> out = cast_to_float(s)
    >>> str(out.dtype)
    'float64'
    """
    return X.astype(np.float64)


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
    Build numerical preprocessing pipelines (with explicit float casting).

    Two pipelines are returned:
    1) ``cast -> median imputation -> log1p -> standard scaling``
    2) ``cast -> median imputation -> standard scaling``

    The initial casting step converts pandas nullable numeric dtypes (e.g., ``Int64``) to
    ``float64`` so that missing values are represented as ``np.nan`` and downstream
    scikit-learn transformers (imputers/scalers) behave consistently.

    Returns
    -------
    pipeline_num_log_std : imblearn.pipeline.Pipeline
        Numerical pipeline with steps:
        - ``FunctionTransformer(cast_to_float)`` (casts to ``float64``)
        - ``SimpleImputer(strategy="median")``
        - ``FunctionTransformer(np.log1p)``
        - ``StandardScaler()``
    pipeline_num_std : imblearn.pipeline.Pipeline
        Numerical pipeline with steps:
        - ``FunctionTransformer(cast_to_float)`` (casts to ``float64``)
        - ``SimpleImputer(strategy="median")``
        - ``StandardScaler()``

    Raises
    ------
    TypeError
        If the input passed through the pipeline does not support casting via
        ``astype(np.float64)`` in ``cast_to_float``.
    ValueError
        If casting fails due to non-numeric values, or if ``log1p`` is applied to
        values where it is not defined (e.g., values < -1).

    Notes
    -----
    - Use the log1p pipeline only for features where a log transform is sensible
      (typically non-negative, right-skewed variables).
    - ``np.log1p`` is defined for values greater than ``-1``. If your feature can take
      values below ``-1``, remove the log step for that feature group.
    - The casting step is intentionally placed before imputation to ensure missing values
      are properly represented as ``np.nan``.

    Examples
    --------
    Fit/transform a numeric column with missing values::

        >>> import pandas as pd
        >>> X = pd.DataFrame({"x": [0.0, 1.0, 10.0, None]})
        >>> p_log, p_std = get_preprocessing_pipeline_numerical()
        >>> Xt = p_log.fit_transform(X[["x"]])
        >>> Xt.shape
        (4, 1)

    Handle pandas nullable integers (``Int64``) safely::

        >>> s = pd.Series([1, None, 3], dtype="Int64")
        >>> X = s.to_frame(name="x")
        >>> p_log, _ = get_preprocessing_pipeline_numerical()
        >>> Xt = p_log.fit_transform(X)
        >>> Xt.shape
        (3, 1)
    """

    caster = FunctionTransformer(cast_to_float, feature_names_out="one-to-one")

    # Pipeline: impute -> log1p -> standardize
    pipeline_num_log_std = Pipeline(
        steps=[
            ("cast", caster),
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline: impute -> standardize
    pipeline_num_std = Pipeline(
        steps=[
            ("cast", caster),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return pipeline_num_log_std, pipeline_num_std


def get_preprocessing_pipeline_categorical(
    config_cols: dict[str, Any],
) -> tuple[Pipeline, Pipeline, Pipeline, Pipeline]:
    """
    Build categorical preprocessing pipelines (with explicit float casting).

    This function creates and returns four pipelines, one per categorical subtype:

    1) Nominal categorical:
       ``cast -> most_frequent imputation -> one-hot encoding``
    2) Ordinal categorical:
       ``cast -> most_frequent imputation -> ordinal encoding (explicit category order)``
    3) Partial-ordinal categorical (domain mapping, e.g. ``ZSN_A``):
       ``cast -> most_frequent imputation -> custom mapping (1 col -> 2 engineered cols)``
    4) Binary categorical:
       ``cast -> most_frequent imputation``

    The initial casting step converts pandas nullable numeric dtypes (e.g., ``Int64``) to
    ``float64`` so that missing values are represented as ``np.nan`` and scikit-learn
    transformers behave consistently. This is especially important when categorical
    variables are stored as numeric codes in a pandas nullable dtype.

    Parameters
    ----------
    config_cols : dict[str, Any]
        Configuration dictionary defining which columns belong to each categorical group and,
        for ordinal variables, the explicit category ordering. Expected keys:

        - ``"cat_nominal"`` : list[str]
            Nominal categorical feature names.
        - ``"cat_ordinal"`` : list[str]
            Ordinal categorical feature names.
        - ``"cat_ordinal_order"`` : list[list[int]]
            Categories (ordering) for each ordinal feature in ``cat_ordinal``.
            Must satisfy ``len(cat_ordinal_order) == len(cat_ordinal)``.
        - ``"cat_partial_ordinal"`` : list[str]
            Categorical features that require custom mapping (e.g., ``["ZSN_A"]``).
        - ``"cat_binary"`` : list[str]
            Binary categorical feature names.

    Returns
    -------
    pipeline_cat_nominal : imblearn.pipeline.Pipeline
        Pipeline with steps:
        - ``FunctionTransformer(cast_to_float)`` (casts to ``float64``)
        - ``SimpleImputer(strategy="most_frequent")``
        - ``OneHotEncoder(handle_unknown="ignore", sparse_output=False)``
    pipeline_cat_ordinal : imblearn.pipeline.Pipeline
        Pipeline with steps:
        - ``FunctionTransformer(cast_to_float)`` (casts to ``float64``)
        - ``SimpleImputer(strategy="most_frequent")``
        - ``OrdinalEncoder(categories=config_cols["cat_ordinal_order"])``
    pipeline_cat_partial_ordinal : imblearn.pipeline.Pipeline
        Pipeline with steps:
        - ``FunctionTransformer(cast_to_float)`` (casts to ``float64``)
        - ``SimpleImputer(strategy="most_frequent")``
        - ``FunctionTransformer(map_zsn_a_logic)`` producing two engineered features
          (e.g., ``HF_right_line`` and ``HF_left_line``).
    pipeline_cat_binary : imblearn.pipeline.Pipeline
        Pipeline with steps:
        - ``FunctionTransformer(cast_to_float)`` (casts to ``float64``)
        - ``SimpleImputer(strategy="most_frequent")``

    Raises
    ------
    KeyError
        If any required key is missing from ``config_cols``.
    ValueError
        If ``len(config_cols["cat_ordinal"]) != len(config_cols["cat_ordinal_order"])``.
    TypeError
        If casting to ``float64`` fails inside ``cast_to_float`` for any input column.

    Notes
    -----
    - ``OneHotEncoder(handle_unknown="ignore")`` avoids inference-time failures when new
      categories appear in nominal features.
    - The partial-ordinal mapping expands dimensionality (1 input column -> 2 output columns).
    - ``OrdinalEncoder`` uses your provided category order; any value not present in the
      categories list may raise an error at transform-time unless handled upstream.

    Examples
    --------
    Build pipelines and transform example columns::

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
        >>> p_ord.fit_transform(df[["ord"]]).shape
        (2, 1)
        >>> p_part.fit_transform(df[["ZSN_A"]]).shape
        (2, 2)
        >>> p_bin.fit_transform(df[["bin"]]).shape
        (2, 1)

    Handle pandas nullable integer codes safely::

        >>> s = pd.Series([1, None, 0], dtype="Int64")
        >>> df = pd.DataFrame({"bin": s})
        >>> config = {
        ...     "cat_nominal": [],
        ...     "cat_ordinal": [],
        ...     "cat_ordinal_order": [],
        ...     "cat_partial_ordinal": [],
        ...     "cat_binary": ["bin"],
        ... }
        >>> _, _, _, p_bin = get_preprocessing_pipeline_categorical(config)
        >>> Xt = p_bin.fit_transform(df[["bin"]])
        >>> Xt.shape
        (3, 1)
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

    caster = FunctionTransformer(cast_to_float, feature_names_out="one-to-one")

    # 1) Nominal categorical: impute -> one-hot encode
    pipeline_cat_nominal = Pipeline(
        steps=[
            ("cast", caster),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 2) Ordinal categorical: impute -> ordinal encode with explicit ordering
    pipeline_cat_ordinal = Pipeline(
        steps=[
            ("cast", caster),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(categories=cat_ordinal_order)),
        ]
    )

    # 3) Partial ordinal (e.g., ZSN_A): impute -> map into two engineered columns
    pipeline_cat_partial_ordinal = Pipeline(
        steps=[
            ("cast", caster),
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
            ("cast", caster),
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
    Build the full preprocessing ``ColumnTransformer`` from an explicit column configuration.

    This function assembles the complete preprocessing graph by combining:
    - numerical pipelines (with casting, imputation, optional log1p, scaling)
    - categorical pipelines (with casting, imputation, and appropriate encoding/mapping)

    The returned ``ColumnTransformer`` applies each sub-pipeline to the corresponding
    column group defined in ``config_cols`` and either drops or passes through any
    remaining columns according to ``remainder``.

    Parameters
    ----------
    config_cols : dict[str, Any]
        Configuration dictionary defining which columns belong to each preprocessing
        group. Expected keys:

        - ``"num_log1p_standard_scaler"`` : list[str]
            Numerical features that will be cast to float, median-imputed, transformed
            via ``log1p``, then standardized.
        - ``"num_standard_scaler"`` : list[str]
            Numerical features that will be cast to float, median-imputed, then standardized
            (no log transform).
        - ``"cat_nominal"`` : list[str]
            Nominal categorical features to be imputed and one-hot encoded.
        - ``"cat_ordinal"`` : list[str]
            Ordinal categorical features to be imputed and ordinal-encoded using an explicit
            category ordering.
        - ``"cat_ordinal_order"`` : list[list[int]]
            Category ordering for each feature in ``"cat_ordinal"``. Must have the same length
            as ``"cat_ordinal"`` and be aligned by position.
        - ``"cat_partial_ordinal"`` : list[str]
            Features requiring domain-specific mapping via ``map_zsn_a_logic`` (e.g., ``["ZSN_A"]``).
            This mapping expands 1 column into 2 engineered columns.
        - ``"cat_binary"`` : list[str]
            Binary categorical features to be imputed (kept as numeric after casting).

    remainder : {'drop', 'passthrough'}, default 'drop'
        Behavior for columns not assigned to any transformer:
        - ``'drop'`` removes them from the output.
        - ``'passthrough'`` keeps them unchanged (appended after transformed columns).

    Returns
    -------
    preprocessor : sklearn.compose.ColumnTransformer
        A fitted-ready transformer that applies the configured preprocessing steps to each
        defined column group.

    Raises
    ------
    KeyError
        If any required key is missing from ``config_cols``.
    ValueError
        If the ordinal configuration is inconsistent (e.g., the number of ordinal columns does
        not match the number of provided category lists).
    TypeError
        If casting to ``float64`` fails inside ``cast_to_float`` for any configured column group.

    Notes
    -----
    - Casting (via ``cast_to_float``) is used throughout to ensure pandas nullable numeric dtypes
      (e.g., ``Int64`` with ``<NA>``) are converted to ``float64`` with proper ``np.nan`` handling.
    - The ``cat_partial_ordinal`` block can increase dimensionality (e.g., 1 input column -> 2 outputs).
    - If you need feature names after fitting, you can typically call
      ``preprocessor.get_feature_names_out()`` (availability and formatting depend on scikit-learn
      version and the transformers used).

    Examples
    --------
    Build and apply the preprocessor to a toy DataFrame::

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
        >>> pre = get_preprocessing_pipeline(config, remainder="drop")
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

    Keep unassigned columns with ``passthrough``::

        >>> config["num_standard_scaler"] = ["age"]
        >>> pre = get_preprocessing_pipeline(config, remainder="passthrough")
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