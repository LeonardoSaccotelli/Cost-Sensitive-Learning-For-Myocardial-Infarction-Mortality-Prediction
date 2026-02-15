from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union

from imblearn.base import BaseSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTEN,
    SMOTENC,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    AllKNN,
    ClusterCentroids,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)


def get_resampling_pipeline(
    strategy_name: Optional[str],
    **kwargs: Any,
) -> Union[BaseSampler, str]:
    """
    Instantiate an imbalanced-learn sampler for a given resampling strategy.

    This factory returns a configured imblearn sampler instance (not a Pipeline),
    intended to be used directly as a resampling step inside an imblearn Pipeline.
    If ``strategy_name`` is ``None`` or ``'none'``, the function returns the literal
    string ``'passthrough'`` so the pipeline step becomes a no-op.

    Parameters
    ----------
    strategy_name : str or None
        Canonical sampler class name (case-sensitive), e.g. ``'SMOTE'``. If ``None``
        or ``'none'``, returns ``'passthrough'``.
    **kwargs : dict
        Keyword arguments forwarded to the sampler constructor. The accepted keys
        depend on the chosen sampler (e.g., ``sampling_strategy``, ``random_state``,
        ``k_neighbors``). Hybrid samplers may require passing pre-instantiated
        components (e.g., ``smote=SMOTE(...)``, ``enn=EditedNearestNeighbours(...)``,
        ``tomek=TomekLinks(...)``).

    Returns
    -------
    imblearn.base.BaseSampler or str
        Configured sampler instance, or the literal string ``'passthrough'``.

    Raises
    ------
    ValueError
        If ``strategy_name`` is not supported.

    Notes
    -----
    - The mapping of supported strategies is implemented via an internal registry.
    - Returning ``'passthrough'`` enables a uniform pipeline definition where the
      resampling step can be disabled without branching.

    Examples
    --------
    >>> from imblearn.pipeline import Pipeline as ImbPipeline
    >>> sampler = get_resampling_pipeline(
    ...     "SMOTE", sampling_strategy="auto", k_neighbors=5, random_state=42
    ... )
    >>> pipe = ImbPipeline([("resample", sampler), ("clf", ...)])

    >>> sampler = get_resampling_pipeline("none")
    >>> pipe = ImbPipeline([("resample", sampler), ("clf", ...)])
    """

    name = "none" if strategy_name is None else strategy_name

    registry: Dict[str, Type] = {
        # Undersampling
        "RandomUnderSampler": RandomUnderSampler,
        "NearMiss": NearMiss,
        "TomekLinks": TomekLinks,
        "EditedNearestNeighbours": EditedNearestNeighbours,
        "RepeatedEditedNearestNeighbours": RepeatedEditedNearestNeighbours,
        "AllKNN": AllKNN,
        "CondensedNearestNeighbour": CondensedNearestNeighbour,
        "OneSidedSelection": OneSidedSelection,
        "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule,
        "InstanceHardnessThreshold": InstanceHardnessThreshold,
        "ClusterCentroids": ClusterCentroids,
        # Oversampling
        "RandomOverSampler": RandomOverSampler,
        "SMOTE": SMOTE,
        "SMOTENC": SMOTENC,
        "SMOTEN": SMOTEN,
        "ADASYN": ADASYN,
        "BorderlineSMOTE": BorderlineSMOTE,
        "KMeansSMOTE": KMeansSMOTE,
        "SVMSMOTE": SVMSMOTE,
        # Hybrid
        "SMOTEENN": SMOTEENN,
        "SMOTETomek": SMOTETomek,
    }

    if name in {"none", "passthrough"}:
        return "passthrough"

    if name not in registry:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown resampler '{strategy_name}'. Supported: {supported}")

    sampler_cls = registry[name]
    return sampler_cls(**kwargs)
