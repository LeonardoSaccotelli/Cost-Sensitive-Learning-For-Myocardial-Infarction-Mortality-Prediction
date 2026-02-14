from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from myocardial_infarction_mortality.config import (
    CATEGORICAL_NOMINAL_FEATURES,
    CATEGORICAL_ORDINAL_FEATURES_WITH_ORDER,
    EXCLUDE_TARGETS,
    FILENAME_BASE,
    FLOAT_FEATURES,
    INT_FEATURES,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    SELECTED_TIME_SLOT,
)
from myocardial_infarction_mortality.utils.io_utils import load_dataset_with_schema

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / f"{FILENAME_BASE}_{SELECTED_TIME_SLOT}.csv",
    output_path: Path = RAW_DATA_DIR,
    filename_base: str = FILENAME_BASE,
    selected_time_slot: str = SELECTED_TIME_SLOT,
    target: str = "LET_IS_BINARY",
    threshold_drop_missing_rows: float = 0.20,
    threshold_drop_missing_cols: float = 0.30,
) -> None:
    """
       Clean the time-slot-specific dataset by removing high-missingness rows/columns and excluding extra targets.

       This script loads the time-slot dataset produced by
       ``myocardial_infarction_mortality/dataset_time_construction.py`` and applies:

       1) Row filtering: drop rows whose missingness fraction is strictly greater than
          ``threshold_drop_missing_rows``.
       2) Column filtering: drop columns whose missingness fraction is strictly greater than
          ``threshold_drop_missing_cols`` (the selected ``target`` is never dropped here).
       3) Target pruning: drop any columns listed in ``EXCLUDE_TARGETS`` if present, except the
          selected ``target``.
       4) Save the cleaned dataset to ``INTERIM_DATA_DIR`` as:
          ``{filename_base}_{selected_time_slot}_{target}_cleaned.csv``.

       Parameters
       ----------
       input_path : pathlib.Path, optional
           Input time-slot dataset path. Defaults to
           ``RAW_DATA_DIR / f"{FILENAME_BASE}_{SELECTED_TIME_SLOT}.csv"``.
       output_path : pathlib.Path, optional
           Unused path parameter kept for CLI compatibility with the current signature.
           The cleaned dataset is always written to ``INTERIM_DATA_DIR``.
       filename_base : str, optional
           Base filename used to build the output filename.
       selected_time_slot : str, optional
           Selected time slot key used in the output filename.
       target : str, optional
           Target column that must exist and must be preserved. Default is ``"LET_IS_BINARY"``.
       threshold_drop_missing_rows : float, optional
           Drop rows with missingness strictly greater than this fraction. Default is ``0.20``.
       threshold_drop_missing_cols : float, optional
           Drop columns with missingness strictly greater than this fraction. Default is ``0.30``.

       Returns
       -------
       None
           Side effects only (read, clean, write).

       Raises
       ------
       typer.Exit
           Raised with exit code ``1`` if:
           - ``input_path`` does not exist
           - ``target`` is missing
           - saving fails
       ValueError
           If thresholds are not in ``[0, 1]``.

       Examples
       --------
       Clean the admission dataset with defaults::

           python myocardial_infarction_mortality/cleaning.py \
               --selected-time-slot admission

       Clean and tune missingness thresholds::

           python myocardial_infarction_mortality/cleaning.py \
               --input-path data/raw/myocardial_infarction_admission.csv \
               --threshold-drop-missing-rows 0.15 \
               --threshold-drop-missing-cols 0.25 \
               --target LET_IS_BINARY
    """

    logger.info("Running myocardial_infarction_mortality/cleaning.py ...")

    if not (0.0 <= threshold_drop_missing_rows <= 1.0):
        raise ValueError("threshold_drop_missing_rows must be in [0, 1].")
    if not (0.0 <= threshold_drop_missing_cols <= 1.0):
        raise ValueError("threshold_drop_missing_cols must be in [0, 1].")

    # Preconditions
    if not input_path.exists():
        logger.error(f"Dataset not found at path:\n\t{input_path}")
        logger.error(
            "Run the construction time split step first, e.g.: `python myocardial_infarction_mortality/dataset_time_split.py`."
        )
        raise typer.Exit(code=1)

    # Load and basic checks
    logger.info(f"Loading RAW dataset at path:\n\t{input_path}")
    df: pd.DataFrame = load_dataset_with_schema(
        file_path=input_path,
        delimiter=",",
        header=0,
        index_col=0,
        float_features=FLOAT_FEATURES,
        int_features=INT_FEATURES,
        categorical_nominal_features=CATEGORICAL_NOMINAL_FEATURES,
        categorical_ordinal_features=CATEGORICAL_ORDINAL_FEATURES_WITH_ORDER,
    )
    logger.info(f"Input dataset shape: {df.shape} (rows, cols)")

    # Check that the target exists BEFORE dropping columns
    if target not in df.columns:
        logger.error(f"Target column '{target}' not found in the dataset.")
        raise typer.Exit(code=1)

    # Drop rows exceeding missingness threshold
    n_rows_before = df.shape[0]
    row_missing_frac = df.isna().mean(axis=1)
    rows_to_drop_mask = row_missing_frac > threshold_drop_missing_rows
    n_rows_to_drop = int(rows_to_drop_mask.sum())

    if n_rows_to_drop > 0:
        df = df.loc[~rows_to_drop_mask].copy()
        logger.warning(
            f"Dropped {n_rows_to_drop} rows with missingness > {threshold_drop_missing_rows:.2f} "
            f"(from {n_rows_before} to {df.shape[0]} rows)."
        )
    else:
        logger.info(
            f"No rows dropped for missingness > {threshold_drop_missing_rows:.2f} "
            f"({n_rows_before} rows kept)."
        )

    # Drop columns exceeding missingness threshold (but never drop the target)
    n_cols_before = df.shape[1]
    col_missing_frac = df.isna().mean(axis=0)
    cols_to_drop = col_missing_frac[col_missing_frac > threshold_drop_missing_cols].index.tolist()
    if target in cols_to_drop:
        cols_to_drop.remove(target)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.warning(
            f"Dropped {len(cols_to_drop)} columns with missingness > {threshold_drop_missing_cols:.2f} "
            f"(from {n_cols_before} to {df.shape[1]} cols)."
        )
    else:
        logger.info(
            f"No columns dropped for missingness > {threshold_drop_missing_cols:.2f} "
            f"({n_cols_before} cols kept)."
        )

    # Drop excluded targets (if present), but do not drop the selected target
    exclude_present = [c for c in EXCLUDE_TARGETS if c in df.columns and c != target]
    if exclude_present:
        df = df.drop(columns=exclude_present)
        logger.warning(
            f"Dropped {len(exclude_present)} columns from EXCLUDE_TARGETS: {exclude_present}"
        )
    else:
        logger.info(
            "No columns from EXCLUDE_TARGETS found to drop (or only the selected target was present)."
        )

    # Final sanity check: target still exists
    if target not in df.columns:
        logger.error(
            f"Target column '{target}' was removed during cleaning; this should not happen."
        )
        raise typer.Exit(code=1)

    # Ensure INTERIM_DATA_DIR exists
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename_output_path = INTERIM_DATA_DIR / f"{filename_base}_{selected_time_slot}_{target}_cleaned.csv"

    df.to_csv(filename_output_path, index=True, sep=",")

    # Store the cleaned dataset
    logger.success(f"Wrote INTERIM cleaned dataset to path:\n\t{output_path}")
    logger.info(f"Output dataset shape: {df.shape} (rows, cols)")

    logger.success("Running myocardial_infarction_mortality/dataset_time_split.py COMPLETED!")


if __name__ == "__main__":
    app()
