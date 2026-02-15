from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from myocardial_infarction_mortality.config import (
    CATEGORICAL_NOMINAL_FEATURES,
    CATEGORICAL_ORDINAL_FEATURES_WITH_ORDER,
    FILENAME_BASE,
    FLOAT_FEATURES,
    INT_FEATURES,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    SELECTED_TIME_SLOT,
)
from myocardial_infarction_mortality.utils.io_utils import load_dataset_with_schema

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR
    / f"{FILENAME_BASE}_{SELECTED_TIME_SLOT}_LET_IS_BINARY_cleaned.csv",
    output_path: Path = PROCESSED_DATA_DIR,
    filename_base: str = FILENAME_BASE,
    selected_time_slot: str = SELECTED_TIME_SLOT,
    target: str = "LET_IS_BINARY",
    target_alias: str = "CLASS",
) -> None:
    """
    Create the PROCESSED (features) dataset from the cleaned INTERIM dataset and standardize the target name.

    This script loads the cleaned dataset produced by ``myocardial_infarction_mortality/cleaning.py``,
    performs minimal feature-engineering steps, and writes the resulting dataset to
    ``PROCESSED_DATA_DIR``.

    Current behavior:
    - The target column specified by ``target`` (default: ``"LET_IS_BINARY"``) is renamed to
      ``target_alias`` (default: ``"CLASS"``) to enforce a consistent target name for modeling.
    - The output filename intentionally includes the *original* target name (``{target}``) to
      keep traceability across datasets that may use different target definitions.

    The output file is saved as:
    ``{filename_base}_{selected_time_slot}_{target}_features.csv`` in ``output_path``.

    Parameters
    ----------
    input_path : pathlib.Path, optional
        Path to the cleaned INTERIM dataset CSV. Defaults to
        ``INTERIM_DATA_DIR / f"{FILENAME_BASE}_{SELECTED_TIME_SLOT}_LET_IS_BINARY_cleaned.csv"``.
    output_path : pathlib.Path, optional
        Destination directory where the processed dataset will be written. Defaults to
        ``PROCESSED_DATA_DIR``.
    filename_base : str, optional
        Base filename used to build the output filename.
    selected_time_slot : str, optional
        Selected ICU time slot used in the output filename (e.g., "admission", "day1", "day2", "day3").
    target : str, optional
        Name of the target column expected in the cleaned dataset. This name is also used in the
        output filename for traceability. Default is ``"LET_IS_BINARY"``.
    target_alias : str, optional
        Standardized target name used inside the processed dataset (i.e., the column is renamed to
        this value). Default is ``"CLASS"``.

    Returns
    -------
    None
        Side effects only (read CSV, rename column, write CSV, logging).

    Raises
    ------
    typer.Exit
        Raised with exit code ``1`` if:
        - ``input_path`` does not exist
        - ``target`` is missing from the dataset
        - ``target_alias`` already exists (to avoid overwriting)
        - writing fails

    Notes
    -----
    - This step does not currently perform scaling, encoding, or imputation. It only standardizes
      the target column name and persists the dataset for downstream modeling.
    - The dataset index is preserved (the script writes with ``index=True``), consistent with
      reading via ``index_col=0`` in ``load_dataset_with_schema``.

    Examples
    --------
    Run with defaults (paths/time slot from config)::

        python myocardial_infarction_mortality/features.py

    Use a different cleaned input file and keep traceability in the output filename::

        python myocardial_infarction_mortality/features.py \
            --input-path data/interim/myocardial_infarction_day2_LET_IS_BINARY_cleaned.csv \
            --selected-time-slot day2 \
            --target LET_IS_BINARY

    Standardize a different target name to ``CLASS``::

        python myocardial_infarction_mortality/features.py \
            --target SOME_OTHER_TARGET \
            --target-alias CLASS
    """

    logger.info("Running myocardial_infarction_mortality/features.py ...")

    # Preconditions
    if not input_path.exists():
        logger.error(f"Dataset not found at path:\n\t{input_path}")
        logger.error(
            "Run the cleaning step first, e.g.: `python myocardial_infarction_mortality/cleaning.py`."
        )
        raise typer.Exit(code=1)

    # Load and basic checks
    logger.info(f"Loading CLEANED dataset at path:\n\t{input_path}")
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

    logger.info("Starting feature engineering...")

    ################################# RENAME TARGET IN "CLASS" #################################
    if target not in df.columns:
        logger.error(f"Target column '{target}' not found. Cannot rename to '{target_alias}'.")
        raise typer.Exit(code=1)

    if target_alias in df.columns and target_alias != target:
        logger.error(
            f"Target alias '{target_alias}' already exists in the dataset. "
            f"Refusing to overwrite it (target='{target}')."
        )
        raise typer.Exit(code=1)

    df.rename(columns={target: target_alias}, inplace=True)
    logger.success(f"Renamed target '{target}' -> '{target_alias}'")

    # Ensure PROCESSED_DATA_DIR exists
    output_path.mkdir(parents=True, exist_ok=True)
    filename_output_path = (
        output_path / f"{filename_base}_{selected_time_slot}_{target}_features.csv"
    )

    df.to_csv(filename_output_path, index=True, sep=",")

    logger.info(f"Output dataset shape: {df.shape} (rows, cols)")
    logger.success(f"Wrote FEATURES dataset to path:\n\t{filename_output_path}")

    logger.success("Running myocardial_infarction_mortality/features.py COMPLETED!")


if __name__ == "__main__":
    app()
