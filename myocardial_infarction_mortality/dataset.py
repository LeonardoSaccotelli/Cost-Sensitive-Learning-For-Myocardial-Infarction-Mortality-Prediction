from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd
import typer
from ucimlrepo import fetch_ucirepo

from myocardial_infarction_mortality.config import (
    EXTERNAL_DATA_DIR,
    EXTERNAL_DATASET_ID,
    EXTERNAL_FILENAME,
    EXTERNAL_FORCE_DOWNLOAD,
    EXTERNAL_METADATA_FILENAME,
    EXTERNAL_VARIABLES_FILENAME,
)

app = typer.Typer()


@app.command()
def main(
    filename_output_path: Path = EXTERNAL_DATA_DIR / EXTERNAL_FILENAME,
    metadata_output_path: Path = EXTERNAL_DATA_DIR / EXTERNAL_METADATA_FILENAME,
    variable_output_path: Path = EXTERNAL_DATA_DIR / EXTERNAL_VARIABLES_FILENAME,
    dataset_id: int = EXTERNAL_DATASET_ID,
    force: bool = EXTERNAL_FORCE_DOWNLOAD,
) -> None:
    """
    Ensure the UCI dataset is available locally in EXTERNAL, downloading it if missing.

    The script downloads the dataset via ``ucimlrepo.fetch_ucirepo(id=dataset_id)`` and writes:
    - the main dataset as CSV to ``filename_output_path`` using ``dataset.data.original``
    - metadata as plain TXT to ``metadata_output_path``
    - variables details as CSV to ``variable_output_path`` (from ``dataset.variables``)

    Parameters
    ----------
    filename_output_path : pathlib.Path, optional
        Destination path for the main dataset CSV. Defaults to
        ``EXTERNAL_DATA_DIR / EXTERNAL_FILENAME``.
    metadata_output_path : pathlib.Path, optional
        Destination path for the metadata TXT. Defaults to
        ``EXTERNAL_DATA_DIR / EXTERNAL_METADATA_FILENAME``.
    variable_output_path : pathlib.Path, optional
        Destination path for the variables details CSV. Defaults to
        ``EXTERNAL_DATA_DIR / EXTERNAL_VARIABLES_FILENAME``.
    dataset_id : int, optional
        UCI dataset ID to fetch. Defaults to ``EXTERNAL_DATASET_ID``.
    force : bool, optional
        If True, re-download and overwrite outputs even if they already exist. Defaults to
        ``EXTERNAL_FORCE_DOWNLOAD``.

    Returns
    -------
    None
        Side effects only (download + file writes).

    Raises
    ------
    typer.Exit
        Raised with exit code ``1`` if download or writing fails, or if the fetched object does
        not expose ``dataset.data.original`` as a non-empty pandas DataFrame, or if
        ``dataset.variables`` is not a non-empty pandas DataFrame.

    Examples
    --------
    Run using the module script (paths shown as examples)::

        python myocardial_infarction_mortality/dataset.py \
            --filename-output-path data/external/myocardial_infarction.csv \
            --metadata-output-path data/external/METADATA.txt \
            --variable-output-path data/external/VARIABLES.csv \
            --dataset-id 579 \
            --force
    """
    logger.info("Running myocardial_infarction_mortality/dataset.py ...")

    # Check if the dataset already exists or the user request to force dataset download
    need_fetch = (
        force
        or (not filename_output_path.exists())
        or (not metadata_output_path.exists())
        or (not variable_output_path.exists())
    )

    # If fetch is not request, return the control
    if not need_fetch:
        logger.info(f"Dataset ALREADY available at:\n\t{filename_output_path}")
        logger.info(f"Metadata ALREADY available at:\n\t{metadata_output_path}")
        logger.info(f"Variables ALREADY available at:\n\t{variable_output_path}")
        logger.success("Running myocardial_infarction_mortality/dataset.py COMPLETED!")
        return

    # If the dataset does not exist or user request to force dataset download
    try:
        # Ensure EXTERNAL_DATA_DIR exists
        EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Download the dataset from ucimlrepo
        logger.info(f"Downloading UCI dataset id={dataset_id} via ucimlrepo...")
        dataset: Any = fetch_ucirepo(id=dataset_id)

        data_obj = getattr(dataset, "data", None)
        if data_obj is None:
            raise ValueError("Fetched object has no `.data` attribute.")

        original = getattr(data_obj, "original", None)
        if not isinstance(original, pd.DataFrame) or original.empty:
            raise ValueError(
                "Expected the complete dataset as a non-empty pandas DataFrame in `dataset.data.original`."
            )

        # Write main dataset
        original.to_csv(filename_output_path, index=False, sep=",")
        logger.success(f"Saved main dataset to:\n\t{filename_output_path}")
        logger.info(f"Main dataset shape: {original.shape} (rows, cols)")

        # Write metadata as plain text
        metadata_obj = getattr(dataset, "metadata", None)
        metadata_output_path.write_text(str(metadata_obj), encoding="utf-8")
        logger.success(f"Saved metadata TXT to:\n\t{metadata_output_path}")

        # Write variables details as CSV
        variables_obj = getattr(dataset, "variables", None)
        if not isinstance(variables_obj, pd.DataFrame) or variables_obj.empty:
            raise ValueError("Expected `dataset.variables` to be a non-empty pandas DataFrame.")

        variables_obj.to_csv(variable_output_path, index=False, sep=",")
        logger.success(f"Saved variables CSV to:\n\t{variable_output_path}")

    except Exception as e:
        logger.error(f"Failed to fetch/write dataset id={dataset_id}: {e}")
        raise typer.Exit(code=1)

    logger.success("Running myocardial_infarction_mortality/dataset.py COMPLETED!")


if __name__ == "__main__":
    app()
