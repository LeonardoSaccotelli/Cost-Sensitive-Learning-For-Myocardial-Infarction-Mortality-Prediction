# Cost-Sensitive Learning for Myocardial Infarction Mortality Prediction: A Static and Dynamic Ensemble Approach

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## 📖 Project Description
This project addresses the critical challenge of predict mortality outcomes due to myocardial infarction within highly **unbalanced dataset**. In the context of myocardial infarction mortality prediction, non-deadly outcomes vastly outnumber deadly outcomes, making standard machine learning approaches biased toward the majority class.

The core of this research is a comparative analysis between **Static Ensemble Learning** and **Dynamic Ensemble Selection (DES)** under **Cost-Sensitive-Learning** and **Cost-Sensitive-Evaluation** schemas. By evaluating how these models behave when the "cost" of misclassification is high, this project aims to identify the most robust architecture for myocardial infarction mortality.

### 🧪 Handling Class Imbalance
The framework implements and compare three main strategies to handle the class distribution during the training phase:
* **Baseline:** No Cost-Sensitive-Learning (algorithm-level) or Resampling (data-level), using the original imbalanced dataset. It serves as a standard cost-unaware benchmark approach.
* **Cost-Sensitive Learning:** Implementation of the `class_weight` parameter across models to assign a higher penalty to **DEAD** misclassifications, forcing the algorithms to prioritize the minority class.
   * _balanced_ : uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as `n_samples / (n_classes * np.bincount(y))`
   * _cost_sensitive_cost_matrix_ : use the business cost matrix following the rule `FN=10:FP=1 → DEAD=10, ALIVE=1` to adjust the class weights.
* **Resampling:** Fully support to `imbalanced-learn` resampling method to undersample the majority class or oversample the minority class.

The framework implements two main strategies to handle the class distribution during the evaluation phase:
* **Standard Threshold:** Models use a fixed 0.5 probability threshold.
* **Minimum Expected Cost (MEC):** The optimal threshold is determined based on the misclassification costs. For each prediction, we choose the class that minimizes the expected financial loss.

In addition to the standard classification metrics, the framework supports also the `Average Cost per Prediction` metric, which directly quantifies the real-world economic impact of the model's predictions.

Finally, the framework supports a combination between one training strategy and one evaluation strategy.

---

## 🔬 Core Objectives
1.  **Benchmarking:** Comparing the predictive power of static ensembles (fixed at training) against dynamic ensembles (which adaptively select the best model for each specific transaction) under **Cost-Sensitive-Learning** and **Cost-Sensitive-Evaluation** schemas.
2.  **Strategy Comparison:** Evaluating the trade-offs between "Cost-Sensitive-Learning" approaches vs. "Cost-Sensitive-Evaluation" approaches.
3.  **Modular Scalability:** Providing a codebase that can easily swap different techniques to find the optimal configuration for any imbalanced dataset.

---

## 🛠 Installation & Environment Setup

This project uses a `Makefile` to automate the setup process. Ensure you have **Python 3.10** installed on your system before proceeding.

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/Cost-Sensitive-Learning-For-Myocardial-Infarction-Mortality-Prediction.git](https://github.com/YourUsername/Cost-Sensitive-Learning-For-Myocardial-Infarction-Mortality-Prediction.git)
cd Cost-Sensitive-Learning-For-Myocardial-Infarction-Mortality-Prediction
```

### 2. Create the Virtual Environment
```bash
make create_environment
```

### 3. Activate the Environment
Based on your operating system, run the activation command:
- Linux/macOS:
```bash
source venv/bin/activate
```

- Windows (Command Prompt):
```bash
.\venv\Scripts\activate.bat
```

- Windows (PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
Once the environment is activated, install the required libraries (including the dynamic ensemble dependencies):
```bash
make requirements
```

---


## 🧹 Maintenance Commands

The Makefile also includes utility commands for project maintenance:


| Command                       | Description |
|-------------------------------| ------------- |
| ```make help ```              | Display a list of all available commands and their descriptions.|
| ```make lint ```              | Check code quality and formatting using Ruff. |
| ```make format ```            | Automatically fix linting issues and format code. |
| ```make clean ```             | Remove __pycache__ and compiled Python files  |
| ```make clean_environment ``` | Completely remove the venv directory. |
| ```make freeze ```            | Update the requirements.txt file with current environment state.|

---

## 📊 Data Workflow

The project manages data through a structured pipeline. Before running any analysis, you must retrieve the raw dataset.

### 1. External Dataset Acquisition
The first step is to download the "Myocardial infarction complications" dataset from UC Irvine Machine Learning Repository. This is handled automatically by the ```dataset.py``` script.

**Execution:**

```bash
make dataset
```

**What this command does:**
- Checks if the dataset already exists in ```data/external/```. 
- If missing, it uses ```ucimlrepo``` to download the ```fetch_ucirepo(id=579) ``` dataset. 
- Moves and renames the file to match the project's internal configuration.
 
**Related Configuration** (```myocardial_infarction_mortality/config.py```): The script relies on these path definitions. If you wish to change where data is stored, modify these variables:

| Variable                      | Default Value                             | Description                                              |
|-------------------------------|-------------------------------------------|----------------------------------------------------------|
| ```EXTERNAL_DATA_DIR ```      | ```PROJ_ROOT / "data" / "external"```     | The directory where external raw data is stored.         |
| ```EXTERNAL_FILENAME ```      | ```myocardial_infarction_external.csv```  | The final filename used by the project scripts.          |
| ```EXTERNAL_METADATA_FILENAME ```           | ```myocardial_infarction_metadata.txt```  | The name of the metadata file.                           |
| ```EXTERNAL_VARIABLES_FILENAME ```           | ```myocardial_infarction_variables.csv``` | The name of the file with variables list.                |
| ```EXTERNAL_DATASET_ID ```           | ```579```                                  |  UCI dataset ID to fetch. |
| ```EXTERNAL_FORCE_DOWNLOAD ```           | ```False```                                  |  If True, re-download and overwrite outputs even if they already exist. |

--- 
## 📉 Data Sampling & Reduction

The second step in the pipeline is to transition from the **External** (full) data to a **Raw** (subsampled) dataset.
The dataset provides 4 different time windows for the training of the machine learning models, based on the ICU time points: 
- the time of admission to hospital
- the end of the first day (24 hours after admission to the hospital)
- the end of the second day (48 hours after admission to the hospital)
- the end of the third day (72 hours after admission to the hospital)

**1. Subsampling Execution**


Use the following command to generate your subsampled dataset based on the current configuration:
```bash
make dataset_time_split
````
**What this command does:**

- Loads the full dataset from `data/external/`.
- Construct a time-slot-specific dataset by dropping unavailable features and adding a binary target (`LET_IS_BINARY`)
- Saves the resulting subset to `data/raw/myocardial_infarction_<selected_time_slot>.csv`.

**2. Sampling Configuration**

You can control how the data is reduced by modifying these variables in `myocardial_infarction_mortality/config.py`.


| Variable      | Default Value                       | Description |
|---------------|-------------------------------------|------------|
| ```EXTERNAL_DATA_DIR ```      | ```PROJ_ROOT / "data" / "external"``` | The directory where external raw data is stored.      |
| ```EXTERNAL_FILENAME ```      | ```myocardial_infarction_external.csv``` | The final filename used by the project scripts.       |
| ```RAW_DATA_DIR ``` | ```PROJ_ROOT / "data" / "raw"```    | The directory where raw data is stored.            |
| ```FILENAME_BASE ``` | ```myocardial_infarction```                     |  Base filename (without extension) used to build the output filename. |
| ```SELECTED_TIME_SLOT ```   | ```admission```                     |       Time slot key used to select which features to drop. Must be a key of``EXCLUDE_FEATURES_BY_SLOT`` (e.g., "admission", "day1", "day2", "day3").|


---

## 🧹 Data Cleaning & Deduplication
Once the raw sample is created, the next phase of the pipeline is cleaning. This script focuses on creating a clean time-slot-specific RAW dataset by filtering missingness, pruning extra targets, and removing duplicates.

**1. Cleaning Execution**

To process your raw data into a cleaned format, run:
```bash
make cleaning
````
**What this command does:**

- Loads the subsampled data from `data/raw/`.
- Row filtering: drop rows whose missingness fraction is strictly greater than ``threshold_drop_missing_rows``.
- Column filtering: drop columns whose missingness fraction is strictly greater than ``threshold_drop_missing_cols``. The selected ``target`` is never dropped here. The script logs both the number of dropped columns and their names.
- Target pruning: drop any columns listed in ``EXCLUDE_TARGETS`` if present, except the selected ``target``.
- Duplicate removal: remove duplicated rows using ``remove_duplicates`` with the current policy ``subset=None`` and ``keep="first"``. The script logs how many rows were removed.
- Saves the result to `data/interim/myocardial_infarction_<selected_time_slot>_<target>_cleaned.csv`

**2. Cleaning Configuration**

This script relies on the logic defined in `myocardial_infarction_mortality/config.py`. While the core logic is automated, the following paths are used:

| Variable                           | Default Value                        | Description                                                                                                                                    |
|------------------------------------|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| ```RAW_DATA_DIR ```                | ```PROJ_ROOT / "data" / "raw"```     | The directory where raw data is stored.                                                                                                        |
| ```FILENAME_BASE ```               | ```myocardial_infarction```          | Base filename (without extension) used to build the output filename.                                                                           |
| ```SELECTED_TIME_SLOT ```          | ```admission```                      | Time slot key used to select which features to drop. Must be a key of``EXCLUDE_FEATURES_BY_SLOT`` (e.g., "admission", "day1", "day2", "day3"). |
| ```INTERIM_DATA_DIR ```            | ```PROJ_ROOT / "data" / "interim"``` | The directory where interim data is stored.                                                                                                    |
| ```target ```                      | ```LET_IS_BINARY```                  | Target column that must exist and must be preserved.                                                                                           |
| ```threshold_drop_missing_rows ``` | ```0.20```                           | Drop rows with missingness strictly greater than this fraction.                                                                                                   |
| ```threshold_drop_missing_cols ``` | ```0.30```                           |  Drop columns with missingness strictly greater than this fraction.                                                                                                   |


---

## 🛠 Feature Engineering & Transformation
The final stage of the data pipeline converts cleaned data into a **Processed** dataset. Create the PROCESSED (features) dataset from the cleaned INTERIM dataset and standardize the target name.

**1. Feature Engineering Execution**

To generate the final features for your models, run:
```bash
make features
````

**What this command does:**

- The target column specified by ``target`` (default: ``"LET_IS_BINARY"``) is renamed to ``target_alias`` (default: ``"CLASS"``) to enforce a consistent target name for modeling.
- Saves the result to `data/processed/myocardial_infarction_<selected_time_slot>_<target>_features.csv`.

**2. Configuration & Parameters**

This script relies on the logic defined in `myocardial_infarction_mortality/config.py`.

| Variable                  | Default Value                          | Description                                                                                                                                    |
|---------------------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| ```INTERIM_DATA_DIR ```   | ```PROJ_ROOT / "data" / "interim"```   | The directory where interim data is stored.                                                                                                    |
| ```FILENAME_BASE ```      | ```myocardial_infarction```            | Base filename (without extension) used to build the output filename.                                                                           |
| ```SELECTED_TIME_SLOT ``` | ```admission```                        | Time slot key used to select which features to drop. Must be a key of``EXCLUDE_FEATURES_BY_SLOT`` (e.g., "admission", "day1", "day2", "day3"). |
| ```PROCESSED_DATA_DIR ``` | ```PROJ_ROOT / "data" / "processed"``` | The directory where processed data is stored.                                                                                                  |
| ```target ```             | ```LET_IS_BINARY```                    | Selected target column.                                                                                                                        |
| ```target_alias ```       | ```CLASS```                            | Standardized target name used inside the processed dataset (i.e., the column is renamed to this value).                                        |

---
## ⚙️ ️ Model Training & Evaluation
**1. The Cost Matrix & Decision Policy**

Traditional machine learning assumes all classification errors are equal. We override this using a specific clinical cost matrix where a **Missed Death (FN) is penalized 10x more heavily than a False Alarm (FP)**:
* **TN (True ALIVE predicted ALIVE):** Cost = 0
* **FP (True ALIVE predicted DEAD):** Cost = 1
* **FN (True DEAD predicted ALIVE):** Cost = 10
* **TP (True DEAD predicted DEAD):** Cost = 0

**2. Experimental Schemas**

The `EXPERIMENTS` dictionary defines multiple configurations to test which approach best handles the severe class imbalance:
* **Baseline:** Standard models using a default `0.5` decision threshold.
* **Cost-Sensitive Learning (CSL):** Adjusting `class_weight` (e.g., `{0: 1, 1: 10}`) directly within the model algorithms.
* **Data-Level Resampling:** Utilizing `SMOTE` or `SMOTEENN` to synthetically balance the training data distributions.
* **Minimum Expected Cost (MEC) Policy:** Shifting the final probability decision threshold to mathematically minimize the overall risk based on the cost matrix.

**3. Training**

The training pipeline uses a rigorous **Repeated Nested Cross-Validation** approach to ensure the models generalize reliably to unseen patients.

**4. Execution**
To execute the training pipeline based on the currently active `EXPERIMENT_ID` (e.g., `smoteenn_auto__mec_fp1_fn10`), run the following command:

```bash
make train
```

---
## 📓 Notebooks & In-depth Analysis
The `notebooks/` directory contains the experimental logs and visual analyses of the project. These are organized chronologically to mirror the research workflow.

**Data Understanding (DU) & Preparation (DP)**
- `1.0-ls-DU-data-quality-inspection`: Initial assessment of the external dataset to identify missing values, data types, and potential inconsistencies.
- `2.0-ls-DU-data-exploration`: Comprehensive Exploratory Data Analysis (EDA) focused on feature distributions and the extreme skewness of the target class.
- `3.0-ls-DP-data-exploration-cleaned-dataset`:  Assessment of the dataset quality after the cleaning step.

**Evaluation (EV) & Experimental Results**
- `4.0-ls-EV-feature-selection-analysis`: A post-hoc analysis of the feature selection process, identifying which variables were most frequently selected during hyperparameter tuning across different models.
- `5.0-ls-EV-models-evaluation`: Detailed performance breakdown for individual models within a specific experimental configuration.
- `5.1-ls-EV-models-evaluation-comparison`: Comparison of different models within the same experiment setting, utilizing qualitative plots, quantitative metrics, and **Resampled Corrected t-tests** for statistical significance.
- `5.2-ls-EV-experimental-schemas-comparison`: An "All-vs-All" comparison across different experiment settings to determine the optimal pipeline.

---
## License

This project is released under the [LICENSE](LICENSE). See the LICENSE file for details.