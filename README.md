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

## License

This project is released under the [LICENSE](LICENSE). See the LICENSE file for details.