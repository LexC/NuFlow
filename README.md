# NuFlow

## Overview

This project provides a modular pipeline for statistical analysis and visualization of experimental datasets defined through Excel-based instructions. It processes tabular experiment results, groups and filters them based on defined scores, and generates statistical comparisons and plots.

## Folder Structure

```
NuFlow/
├── data/
├── resources/
│   └── Instructions_template.xlsx
├── setting/
│   ├── miniconda_env_manager.sh
│   └── requirements.txt
├── src/
│   ├── __init__.py
│   └── analyses.py
├── utils/
│   ├── __init__.py
│   └── general_functions.py
├── LICENSE
├── README.md
└── nuflow.py
```

## Setting up

To set up the environment, you simply need to have Python installed and then install the required dependencies listed in `setting/requirements.txt`:

```bash
pip install -r setting/requirements.txt
```

### Miniconda Environment Manager

Optionally, you can use the provided script `miniconda_env_manager.sh` to:

* Install Miniconda (if not already installed)
* Create a Conda virtual environment
* Install the required packages within that environment

To launch the script bellow and follow the instructions.

```bash
bash setting/miniconda_env_manager.sh
```

After creating a virtual environment, you can activate it with:

```bash
conda activate <your-environment-name>
```

## Usage

### Command-Line Interface (CLI)

```bash
python nuflow.py -i resources/Instructions_template.xlsx -p
```

* `-i` or `--input`: Path to instruction file
* `-p` or `--paired`: Use paired tests (e.g., Wilcoxon)

### Filling the Instructions Excel File

To run the analysis properly, you must configure the Excel instruction file correctly:

#### Tab: `Data`

* **Column `Data`**: Provide the full path to the experiment file (e.g., a `.xlsx` or `.csv`).
* **Column `Experiment`**: An identifier for each experiment; this can be a string or a number.

#### Tab: `Configuration`

* **Column `Scores`**: List all subject groups/categories (e.g., `High`, `Low`, `HC`). Ensure your experiment data file has a column named exactly `Scores`.
* **Column `Output Folder`**: In the cell below this column header, provide the full path to the folder where result files and plots should be saved.

### ⚠️ Important Note on Statistical Analysis

The statistical results should be interpreted with caution. This tool performs multiple comparisons without applying corrections for multiple testing. It is intended to provide a **quick overview** of patterns and differences in the current data — not to serve as a definitive statistical conclusion.

I recommend using this output to guide further, more rigorous statistical analysis.

#### Statistical Methods Used

Depending on the number of groups and whether pairing is requested, the following statistical tests are applied:

* **Wilcoxon Signed-Rank Test**: For paired data with exactly 2 groups of equal size.
* **Mann-Whitney U Test**: For unpaired 2-group comparisons.
* **Kruskal-Wallis H Test**: For 3 to 5 group comparisons (non-parametric).
* **ANOVA (F-test)**: For more than 5 groups.

These are applied automatically based on data structure and CLI flags.

## Output

* Excel file with grouped experimental results
* Excel file with statistical test results
* PNG images of plots per metric in the `images` folder (relative to output path specified in config)

## Components

### 1. `src/analyses.py`

The core script handling the entire data flow. It:

* Reads experiment metadata and configuration from a spreadsheet
* Normalizes, loads, and merges datasets
* Performs score-based grouping and statistical analysis (paired and unpaired)
* Outputs grouped data and analysis results to Excel
* Generates visualizations (swarm + boxplots) for each metric

### 2. `nuflow.py`

Acts as the command-line entry point. It:

* Parses CLI arguments for input Excel path and statistical mode (paired)
* Invokes the `main()` function from `analyses.py`

### 3. `utils/general_functions.py`

A utility library supporting:

* File validation, path normalization
* Spreadsheet loading and Excel appending
* Logging, string normalization, and system-safe I/O

### 4. `setting/requirements.txt`

Defines the Python dependencies:

* pandas, openpyxl, xlrd
* numpy, scipy, seaborn, matplotlib

### 5. `setting/miniconda_env_manager.sh`

A bash tool to manage Conda environments:

* Install Miniconda
* Create/delete virtual environments
* Install requirements in a selected environment

### 6. `data/resources/Instructions_template.xlsx`

An example template file to define experiment data sources and configurations (like score labels).


## License

MIT License
