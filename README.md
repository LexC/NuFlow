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

## Usage

### Command-Line Interface (CLI)

```bash
python nuflow.py -i data/resources/Instructions_template.xlsx -p
```

* `-i` or `--input`: Path to instruction file
* `-p` or `--paired`: Use paired tests (e.g., Wilcoxon)

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
