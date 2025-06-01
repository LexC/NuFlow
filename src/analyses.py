""" README
This script loads experimental data as specified in an instruction file. It
validates input paths, consolidates data from multiple spreadsheets, and appends
metadata for each experiment. All spreadsheets are converted using the utilities
provided in general_functions.py.
"""

#%% === Libraries ===

import os
import re 
import math

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.stats import f_oneway, kruskal, mannwhitneyu, wilcoxon


from utils import general_functions as gf

#%% === General Tools ===

# ---------- Variables ----------
def global_variables():
    """
    Defines and returns a dictionary of global variables used in the script.

    Returns:
        dict: A dictionary containing key configuration values and constants.
    """
    instructions_db_data_cols = ["data", "experiment"]
    instructions_db_config_cols = ["scores"]

    return {
        "instructions_db": {
            "data":{
                "tab":"Data",
                "cols": [gf.str_normalize(col) for col in instructions_db_data_cols]
                },
            "config":{
                "tab":"Configuration",
                "cols": [gf.str_normalize(col) for col in instructions_db_config_cols],
                "outputfolder": "output folder",
                }
        },
        "dividers":["|","_"],
        "outputs": {
            "grouped_data": "grouped_data.xlsx",
            "results": "results.xlsx",
            "images_folder": "images"           
        }
    }
VAR = global_variables()

# ---------- Support Functions ----------

def normalize_pd_header(df: pd.DataFrame,lower=False) -> pd.DataFrame:
    """
    Normalize all column headers in a DataFrame using `normalize_string`.

    Args:
        df (pd.DataFrame): DataFrame with original column names.

    Returns:
        pd.DataFrame: DataFrame with normalized column names.
    """
    df = df.copy()
    df.columns = [gf.str_normalize(col,lower=lower) for col in df.columns]
    return df

def normalize_pd_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize all string values in a specified column using `normalize_string`.

    Args:
        df (pd.DataFrame): DataFrame containing the column.
        column (str): Name of the column to normalize.

    Returns:
        pd.DataFrame: DataFrame with the normalized column values.

    Raises:
        KeyError: If the specified column does not exist.
    """
    if column not in df.columns:
        gf.message_error(f"Column '{column}' not found in DataFrame.")

    df = df.copy()
    df[column] = df[column].apply(gf.str_normalize)
    return df

def check_pd_columns(cols, required_cols):
    """
    Check for the presence of required columns in a list of columns.

    Args:
        cols (iterable): List of column names.
        required_cols (set): Required column names.

    Raises:
        RuntimeError: If any required columns are missing.
    """
    missing = required_cols - set(cols)
    if missing: gf.message_exit(f"Missing required columns: {', '.join(missing)}")

#%% === Functions ===

# ---------- Data ----------
def load_data(input: str) -> pd.DataFrame:
    """
    Load and validate the 'data' sheet from an instruction Excel file.

    Args:
        path (str): Path to the instruction Excel file.

    Returns:
        pd.DataFrame: Validated and normalized data instructions.

    Raises:
        RuntimeError: If the file cannot be read.
        KeyError: If required columns are missing.
    """

    df = gf.spreedsheet2dataframe(input,VAR["instructions_db"]["data"]["tab"],index_col=None)
    df = normalize_pd_header(df,lower=True)

    check_pd_columns(df.columns, set(VAR["instructions_db"]["data"]["cols"]))


    return df

def load_config(input: str) -> dict:
    """
    Load and normalize the 'config' sheet from an instruction Excel file,
    and return it as a dictionary mapping columns to non-None values.

    Args:
        input (str): Path to the instruction Excel file.

    Returns:
        dict: Dictionary of normalized configuration settings with non-None values.

    Raises:
        RuntimeError: If the file or sheet fails to load.
    """
    df = gf.spreedsheet2dataframe(input,VAR["instructions_db"]["config"]["tab"],index_col=None)
    df = normalize_pd_header(df,lower=True)

    check_pd_columns(df.columns, set(VAR["instructions_db"]["config"]["cols"]))
    
    config_dict = {
        col: [val for val in df[col] if pd.notna(val)]
        for col in df.columns
    }

    ouputfolder_tab = VAR["instructions_db"]["config"]["outputfolder"]

    config_dict[ouputfolder_tab] = gf.path_fix(config_dict[ouputfolder_tab][0])
    gf.create_folders(config_dict[ouputfolder_tab])

    return config_dict

def load_experiments(instructions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Read all spreadsheet data from paths specified in the instructions,
    appending an 'experiment' label to each.

    Args:
        instructions_df (pd.DataFrame): DataFrame containing experiment metadata.

    Returns:
        pd.DataFrame: Merged dataset of all experiments.

    Raises:
        RuntimeError: If any spreadsheet fails to load.
    """
    df_list = []

    for _, row in instructions_df.iterrows():
        data_path = row["data"]
        experiment_name = row["experiment"]
        try:
            data_df = gf.spreedsheet2dataframe(data_path)
            data_df = normalize_pd_header(data_df)
        except Exception as e:
            gf.message_error(f"Error loading '{data_path}': {e}")

        data_df["experiment"] = experiment_name
        df_list.append(data_df)
    
    return df_list

def merge_and_filter_dfs(df_list: list[pd.DataFrame], configs: dict) -> pd.DataFrame:
    """
    Merge a list of DataFrames while preserving indices and filter out rows
    not listed in configs['scores'].

    Args:
        df_list (list[pd.DataFrame]): List of DataFrames to merge.
        configs (dict): Configuration dictionary containing a 'scores' key
                        with a list of valid indices to keep.

    Returns:
        pd.DataFrame: The merged and filtered DataFrame.
    """

    merged_df = pd.concat(df_list, axis=0)
    merged_df = merged_df[merged_df['scores'].isin(configs.get('scores', []))]
    
    return parse_columns(merged_df)

def grouping_by_scores(experiments,configs):
    """
    Group experiment values by score for each relevant variable column.

    Args:
        experiments (pd.DataFrame): The experiment data including 'scores' and measured variables.
        configs (dict): Configuration dictionary including 'scores'.

    Returns:
        pd.DataFrame: A DataFrame with variables as columns and grouped score data as rows.
    """

    div = VAR["dividers"][0]
    scores = configs['scores']
    cols = [col for col in experiments.columns if div in col]

    result = {}

    for col in cols:
        grouped = {}
        for score in scores:
            if score in experiments['scores'].values:
                 grouped[score] = experiments[experiments['scores'] == score][col].dropna().to_numpy()
        result[col] = grouped

    return grouped_data_to_df(result)

def grouped_data_to_df(grouped_data: dict) -> pd.DataFrame:
    """
    Convert a nested dictionary of grouped NumPy arrays into a DataFrame.

    The outer keys become columns, the inner keys become the index, and each cell
    contains the group's values as a comma-separated string.

    Args:
        grouped_data (dict): Dictionary of variable -> {group -> np.ndarray}

    Returns:
        pd.DataFrame: Structured DataFrame with stringified array values.
    """
    records = {}

    for var, groups in grouped_data.items():
        col_data = {}
        for group, arr in groups.items():
            if isinstance(arr, np.ndarray):
                col_data[group] = arr
            else:
                col_data[group] = str(arr)
        records[var] = pd.Series(col_data)

    return pd.DataFrame(records)

#%% === Calculation ===

def statistical_analysis(df: pd.DataFrame, paired: bool = False) -> pd.DataFrame:
    """
    Perform statistical comparisons on a DataFrame where each cell contains a NumPy array.

    Args:
        df (pd.DataFrame): DataFrame with group labels as index and variables as columns,
                           each cell being a NumPy array of values.
        paired (bool): Whether to use paired statistical tests for 2-group comparisons.

    Returns:
        pd.DataFrame: DataFrame of p-values, with comparisons as index and variables as columns.
    """
    results = {}

    for var_name in df.columns:
        group_dict = df[var_name].dropna().to_dict()
        keys = list(group_dict.keys())
        comparisons = []

        for r in range(2, len(keys) + 1):
            for combo in combinations(keys, r):
                label = "_".join(combo)
                data_arrays = [group_dict[k] for k in combo if isinstance(group_dict[k], np.ndarray) and len(group_dict[k]) > 0]

                if len(data_arrays) != len(combo):
                    continue

                try:
                    if len(combo) == 2:
                        if paired and len(data_arrays[0]) == len(data_arrays[1]):
                            stat, pval = wilcoxon(data_arrays[0], data_arrays[1])
                        else:
                            stat, pval = mannwhitneyu(data_arrays[0], data_arrays[1], alternative='two-sided')
                    elif len(combo) <= 5:
                        stat, pval = kruskal(*data_arrays)
                    else:
                        stat, pval = f_oneway(*data_arrays)
                except Exception:
                    pval = np.nan

                comparisons.append((label, pval))

        results[var_name] = dict(comparisons)

    df_result = pd.DataFrame(results)
    gf.message_warning("Statistical correction for multiple comparisons was not implemented.")
    return df_result

#%% === Plots ===

def plot_swarm_box_grid(df: pd.DataFrame, configs: dict, grid_size: int = 5):
    """
    Save swarm + boxplot pages for each DataFrame column in a square grid layout.

    Args:
        df (pd.DataFrame): Each cell contains a NumPy array of values.
        grid_size (int): Number of rows/cols per page (e.g., 4 → 4x4 grid).
        configs (dict): Configuration dictionary including output folder path.
    """
    sns.set(style="whitegrid", palette="pastel", font_scale=1.0)
    xlabel = "Scores"
    div = VAR["dividers"][0]
    output_folder = configs[VAR["instructions_db"]["config"]["outputfolder"]]
    images_folder = os.path.join(output_folder,VAR["outputs"]["images_folder"])
    gf.create_folders(images_folder)

    cols = df.columns.tolist()
    plots_per_page = grid_size * grid_size
    num_pages = math.ceil(len(cols) / plots_per_page)

    print(f"\n{'-'*50}\nSaving plot images in {output_folder}{os.sep}")

    for page in range(num_pages):
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size))
        axes = axes.flatten()
        start = page * plots_per_page
        end = start + plots_per_page

        for i, column in enumerate(cols[start:end]):
            data = []

            for group in df.index:
                values = df.at[group, column]
                if isinstance(values, np.ndarray) and len(values) > 0:
                    data.extend([(group, val) for val in values])

            if not data:
                continue

            ylabel = re.split(r"\s*\|\s*", column)[-1].capitalize()
            temp_df = pd.DataFrame(data, columns=[xlabel, ylabel])

            ax = axes[i]
            sns.boxplot(x=xlabel, y=ylabel, data=temp_df, whis=1.5, showfliers=False, ax=ax)
            sns.swarmplot(x=xlabel, y=ylabel, data=temp_df, color=".25", size=5, ax=ax)

            ax.set_title(div.join(column.split(div)[:-1]), fontsize=12, weight='bold', pad=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Hide unused axes
        for j in range(i + 1, plots_per_page):
            fig.delaxes(axes[j])

        plt.tight_layout()

        filename = f"swarm_box_page_{page+1}.png"
        relative_path = os.path.join(VAR["outputs"]["images_folder"],filename)
        
        save_path = os.path.join(images_folder,filename)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

        print(gf.dotted_line_fill(relative_path,f"{page+1}/{num_pages}"))
    
    print(f"DONE: All images saved\n{'-'*50}")


#%% === Outputs ===

def export_experiments_to_excel(experiments: dict, configs: str) -> None:
    """
    Converts a grouped_data dictionary into a DataFrame and exports it to an Excel file.

    Args:
        grouped_data (dict): Dictionary of variable -> {group -> np.ndarray}
        file_path (str): Destination path for the Excel file.
    """

    df = experiments.copy()
    filename = os.path.join(
        configs[VAR["instructions_db"]["config"]["outputfolder"]],
        VAR["outputs"]["grouped_data"]
    )
    div = VAR["dividers"][0]
    
    cols = [col for col in df.columns if div in col]
    cols_metad = [col for col in df.columns if div not in col]
    reorder_cols = cols_metad+cols
    df = df[reorder_cols]

    df['scores'] = pd.Categorical(df['scores'], categories=configs['scores'], ordered=True)
    df = df.sort_values(by='scores')


    gf.safe_append_to_excel(filename,df,"columns")

def export_grouped_data_to_excel(grouped_data: dict, configs: str) -> None:
    """
    Export the experiments DataFrame to an Excel file, reordered and sorted by scores.

    Args:
        experiments (pd.DataFrame): The consolidated experiments DataFrame.
        configs (dict): Configuration dictionary including output folder and sorting info.
    """

    df = grouped_data.copy()
    filename = os.path.join(
        configs[VAR["instructions_db"]["config"]["outputfolder"]],
        VAR["outputs"]["grouped_data"]
    )
    
    for col in df.columns:
        df[col] = df[col].map(lambda x: ",".join(map(str, x)) if isinstance(x, np.ndarray) else str(x))

    gf.safe_append_to_excel(filename,df,"arrays")
    
def export_stats(stats,configs):
    """
    Export statistical analysis results to an Excel file.

    Args:
        stats (pd.DataFrame): DataFrame containing statistical results.
        configs (dict): Configuration dictionary with output paths.
    """
    df = stats.copy()
    filename = os.path.join(
        configs[VAR["instructions_db"]["config"]["outputfolder"]],
        VAR["outputs"]["results"]
    )

    gf.safe_append_to_excel(filename,df,"stats")

#%% === Parses ===

def parse_columns(experiments):
    """
    Rename columns in the DataFrame by parsing and formatting their structure.

    Args:
        experiments (pd.DataFrame): Input DataFrame with raw column names.

    Returns:
        pd.DataFrame: DataFrame with parsed and renamed columns.

    Raises:
        RuntimeError: If multiple column formats are detected.
    """
    div = VAR["dividers"][0]

    cols = [col for col in experiments.columns if div in col]

    numb_divs = np.unique([len(col.split(div)) for col in cols])
    if len(numb_divs) > 1: gf.message_exit("There are more then one type of experiment as data input")
    
    newcolumns = {}
    if numb_divs[0] == 2:
        for col in cols:
            parameter1,parameter2,method = parse1col(col, numb_divs[0])
            newcolumns[col] = f" {div} ".join([parameter1,parameter2,method])
    
    experiments = experiments.rename(columns=newcolumns)

    cols = sorted([col for col in experiments.columns if div in col])
    cols_metad = [col for col in experiments.columns if div not in col]
    reorder_cols = cols_metad + cols
    experiments = experiments[reorder_cols]

    return experiments
    
def parse1col(column_name, exptype):
    """
    Parse a column name string into meaningful component parts based on experiment type.

    Args:
        column_name (str): Raw column name string.
        exptype (int): Expected format type (2 or 3 segments).

    Returns:
        tuple: Extracted parts depending on format type.

    Raises:
        ValueError: If parsing fails due to unexpected format.
    """
    div0 = VAR["dividers"][0]
    div1 = VAR["dividers"][1]

    match exptype:
        case 2:
            expression,parts = column_name.split(div0)
            method,parenting = parts.split(div1)
            parenting = parenting.replace("gp","grandparent")
            return (expression,parenting,method)
            
        case 3:
            intensity1,intensity2,method = column_name.split(div0)
            return (intensity1,intensity2,method)
            
#%% === Show Time ===
def main(input:str, paired: bool = False):
    """
    Main function to parse command-line input, load instructions,
    and consolidate all experiment data.
    """
    instructions = load_data(input)
    configs = load_config(input)

    experiments = load_experiments(instructions)
    experiments = merge_and_filter_dfs(experiments,configs)
    grouped_data = grouping_by_scores(experiments,configs)

    stats = statistical_analysis(grouped_data, paired) 

    export_experiments_to_excel(experiments, configs)
    export_grouped_data_to_excel(grouped_data, configs)
    export_stats(stats,configs)

    plot_swarm_box_grid(grouped_data, configs)

    
#%%
if __name__ == "__main__":
    main()
