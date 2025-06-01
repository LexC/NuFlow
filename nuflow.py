""" README
This script serves as the main interface to run the experimental analysis workflow.
It reads command-line arguments and calls the `main()` function in `analyses.py`.
"""

#%% === Libraries ===

import os
import sys
import argparse

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src import analyses
from utils import general_functions as gf

#%% === General Tools ===

# ---------- Variables ----------
def global_variables():
    """
    Defines and returns a dictionary of global variables used in the script.

    Returns:
        dict: A dictionary containing key configuration values and constants.
    """
    return {
    }

VAR = global_variables()

#%% === Functions ===

# ---------- Argument Parsing ----------
def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments containing the input path and paired flag.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to the instruction Excel file"
    )
    parser.add_argument(
        "-p", "--paired",
        action="store_true",
        help="Use paired statistical tests (Wilcoxon if applicable)"
    )
    return parser.parse_args()

#%% === Show Time ===

def main():
    """
    Main function to handle CLI inputs and execute the experimental analysis.
    """
    args = parse_arguments()
    analyses.main(args.input, paired=args.paired)

if __name__ == "__main__":
    main()
