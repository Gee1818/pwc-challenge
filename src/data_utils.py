"""
Data loading and utilities functions.
"""

import polars as pl
from config import DEFAULT_DATA_PATH, COLAB_DATA_PATH


def load_data(data_path=DEFAULT_DATA_PATH):
    """Load all required datasets."""
    df_salary = pl.read_csv(f"{data_path}salary.csv")
    df_people = pl.read_csv(f"{data_path}people.csv")
    df_descriptions = pl.read_csv(f"{data_path}descriptions.csv")
    return df_salary, df_people, df_descriptions


def load_data_environment():
    """Load data based on environment (Colab or local)."""
    try:
        # Check if running in Google Colab
        if 'google.colab' in str(get_ipython()):
            from google.colab import drive
            drive.mount('/content/drive')
            return load_data(data_path=COLAB_DATA_PATH)
        else:
            return load_data(data_path=DEFAULT_DATA_PATH)
    except NameError:
        # Not in IPython/Jupyter environment, assume local
        return load_data(data_path=DEFAULT_DATA_PATH)


def join_salary_people(df_salary_raw, df_people_raw):
    """Join salary and people dataframes."""
    return df_salary_raw.join(df_people_raw, on="id", how="inner")
