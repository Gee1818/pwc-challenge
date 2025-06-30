"""
Configuration file containing all constants and parameters.
"""

# Global constants
SEED = 100622
TARGET = 'Salary'
ID = 'id'
N_FOLDS = 5

# Column definitions (will be modified during feature engineering)
NUM_COLS = ['Age']
CAT_COLS = ['Gender', 'Education Level', 'Job Title', 'Years of Experience']

# Data paths
DEFAULT_DATA_PATH = "../data/"
COLAB_DATA_PATH = "/content/drive/MyDrive/Postgrado ciencia de datos/pwc-challenge/data/"

# Model export paths
MODELS_PATH = "../models/"
OOF_PATH = "../oof/"
