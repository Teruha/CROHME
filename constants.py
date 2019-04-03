RFC_MODEL_FILE_NAME = 'rfc_model'
KD_TREE_MODEL_NAME = 'kd_tree_model.pkl'
DATA_FRAME_FILE_NAME = 'crohme_data.pkl'
SYMBOL_DATA_ONLY_FILE_NAME = 'symbol_data.pkl'
JUNK_DATA_ONLY_FILE = 'junk_data.pkl'
ISO_GROUND_TRUTH_FILE_NAME = 'iso_GT.txt'
JUNK_GROUND_TRUTH_FILE_NAME = 'junk_GT.txt'
EXCLUDED_FILES = [RFC_MODEL_FILE_NAME, DATA_FRAME_FILE_NAME, ISO_GROUND_TRUTH_FILE_NAME, JUNK_GROUND_TRUTH_FILE_NAME, KD_TREE_MODEL_NAME, \
    JUNK_DATA_ONLY_FILE, SYMBOL_DATA_ONLY_FILE_NAME]

RFC_IMPURITY_CRITERION = ['gini','entropy']

WINDOWS_PLATFORM = "win32"
DEBUG = True  # TODO: SET THIS TO FALSE BEFORE SUBMISSION