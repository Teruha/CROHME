import pandas as pd
from feature_extraction import extract_features
from file_manipulation import map_ids_to_symbols
from sklearn.model_selection import train_test_split

def split_data(df):
    """
    Use scikit-learn traint_test_split method to partition the data into training and testing sets

    Parameters:
    df (Dataframe) - The data to partition into different sets

    Returns:
    x_train - training set for some model
    x_test - testing set for some model
    y_train - training labels for some model
    y_test - testing set for some model
    """
    x = df.drop(list(['SYMBOL_REPRESENTATION', 'UI']), axis=1)
    y = df['SYMBOL_REPRESENTATION']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    return x_train, x_test, y_train, y_test

def build_training_data(symbol_files, junk_files, print_progress=True):
    """
    Given the symbol files as input, create a dataframe from the given data

    Parameters:
    symbol_files (list) - list of symbol file names 

    Returns:
    data (Dataframe) - A pandas dataframe representation of the data
    """
    df = pd.DataFrame([]) # contains both junk and symbol files
    ui_to_symbols = map_ids_to_symbols()
    all_files = symbol_files[:]
    all_files.extend(junk_files)
    num_files = len(all_files)
    for i, data_file in enumerate(all_files):
        row = extract_features(data_file)
        row['SYMBOL_REPRESENTATION'] = ui_to_symbols[row['UI']] if row['UI'] in ui_to_symbols else 'junk'
        if len(df.columns) == 0:
            df = pd.DataFrame(columns=[n for n in row.keys()])
        df.loc[i] = list(row.values())
        percentage = num_files//100
        if print_progress and percentage != 0 and i % percentage == 0:
            print('{0} ({1}%) of {2} files loaded...'.format(i, round((i/num_files)*100), num_files))
    print('Files 100% loaded.')
    return df # use this to operate on the data
