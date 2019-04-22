import pandas as pd
import bs4

# remove these
from random import randint
import sys
#

from classification.feature_extraction import extract_features
from classification.file_manipulation import map_ids_to_symbols
from classification.points_manipulation import get_coordinates_from_trace
from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.30):
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
    # x = df.drop(list(['SYMBOL_REPRESENTATION', 'UI']), axis=1)
    x = df
    y = df['SYMBOL_REPRESENTATION']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    
    return x_train, x_test, y_train, y_test

def build_training_data(symbol_files, junk_files=[], segment_data_files=False, print_progress=True):
    """
    Given the symbol files as input, create a dataframe from the given data

    Parameters:
    1. symbol_files (list) - list of symbol file names 

    Returns:
    1. df (Dataframe) - A pandas dataframe representation of the data
    """
    df = pd.DataFrame([]) # contains both junk and symbol files
    ui_to_symbols = map_ids_to_symbols()
    all_files = symbol_files[:]
    all_files.extend(junk_files)
    num_files = len(all_files)
    for i, data_file in enumerate(all_files):
        trace_dict = create_trace_dict(data_file)
        # unique_id = data_file.split('.')[0].split('/')[-1]
        unique_id = randint(0, sys.maxsize) # TODO: Remove this

        # segmentation to be done here
        if segment_data_files:
            print('Performing segmentation')  
            segemented_trace_dicts = segment_trace_dicts(trace_dict)
            for trace_dict in segemented_trace_dicts:
                row = extract_features(trace_dict, unique_id)
                # row['SYMBOL_REPRESENTATION'] = ui_to_symbols[row['UI']] # I don't there is a ground truth file for this 
                if len(df.columns) == 0:
                    df = pd.DataFrame(columns=[n for n in row.keys()])
                df.loc[i] = list(row.values())

        else:
            row = extract_features(trace_dict, unique_id)
            row['SYMBOL_REPRESENTATION'] = ui_to_symbols[row['UI']] if row['UI'] in ui_to_symbols else 'junk'
            if len(df.columns) == 0:
                df = pd.DataFrame(columns=[n for n in row.keys()])
            df.loc[i] = list(row.values())
        percentage = num_files//100
        if print_progress and percentage != 0 and i % percentage == 0:
            print('{0} ({1}%) of {2} files loaded...'.format(i, round((i/num_files)*100), num_files))
    print('Files 100% loaded.')
    return df # use this to operate on the data


def create_trace_dict(file):
    """
    Create trace_dict given a .inkml file

    Parameters:
    1. file (str) - file name

    Return:
    1. trace_dict (dict) - dictionary of trace id's to the list of coordinates they represent
    """
    with open(file, 'r') as f:
        soup = bs4.BeautifulSoup(f, features='lxml')
        unique_id = None

        for node in soup.findAll('annotation')[1]:
            unique_id = str(node)

        trace_dict = {}
        for trace in soup.findAll('trace'):
            trace_dict[trace['id']] = get_coordinates_from_trace(trace)
        return trace_dict

def segment_training_data(symbol_files, print_progress=True):
    """
    Given the symbol files as input, create a dataframe from the given data

    Parameters:
    1. symbol_files (list) - list of symbol file names 

    Returns:
    1. df (Dataframe) - A pandas dataframe representation of the data
    """
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