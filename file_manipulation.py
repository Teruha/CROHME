import os
import constants as CONST
import pandas as pd
import bs4

from sys import platform, maxsize
from random import randint
from points_manipulation import get_coordinates_from_trace
from feature_extraction import extract_features
from sklearn.model_selection import train_test_split

def file_sorting_helper(full_path_name):
    """
    Grab the iso number from the file provided

    Parameters:
    full_path_name (str) - the full directory listing of the file

    Returns:
    iso_num (int) - the number of the iso file given in the full_path_name
    """
    iso_name = None
    if platform == CONST.WINDOWS_PLATFORM:
        iso_name = full_path_name.split('\\')[-1]
    else:
        iso_name = full_path_name.split('/')[-1]
    file_extension_idx = iso_name.index('.')
    iso_num = int(iso_name[3:file_extension_idx])
    return iso_num

def read_training_symbol_directory():
    """
    This function assumes that the training data is in a folder and that folder is within the same 
    directory as this file. This allows us to use the "os" package to look for and load the training 
    data

    ex.
        ls dir
            CROHME.py
            task2-trainSymb2014
        cd task2-trainSymb2014
        ls 
            trainingSymbols
            trainingJunk
            ...
            etc.
    Parameters:
    None

    Returns:
    training_symbol_files (list) - list of the full paths of all training files
    """
    training_symbol_files = []
    for (dirname, dirs, files) in os.walk(os.getcwd()):
        if 'trainingSymbols' in dirname:
            os.chdir(dirname)
            for (dirname, dirs, files) in os.walk(os.getcwd()):
                for f in files:
                    if (f not in CONST.EXCLUDED_FILES) and ('iso' in f): # we want to ignore these files
                        if platform == CONST.WINDOWS_PLATFORM:
                            training_symbol_files.append(dirname +'\\'+ f)
                        else:
                            training_symbol_files.append(dirname +'/'+ f)
    os.chdir('..')
    training_symbol_files.sort(key=lambda s: file_sorting_helper(s))
    return training_symbol_files

def get_inkml_files(dir):
    """
    This function grabs all .inkml files from a given directory

    Parameters:
    1. dir (str) - the directory that will be walked through to look for '.inkml' files 

    Returns:
    1. inkml_files (list) - list of the full paths of all '.inkml' files
    """
    training_symbol_files = []
    original_dir = os.getcwd()
    os.chdir(dir)
    for (dirname, _, files) in os.walk(os.getcwd()):
        for f in files:
            if (f not in CONST.EXCLUDED_FILES) and ('.inkml' in f): # we want to ignore these files
                if platform == CONST.WINDOWS_PLATFORM:
                    training_symbol_files.append(dirname +'\\'+ f)
                else:
                    training_symbol_files.append(dirname +'/'+ f)
    os.chdir(original_dir)
    training_symbol_files.sort()
    return training_symbol_files

def read_training_junk_directory():
    """
    This function assumes that the training data is in a folder and that folder is within the same 
    directory as this file. This allows us to use the "os" package to look for and load the training 
    data

    ex.
        ls dir
            CROHME.py
            task2-trainSymb2014
        cd task2-trainSymb2014
        ls 
            trainingSymbols
            trainingJunk
            ...
            etc.
    Parameters:
    None

    Returns:
    training_symbol_files (list) - list of the full paths of all junk files
    """
    training_junk_files = []
    for (dirname, dirs, files) in os.walk(os.getcwd()):
        if 'trainingJunk' in dirname:
            os.chdir(dirname)
            for (dirname, dirs, files) in os.walk(os.getcwd()):
                for f in files:
                    if (f not in CONST.EXCLUDED_FILES) and ('junk' in f):
                        if platform == CONST.WINDOWS_PLATFORM:
                            training_junk_files.append(dirname +'\\'+ f)
                        else:
                            training_junk_files.append(dirname +'/'+ f)
    os.chdir('..')
    return training_junk_files


def map_ids_to_symbols():
    """
    Maps the unique id of each file to the symbol the file represents

    Parameters:
    None

    Returns:
    ground_truth_dict (dict {String->String}) - dictionary of the unique id strings to their written symbols 
    """
    ground_truth_dict = {}
    ground_truth_file = None 
    for (dirname, dirs, files) in os.walk(os.getcwd()):
        if 'trainingSymbols' in dirname:
            os.chdir(dirname)
            for (dirname, dirs, files) in os.walk(os.getcwd()):
                for f in files:
                    if (f == CONST.ISO_GROUND_TRUTH_FILE_NAME):
                        if platform == CONST.WINDOWS_PLATFORM:
                            ground_truth_file = dirname +'\\'+ f
                        else:
                            ground_truth_file = dirname +'/'+ f
                        break
                if ground_truth_file != None:
                    break
        if ground_truth_file != None:
            break

    # build the ground truth to id dictionary
    with open(ground_truth_file, 'r') as f:
        for line in f:
            ground_truth_dict[line.split(',')[0]] = line.split(',')[1].strip('\n')
    return ground_truth_dict

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

def build_training_data(symbol_files, junk_files=[], segment_data_func=None, print_progress=True):
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
    row_num = 0
    for data_file in all_files:
        trace_dict = create_trace_dict(data_file)
        unique_id = data_file.split('.')[0].split('/')[-1]

        # segmentation to be done here
        if segment_data_func:
            segemented_trace_dicts = segment_data_func(trace_dict)
            for trace_dict in segemented_trace_dicts:
                row = extract_features(trace_dict, unique_id)
                row['TRACES'] = list(trace_dict.keys())
                # row['SYMBOL_REPRESENTATION'] = ui_to_symbols[row['UI']] # I don't there is a ground truth file for this 
                if len(df.columns) == 0:
                    df = pd.DataFrame(columns=[n for n in row.keys()])
                df.loc[row_num] = list(row.values())
                row_num += 1
            
        else:
            row = extract_features(trace_dict, unique_id)
            row['SYMBOL_REPRESENTATION'] = ui_to_symbols[row['UI']] if row['UI'] in ui_to_symbols else 'junk'
            if len(df.columns) == 0:
                df = pd.DataFrame(columns=[n for n in row.keys()])
            df.loc[row_num] = list(row.values())
        percentage = num_files//100
        if print_progress and percentage != 0 and row_num % percentage == 0:
            print('{0} ({1}%) of {2} files loaded...'.format(row_num, round((row_num/num_files)*100), num_files))
        row_num += 1
    print('Files 100% loaded.')
    return df # use this to operate on the data

def load_files_to_dataframe(dir_name, second_dir=None, segment_data_func=None, save=True):
    """
    Takes the input files and creates a dataframe from them where the model is then trained and tested

    Parameters:
    1. dir_name (str) - first directory where the files are stored
    2. second_dir (str) - second directory where the files are stored
    3. segmentation (boolean) - boolean that specifies segmentation for the build_training_data function
    4. save (boolean) - boolean determining whether we want to save the dataframes

    Returns:
    1. df (DataFrame) - The first dataframe to create
    2. df2 (DataFrame) - The second dataframe to create
    """
    df = None
    df2 = None
    if '.pkl' in dir_name:
        df = pd.read_pickle(dir_name)
    else: # read file(s) from input directory
        print('Building dataframe...')
        df = build_training_data(get_inkml_files(dir_name), segment_data_func=segment_data_func)
        print('Dataframe created: \n {}'.format(df.head()))
        if save:
            df.to_pickle(dir_name + '.pkl')
    if not second_dir:
        return df

    if '.pkl' in second_dir:
        df2 = pd.read_pickle(second_dir)
    else: # read file(s) from input directory
        df2 = build_training_data(get_inkml_files(second_dir), segment_data_func=segment_data_func)
        if save:
            df2.to_pickle(second_dir + '.pkl')
    return df, df2

def create_lg_files(x_test, predictions):
    """
    Creates lg files from the predictions

    Parameters:
    1. x_test (pandas series) - subset of the main dataframe that we will use for testing the classifier
    2. predictions (pandas series) - the predicted symbols given the x_test as input 

    Returns:
    None
    """
    curr_path = os.path.dirname(os.path.abspath(__file__))
    full_lg_dir = os.path.join(curr_path, CONST.LG_PREDICTIONS_DIRECTORY)
    if not os.path.isdir(full_lg_dir):
        os.mkdir(full_lg_dir)
    os.chdir(full_lg_dir)
    for the_file in os.listdir(os.getcwd()): # we should be inside the lg_file_dir
        file_path = os.path.join(full_lg_dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(e)
    
    file_out_dict = {} 
    # make a dictionary for the results of each UID
    for uid, traces, pred in zip(x_test['UI'], x_test['TRACES'], predictions):
        if uid not in file_out_dict:
            file_out_dict[uid] = {}
        if pred not in file_out_dict[uid]:
            file_out_dict[uid][pred] = []    
        file_out_dict[uid][pred].append(traces)
    
    for uid, preds in file_out_dict.items():
        file_name = '{0}/{1}.lg'.format(full_lg_dir, uid)
        with open(file_name, 'a+') as f:
            f.write('# IUD, {0}\n'.format(uid))
            num_elements = 0 
            for p, traces in preds.items():
                num_elements += len(traces)
            f.write('# Objects({0})\n'.format(num_elements))
    
    for uid, preds in file_out_dict.items():
        file_name = '{0}{1}.lg'.format(full_lg_dir, uid)
        with open(file_name, 'a+') as f:
            for p, traces in preds.items():
                for i, trace in enumerate(traces):
                    character_traces = ', '.join(trace)
                    line = 'O, {0}_{1}, {0}, 1.0, {2}\n'.format(p, i+1, character_traces)
                    f.write(line)
    os.chdir('../')
        