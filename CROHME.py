import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

from bs4 import BeautifulSoup

def file_sorting_helper(full_path_name):
    """
    Grab the iso number from the file provided

    Parameters:
    full_path_name (str) - the full directory listing of the file

    Returns:
    iso_num (int) - the number of the iso file given in the full_path_name
    """
    iso_name = full_path_name.split('/')[-1]
    file_extension_idx = iso_name.index('.')
    iso_num = int(iso_name[3:file_extension_idx])
    return iso_num

def draw_xml_file(trace_groups):
    """
    Draw the trace groups from a given XML file

    Parameters:
    trace_groups (list) - List of trace groups

    Returns:
    None
    """
    for trace in trace_groups:
        trace_as_string = str(trace.contents[0])
        points = []
        for i, coor in enumerate(trace_as_string.replace('\n', '').split(',')):
            x, y = coor.split()[:2]
            x, y = int(x), -int(y)
            points.append((x, y))
        # Draw line segments between points on the plot, 
        # to see points set the "marker" parameter to "+" or "o"
        for i in range(len(points)-1):
            plt.plot((points[i][0], points[i+1][0]), (points[i][1], points[i+1][1]), color='black')
    plt.show()

def extract_num_points_and_strokes(trace_groups):
    """
    Extract the total number of points and the number of strokes from a given trace_group 

    Parameters:
    trace_groups (list) - List of trace groups

    Returns:
    num_points (int) - total number of points
    num_strokes (int) - total number of strokes
    """
    num_points = 0
    num_strokes = len(trace_groups)
    for trace in trace_groups:
        trace_as_string = str(trace.contents[0])
        num_points += len(trace_as_string.replace('\n', '').split(','))

    return num_points, num_strokes

def extract_features(file, draw_input_data=False):
    """
    Extracts features from a single data file

    Parameters:
    file (string) - file name to read from current directory

    Returns:
    row (dict) - dictionary mapping the features to the data for a particular sample 
    """
    with open(file, 'r') as f:
        soup = BeautifulSoup(f, features='lxml')
        # you can iterate nd get whatever tag <> is needed
        unique_id = None
        for node in soup.findAll('annotation')[1]:
            unique_id = str(node)
        
        if draw_input_data:
            draw_xml_file(soup.findAll("trace"))
        num_points, num_strokes = extract_num_points_and_strokes(soup.findAll("trace"))

    return {'UI': unique_id, 'NUM_POINTS': num_points, 'NUM_STROKES': num_strokes, 'SYMBOL_REPRESENTATION': None}

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
    training_symbol_files (list) - List of the full paths of all training files
    """
    training_symbol_files = []
    for (dirname, dirs, files) in os.walk(os.getcwd()):
        if 'trainingSymbols' in dirname:
            os.chdir(dirname)
            for (dirname, dirs, files) in os.walk(os.getcwd()):
                for f in files:
                    if (f != 'iso_GT.txt'): # we want to ignore this file
                        training_symbol_files.append(dirname +'/'+ f)
    training_symbol_files.sort(key=lambda s: file_sorting_helper(s))
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
    training_symbol_files (list) - List of the full paths of all junk files
    """
    training_junk_files = []
    for (dirname, dirs, files) in os.walk(os.getcwd()):
        if 'trainingJunk' in dirname:
            os.chdir(dirname)
            for (dirname, dirs, files) in os.walk(os.getcwd()):
                for f in files:
                    training_junk_files.append(dirname +'/'+ f)
    return training_junk_files

def map_ids_to_symbols():
    """

    """
    ground_truth_dict = {}
    ground_truth_file = None 
    for (dirname, dirs, files) in os.walk(os.getcwd()):
        if 'trainingSymbols' in dirname:
            os.chdir(dirname)
            for (dirname, dirs, files) in os.walk(os.getcwd()):
                for f in files:
                    if (f == 'iso_GT.txt'):
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

def build_training_data(symbol_files):
    """
    Given the symbol files as input, create a dataframe from the given data

    Parameters:
    symbol_files (list) - list of symbol file names 

    Returns:
    data (Dataframe) - A pandas dataframe representation of the data
    """
    df = pd.DataFrame(columns=['UI', 'NUM_POINTS', 'NUM_STROKES', 'SYMBOL_REPRESENTATION'])
    ui_to_symbols = map_ids_to_symbols()
    for i, symbol_file in enumerate(symbol_files):
        row = extract_features(symbol_file)
        row['SYMBOL_REPRESENTATION'] = ui_to_symbols[row['UI']]
        df.loc[i] = list(row.values())
    return df # use this to operate on the data


def main():
    symbol_files = read_training_symbol_directory()
    build_training_data(symbol_files)
    junk_files = read_training_junk_directory()

if __name__ == '__main__':
    main()
