import matplotlib.pyplot as plt
import os

from sys import platform
from classification.constants import *

def draw_xml_file(trace_dict):
    """
    Draw the trace groups from a given XML file

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    None
    """
    for _, points in trace_dict.items():
        # Draw line segments between points on the plot, 
        # to see points set the "marker" parameter to "+" or "o"
        for i in range(len(points)-1):
            plt.plot((points[i][0], points[i+1][0]), (points[i][1], points[i+1][1]), color='black')
    plt.show()

def file_sorting_helper(full_path_name):
    """
    Grab the iso number from the file provided

    Parameters:
    full_path_name (str) - the full directory listing of the file

    Returns:
    iso_num (int) - the number of the iso file given in the full_path_name
    """
    iso_name = None
    if platform == WINDOWS_PLATFORM:
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
                    if (f not in EXCLUDED_FILES) and ('iso' in f): # we want to ignore these files
                        if platform == WINDOWS_PLATFORM:
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
            if (f not in EXCLUDED_FILES) and ('.inkml' in f): # we want to ignore these files
                if platform == WINDOWS_PLATFORM:
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
                    if (f not in EXCLUDED_FILES) and ('junk' in f):
                        if platform == WINDOWS_PLATFORM:
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
                    if (f == ISO_GROUND_TRUTH_FILE_NAME):
                        if platform == WINDOWS_PLATFORM:
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
