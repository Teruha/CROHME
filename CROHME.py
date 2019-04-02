import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import bs4

# from bs4 import BeautifulSoup
from sys import platform
from scipy import interpolate

EXCLUDED_FILES = ['iso_GT.txt', 'crohme_data']
WINDOWS_PLATFORM = "win32"
DEBUG = True # TODO: SET THIS TO FALSE BEFORE SUBMISSION

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

def get_coordinates_from_trace(trace):
    """
    Get the coordinates from a single trace group

    Parameters:
    trace (string) - string of the coordinates separated by commas 

    Returns:
    points (list) - a list of tuples that represents the x,y coordinates
    """
    points = []
    trace_as_string = str(trace.contents[0])
    for coor in trace_as_string.replace('\n', '').split(','):
        x, y = coor.split()[:2]
        x, y = int(x), -int(y)
        points.append((x, y))
    return points

def separate_x_y_coors_from_trace(trace):
    """
    Return the all the x_coordinate values and all the y_coordinate values respectively from the points

    Parameters:
    trace (string) - string of the coordinates separated by commas

    Returns:
    x_coors (list) - list of ints representing the x coordinate of their corresponding point
    y_coors (list) - list of ints representing the y coordinate of their corresponding point
    """
    points = get_coordinates_from_trace(trace)
    x_coors = [p[0] for p in points]
    y_coors = [p[1] for p in points]
    return x_coors, y_coors

def draw_xml_file(trace_groups):
    """
    Draw the trace groups from a given XML file

    Parameters:
    trace_groups (list) - list of trace groups

    Returns:
    None
    """
    for trace in trace_groups:
        points = get_coordinates_from_trace(trace)
        # Draw line segments between points on the plot, 
        # to see points set the "marker" parameter to "+" or "o"
        for i in range(len(points)-1):
            plt.plot((points[i][0], points[i+1][0]), (points[i][1], points[i+1][1]), color='black')
    plt.show()

def extract_num_points_and_strokes(trace_groups):
    """
    Extract the total number of points and the number of strokes from a given trace_group 

    Parameters:
    trace_groups (list) - list of trace groups

    Returns:
    num_points (int) - total number of points
    num_strokes (int) - total number of strokes
    """
    num_points = 0
    num_strokes = len(trace_groups)
    for trace in trace_groups:
        num_points += len(get_coordinates_from_trace(trace))

    return num_points, num_strokes

def extract_directions(trace_groups):
    """
    Extract the directions taken to draw a specific symbol

    Parameters:
    trace_groups (list) - list of trace groups

    Returns:
    directions (list) - list of the strokes a user took 
        1 - up
        2 - down
        3 - left
        4 - right
    """
    up = 1
    down = 2
    left = 3
    right = 4
    # get the coordinates from the trace groups 
    directions = []
    for trace_idx, trace in enumerate(trace_groups):
        directions_for_trace = []
        x_coors, y_coors = separate_x_y_coors_from_trace(trace)
        for i in range(1, len(x_coors)): # starting from 1 because we compare the previous point
            if len(directions_for_trace) < 4:
                # Up
                if up not in directions_for_trace and (y_coors[i] - y_coors[i-1] > 0):
                    directions_for_trace.append(up)
                # Down
                if down not in directions_for_trace and (y_coors[i] - y_coors[i-1] < 0):
                    directions_for_trace.append(down)
                # Left
                if left not in directions_for_trace and (x_coors[i] - x_coors[i-1] < 0):
                    directions_for_trace.append(left)
                # Right
                if right not in directions_for_trace and (x_coors[i] - x_coors[i-1] > 0):
                    directions_for_trace.append(right)
        if DEBUG:
            print('Directions for stroke {0}: {1}'.format(trace_idx, directions_for_trace))
        directions.extend(directions_for_trace)
    return directions

def extract_curvature(trace_groups):
    """
    Quantify the curvature of a symbol 

    Parameters:
    trace_groups (list) - list of trace groups

    Returns:
    curvature (float) - The quantified curvature of a symbol
    """
    character_curvature = []
    for trace in trace_groups:
        trace_curves = []
        x_coors, y_coors = separate_x_y_coors_from_trace(trace)
        for i in range(1, len(x_coors)):
            delta_x = 0.01
            delta_y = 0
            if len(x_coors) > 4:
                if i < 2:
                    delta_x = x_coors[i + 2] - x_coors[i]
                    delta_y = y_coors[i + 2] - y_coors[i]
                elif i > len(x_coors) - 3:
                    delta_x = x_coors[i] - x_coors[i - 2]
                    delta_y = y_coors[i] - y_coors[i - 2]
                else:
                    delta_x = x_coors[i + 2] - x_coors[i - 2]
                    delta_y = y_coors[i + 2] - y_coors[i - 2]
                if delta_x != 0:
                    trace_curves.append(np.arctan(delta_y / delta_x))
        if len(trace_curves) == 0:
            character_curvature.append(0)
        else:
            character_curvature.append(np.mean(trace_curves))
    return np.mean(character_curvature)

def extract_aspect_ratio(trace_groups):
    """
    Calculate the aspect ratio of a symbol

    Parameters:
    trace_groups (list) - list of trace groups

    Returns:
    aspect_ratio (float) - aspect ratio of the trace groups
    """
    max_x, min_x = -np.inf, np.inf
    max_y, min_y = -np.inf, np.inf
    for trace in trace_groups:
        x_coors, y_coors = separate_x_y_coors_from_trace(trace)
        max_x = np.amax(x_coors) if np.amax(x_coors) > max_x else max_x
        max_y = np.amax(y_coors) if np.amax(y_coors) > max_y else max_y
        min_x = np.amin(x_coors) if np.amin(x_coors) < min_x else min_x
        min_y = np.amin(y_coors) if np.amin(y_coors) < min_y else min_y

    width = max_x - min_x
    height = max_y - min_y

    if width <= 0:
        width = 0.01
    if height <= 0:
        height = 0.01
    return width / height

# @Rachael do you think we need this?
def normalize_drawing(trace_groups):
    """
    Normalizes points between -1 and 1 in a 2D space

    Parameters:
    trace_groups (list) - list of trace groups

    Returns:
    new_trace_groups (list) - list of smoothed coordinates representing the new trace_groups
    """

def remove_consecutive_duplicate_points(trace_groups):
    """
    Remove duplicate points from a stroke before we smooth the strokes

    Parameters:
    trace_groups (list) - list of trace groups

    Returns:
    new_trace_groups (list) - list of smoothed coordinates
    """
    new_trace_groups = []
    for trace in trace_groups:
        new_trace = bs4.element.Tag(name='trace')
        new_trace.contents = [""]
        points = get_coordinates_from_trace(trace)
        points_to_keep = [points[0]]
        for i in range(len(points)-2):
            if points[i] != points[i+1]:
                points_to_keep.append(points[i+1])
        if points[0] != points[-1]:
            points_to_keep.append(points[-1]) # always keep the last point
        for p in points_to_keep:
            new_trace.contents[0] += '{0} {1},'.format(p[0], p[1])
        new_trace.contents[0] = new_trace.contents[0][:-1] # chop off last comma
        new_trace_groups.append(new_trace)
    
    if DEBUG:
        print('REMOVE_CONSECUTIVE_DUPLICATE_POINTS: ')
        print('trace_groups: {0}'.format(trace_groups))
        print('new_trace_groups: {0}'.format(new_trace_groups))
    return new_trace_groups

def interpolate_spline_points(x_coors, y_coors, deg):
    """
    Fits input points to a spline equation and get coefficients.
    generate interpolated points and return them

    Parameters:
    x_coors (list) - list of x coordinates
    y_coors (list) - list of y coordinates

    Returns:
    interpolated_x_coors (list) - list of points interpolated
    interpolated_y_coors (list) - list of points interpolated
    """
    tupletck, _ = interpolate.splprep([x_coors,y_coors], s=5.0, k=deg)
    steps = 1/len(x_coors)
    num_interpolation_points = np.arange(0, 1, steps)
    interoplated_x_coors, interoplated_y_coors = interpolate.splev(num_interpolation_points, tupletck)
    if DEBUG:
        print('interoplated_x_coors: {0}'.format(interoplated_x_coors))
        print('interoplated_y_coors: {0}'.format(interoplated_y_coors))
    return interoplated_x_coors, interoplated_y_coors

def smooth_points(trace_groups):
    """
    Smooths the points of the trace_group

    Parameters:
    trace_groups (list) - list of trace 

    Returns:
    new_trace_groups (list) - smoothed trace_group points
    """
    new_trace_groups = []
    for trace in trace_groups:
        new_trace = bs4.element.Tag(name='trace')
        new_trace.contents = [""]
        x_coors, y_coors = separate_x_y_coors_from_trace(trace)
        new_x_coors, new_y_coors = [], []

        # TODO for Qadir: understand this - https://github.com/dhavalc25/Handwritten-Math-Expression-Recognition/blob/master/project1.py#L276  
        if(len(x_coors) == 2):
                new_x_coors, new_y_coors = interpolate_spline_points(x_coors,y_coors, deg=1)
        if(len(x_coors) == 3):
            new_x_coors, new_y_coors = interpolate_spline_points(x_coors,y_coors, deg=2)
        if(len(x_coors) > 3):
            new_x_coors, new_y_coors = interpolate_spline_points(x_coors,y_coors, deg=3)
        for new_x, new_y in zip(new_x_coors, new_y_coors):
            new_trace.contents[0] += '{0} {1},'.format(int(new_x), int(new_y))

        new_trace.contents[0] = new_trace.contents[0][:-1] # chop off last comma
        new_trace_groups.append(new_trace)
    if DEBUG:
        print('SMOOTH_POINTS: ')
        print('trace_groups: {0}'.format(trace_groups))
        print('new_trace_groups: {0}'.format(new_trace_groups))
    return new_trace_groups
    
def extract_features(file, draw_input_data=False):
    """
    Extracts features from a single data file

    Parameters:
    file (string) - file name to read from current directory

    Returns:
    row (dict) - dictionary mapping the features to the data for a particular sample 
    """
    with open(file, 'r') as f:
        soup = bs4.BeautifulSoup(f, features='lxml')
        # you can iterate nd get whatever tag <> is needed
        unique_id = None
        for node in soup.findAll('annotation')[1]:
            unique_id = str(node)
        
        trace_groups = soup.findAll('trace') 

        # TODO: Uncomment when debugging is finished for smoothing
        # trace_groups = smooth_points(remove_consecutive_duplicate_points(trace_groups))

        if draw_input_data:
            draw_xml_file(trace_groups)
        num_points, num_strokes = extract_num_points_and_strokes(trace_groups)
        directions = extract_directions(trace_groups)
        curvature = extract_curvature(trace_groups)
        aspect_ratio = extract_aspect_ratio(trace_groups)

    return {'UI': unique_id, 'NUM_POINTS': num_points, 'NUM_STROKES': num_strokes, 'DIRECTIONS': directions, 
            'CURVATURE': curvature, 'ASPECT_RATIO': aspect_ratio, 'SYMBOL_REPRESENTATION': None}

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
    training_symbol_files (list) - list of the full paths of all junk files
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
                    if (f == 'iso_GT.txt'):
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

def build_training_data(symbol_files, print_progress=True):
    """
    Given the symbol files as input, create a dataframe from the given data

    Parameters:
    symbol_files (list) - list of symbol file names 

    Returns:
    data (Dataframe) - A pandas dataframe representation of the data
    """
    df = pd.DataFrame(columns=['UI','NUM_POINTS','NUM_STROKES','DIRECTIONS','CURVATURE','ASPECT_RATIO', 'SYMBOL_REPRESENTATION'])
    ui_to_symbols = map_ids_to_symbols()
    num_files = len(symbol_files)
    for i, symbol_file in enumerate(symbol_files):
        row = extract_features(symbol_file, True)
        row['SYMBOL_REPRESENTATION'] = ui_to_symbols[row['UI']]
        df.loc[i] = list(row.values())
        percentage = num_files//100
        if print_progress and percentage != 0 and i % percentage == 0:
            print('{0} ({1}%) of {2} files loaded...'.format(i, round((i/num_files)*100), num_files))
    print('Files 100% loaded.')
    return df # use this to operate on the data


def main():
    # TODO: Add df.to_pickle and df.read_pickle for saving and reading dataframe 
    # This way we won't have to read the training data everytime
    symbol_files = read_training_symbol_directory()
    df = build_training_data(symbol_files[2:3], False)
    print(df)
    # junk_files = read_training_junk_directory()

if __name__ == '__main__':
    main()
