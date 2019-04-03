import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import bs4
import _pickle as cPickle

from sys import platform
from scipy import interpolate
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

RFC_MODEL_FILE_NAME = 'rfc_model.pkl'
KD_TREE_MODEL_NAME = 'kd_tree_model.pkl'
DATA_FRAME_FILE_NAME = 'crohme_data.pkl'
SYMBOL_DATA_ONLY_FILE_NAME = 'symbol_data.pkl'
JUNK_DATA_ONLY_FILE = 'junk_data.pkl'
ISO_GROUND_TRUTH_FILE_NAME = 'iso_GT.txt'
EXCLUDED_FILES = [RFC_MODEL_FILE_NAME, DATA_FRAME_FILE_NAME, ISO_GROUND_TRUTH_FILE_NAME, KD_TREE_MODEL_NAME, \
    JUNK_DATA_ONLY_FILE, SYMBOL_DATA_ONLY_FILE_NAME]
WINDOWS_PLATFORM = "win32"
DEBUG = True  # TODO: SET THIS TO FALSE BEFORE SUBMISSION

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
        x, y = float(x), -float(y)
        points.append((x, y))
    return points

def separate_x_y_coors_from_points(points):
    """
    Return the all the x_coordinate values and all the y_coordinate values respectively from the points

    Parameters:
    points (list) - list of tuples representing the (x,y) coordinates

    Returns:
    x_coors (list) - list of ints representing the x coordinate of their corresponding point
    y_coors (list) - list of ints representing the y coordinate of their corresponding point
    """
    x_coors = [p[0] for p in points]
    y_coors = [p[1] for p in points]
    return x_coors, y_coors

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

def extract_num_points_and_strokes(trace_dict):
    """
    Extract the total number of points and the number of strokes from a given trace_group 

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    num_points (int) - total number of points
    num_strokes (int) - total number of strokes
    """
    num_points = 0
    num_strokes = len(trace_dict.keys())
    for _, points in trace_dict.items():
        num_points += len(points)

    return num_points, num_strokes

def extract_directions(trace_dict):
    """
    Extract the directions taken to draw a specific symbol

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

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
    for _, points in trace_dict.items():
        directions_for_trace = []
        x_coors, y_coors = separate_x_y_coors_from_points(points)
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
        # if DEBUG:
        #     print('Directions for stroke {0}: {1}'.format(trace_idx, directions_for_trace))
        directions.extend(directions_for_trace)
    return directions

def extract_curvature(trace_dict):
    """
    Quantify the curvature of a symbol 

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    curvature (float) - The quantified curvature of a symbol
    """
    character_curvature = []
    for _, points in trace_dict.items():
        trace_curves = []
        x_coors, y_coors = separate_x_y_coors_from_points(points)
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

def extract_aspect_ratio(trace_dict):
    """
    Calculate the aspect ratio of a symbol

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    aspect_ratio (float) - aspect ratio of the trace groups
    """
    max_x, min_x = -np.inf, np.inf
    max_y, min_y = -np.inf, np.inf
    for _, points in trace_dict.items():
        x_coors, y_coors = separate_x_y_coors_from_points(points)
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

def extract_frequencies(trace_dict):
    """
    Extract the number of points in each of the 5 equally spaced vertical and horizontal bins.

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    frequencies (list) - [# points vertical bins, # points horizontal bins]
    """

    # get the max and min values for x and y respectively
    max_x,min_x = -np.inf, np.inf
    max_y, min_y = -np.inf, np.inf
    for _, points in trace_dict.items():
        x_coors, y_coors = separate_x_y_coors_from_points(points)
        max_x = np.amax(x_coors) if np.amax(x_coors) > max_x else max_x
        max_y = np.amax(y_coors) if np.amax(y_coors) > max_y else max_y
        min_x = np.amin(x_coors) if np.amin(x_coors) < min_x else min_x
        min_y = np.amin(y_coors) if np.amin(y_coors) < min_y else min_y

    # find the boundaries for each of the 5 bins
    range_x = max_x - min_x
    intervals_x = np.linspace(0, range_x, 6)
    intervals_x = [value + min_x for value in intervals_x]  # must add the min_x value back to the binning values
    range_y = max_y - min_y
    intervals_y = np.linspace(0, range_y, 6)
    intervals_y = [value + min_y for value in intervals_y]  # must add the min_y value back to the binning values

    freq_x = [0, 0, 0, 0, 0]
    freq_y = [0, 0, 0, 0, 0]

    for _, points in trace_dict.items():
        x_coors, y_coors = separate_x_y_coors_from_points(points)

        # add each x coordinate to the x bins
        for x in x_coors:
            if (x >= intervals_x[0]) and (x < intervals_x[1]):
                freq_x[0] += 1
            elif (x >= intervals_x[1]) and (x < intervals_x[2]):
                freq_x[1] += 1
            elif (x >= intervals_x[2]) and (x < intervals_x[3]):
                freq_x[2] += 1
            elif (x >= intervals_x[3]) and (x < intervals_x[4]):
                freq_x[3] += 1
            else:
                freq_x[4] += 1

        # add each y coordinate to the y bins
        for y in y_coors:
            if (y >= intervals_y[0]) and (y < intervals_y[1]):
                freq_y[0] += 1
            elif (y >= intervals_y[1]) and (y < intervals_y[2]):
                freq_y[1] += 1
            elif (y >= intervals_y[2]) and (y < intervals_y[3]):
                freq_y[2] += 1
            elif (y >= intervals_y[3]) and (y < intervals_y[4]):
                freq_y[3] += 1
            else:
                freq_y[4] += 1

    frequencies = [freq_x, freq_y]
    return frequencies

# @Rachael do you think we need this?
def normalize_drawing(trace_dict):
    """
    Normalizes points between -1 and 1 in a 2D space

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    new_trace_dict (dict: {int -> arr}) - normalized collection of points
    """
    new_trace_dict = {}
    for trace_id, points in trace_dict.items(): 
        local_min_x = np.inf
        local_max_x = -np.inf
        local_min_y = np.inf
        local_max_y = -np.inf

        x_coors, y_coors = separate_x_y_coors_from_points(points)

        if len(x_coors) == 1:
            local_min_x = x_coors[0] - 10
            local_max_x = x_coors[0] + 10
            local_min_y = y_coors[0] - 20
            local_max_y = y_coors[0] + 20
        else:
            local_min_x = np.amin(x_coors)
            local_max_x = np.amax(x_coors)
            local_min_y = np.amin(y_coors)
            local_max_y = np.amax(y_coors)
        
        # Rescale the min and max depending on which direction maximum variation is
        diff = abs((local_max_x - local_min_x) - (local_max_y - local_min_y))/2
        if (local_max_x - local_min_x) > (local_max_y - local_min_y):
            local_max_y = local_max_y + diff
            local_min_y = local_min_y - diff
        else:
            local_max_x = local_max_x + diff
            local_min_x = local_min_x - diff
        new_trace_dict[trace_id] = scale_points({ trace_id: points }, local_max_x, local_min_x, local_max_y, local_min_y)
    return new_trace_dict

def scale_points(trace_dict, max_x, min_x, max_y, min_y):
    """
    Scale the trace points between -1 and 1

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    new_trace_dict (dict: {int -> arr}) - scaled collection of points
    """
    new_points = None
    for trace_id, points in trace_dict.items(): 
        x_coors, y_coors = separate_x_y_coors_from_points(points)
        for i in range(len(x_coors)):
            if max_x - min_x == 0:
                max_x = x_coors[i] + 10
                min_x = x_coors[i] - 10
            if max_y - min_y == 0:
                max_y = y_coors[i] + 20
                min_y = y_coors[i] - 20
            
            x_coors[i] = ((2*(x_coors[i] - min_x))/(max_x - min_x)) - 1
            y_coors[i] = ((2*(y_coors[i] - min_y))/(max_y - min_y)) - 1
        
        new_points = [(x,y) for (x,y) in zip(x_coors, y_coors)]
    return new_points

def remove_consecutive_duplicate_points(trace_dict):
    """
    Remove duplicate points from a stroke before we smooth the strokes

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    new_trace_dict (dict: {int -> arr}) - unique trace_dict points
    """
    new_trace_dict = {}
    for trace_id, points in trace_dict.items():
        points_to_keep = [points[0]]
        for i in range(len(points)-2):
            if points[i] != points[i+1]:
                points_to_keep.append(points[i+1])
        if points[0] != points[-1] and points[-1] != points[-2]:
            points_to_keep.append(points[-1]) # always keep the last point
        new_trace_dict[trace_id] = points_to_keep
    
    # if DEBUG:
    #     print('REMOVE_CONSECUTIVE_DUPLICATE_POINTS: ')
    #     print('trace_groups: {0}'.format(trace_groups))
    #     print('new_trace_groups: {0}'.format(new_trace_groups))
    return new_trace_dict

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
    # if DEBUG:
    #     print('interoplated_x_coors: {0}'.format(interoplated_x_coors))
    #     print('interoplated_y_coors: {0}'.format(interoplated_y_coors))
    return interoplated_x_coors, interoplated_y_coors

def smooth_points(trace_dict):
    """
    Smooths the points of the trace_group

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    new_trace_dict (dict: {int -> arr}) - smoothed trace_group points
    """
    new_trace_dict = {}
    for trace_id, points in trace_dict.items():
        x_coors, y_coors = separate_x_y_coors_from_points(points)
        new_x_coors, new_y_coors = [], []
        new_points = []
        # TODO for Qadir: understand this - https://github.com/dhavalc25/Handwritten-Math-Expression-Recognition/blob/master/project1.py#L276  
        if(len(x_coors) == 2):
                new_x_coors, new_y_coors = interpolate_spline_points(x_coors,y_coors, deg=1)
        if(len(x_coors) == 3):
            new_x_coors, new_y_coors = interpolate_spline_points(x_coors,y_coors, deg=2)
        if(len(x_coors) > 3):
            new_x_coors, new_y_coors = interpolate_spline_points(x_coors ,y_coors, deg=3)
        for new_x, new_y in zip(new_x_coors, new_y_coors):
            new_points.append((float(new_x), float(new_y)))

        new_trace_dict[trace_id] = new_points if len(new_points) != 0 else points
    # if DEBUG:
    #     print('SMOOTH_POINTS: ')
    #     print('trace_dict: {0}'.format(trace_dict))
    #     print('new_trace_dict: {0}'.format(new_trace_dict))
    return new_trace_dict

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
        unique_id = None

        for node in soup.findAll('annotation')[1]:
            unique_id = str(node)
        
        trace_dict = {}
        for i, trace in enumerate(soup.findAll('trace')):
            trace_dict[i] = get_coordinates_from_trace(trace)
        # trace_dict = smooth_points(remove_consecutive_duplicate_points(trace_dict))
        trace_dict = normalize_drawing(smooth_points(remove_consecutive_duplicate_points(trace_dict)))

        if draw_input_data:
            draw_xml_file(trace_dict)
        num_points, num_strokes = extract_num_points_and_strokes(trace_dict)
        directions = extract_directions(trace_dict)
        if len(directions) == 0:
            directions.append(0)
        initial_direction = directions[0]
        end_direction = directions[-1]

        curvature = extract_curvature(trace_dict)
        aspect_ratio = extract_aspect_ratio(trace_dict)
        frequencies = extract_frequencies(trace_dict)

        row = {'UI': unique_id, 'NUM_POINTS': num_points, 'NUM_STROKES': num_strokes, 'NUM_DIRECTIONS': len(directions), 
            'INITIAL_DIRECTION': initial_direction, 'END_DIRECTION': end_direction, 'CURVATURE': curvature,
            'ASPECT_RATIO': aspect_ratio}

        for i, f_x in enumerate(frequencies[0]):
            row['f_x_{0}'.format(i)] = f_x
        for i, f_y in enumerate(frequencies[1]):
            row['f_y_{0}'.format(i)] = f_y

        row['SYMBOL_REPRESENTATION'] = None
    return row

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
    num_files = len(symbol_files)
    all_files = symbol_files[:]
    all_files.extend(junk_files)
    for i, data_file in enumerate(all_files):
        row = extract_features(data_file)
        row['SYMBOL_REPRESENTATION'] = ui_to_symbols[row['UI']]
        if len(df.columns) == 0:
            df = pd.DataFrame(columns=[n for n in row.keys()])
        df.loc[i] = list(row.values())
        percentage = num_files//100
        if print_progress and percentage != 0 and i % percentage == 0:
            print('{0} ({1}%) of {2} files loaded...'.format(i, round((i/num_files)*100), num_files))
    print('Files 100% loaded.')
    return df # use this to operate on the data

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

def run_random_forest_classifier(x_train, x_test, y_train, y_test):
    """
    Run the Random Forest Classifier on the processed data and output the results.
    NOTE: This function creates a new pickle file if no such file exists or reads an existing one

    Parameters:
    x_train (pandas.series) - Training samples building the model
    x_test (pandas.series) - Testing samples 
    y_train (pandas.series) - Training samples building the model
    y_test (pandas.series) - Testing samples

    Returns:
    None
    """
    rfc = None
    try:
        with open(RFC_MODEL_FILE_NAME, 'rb') as f: 
            rfc = cPickle.load(f)
    except FileNotFoundError:
        rfc = RandomForestClassifier(n_estimators=300)
        rfc.fit(x_train, y_train)
        with open(RFC_MODEL_FILE_NAME, 'wb') as f:
            cPickle.dump(rfc, f)

    rfc_pred = rfc.predict(x_test)
    print_top_n_predictions(rfc, x_test)

    print('Random Forest Classifier results:')
    if len(x_train) < 1000:
        print('Confusion Matrix: ')
        print(confusion_matrix(y_test, rfc_pred))
    print('Classification Report: ')
    print(classification_report(y_test, rfc_pred))

    
def run_KDtree_classifier(x_train, x_test, y_train, y_test):
    """
    Run the KD Tree Classifier on the processed data and output the results.
    NOTE: This function creates a new pickle file if no such file exists or reads an existing one

    Parameters:
    x_train (pandas.series) - Training samples building the model
    x_test (pandas.series) - Testing samples 
    y_train (pandas.series) - Training samples building the model
    y_test (pandas.series) - Testing samples

    Returns:
    None
    """
    kdc = None
    try: 
        with open(KD_TREE_MODEL_NAME, 'rb') as f: 
            kdc = cPickle.load(f)
    except FileNotFoundError:
        kdc = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
        kdc.fit(x_train, y_train)
        with open(KD_TREE_MODEL_NAME, 'wb') as f:
            cPickle.dump(kdc, f)
    
    kdc_pred = kdc.predict(x_test)
    print_top_n_predictions(kdc, x_test)

    print('KDTree Classifier results:')
    if len(x_train) < 1000:
        print('Confusion Matrix: ')
        print(confusion_matrix(y_test, kdc_pred))
    print('Classification Report: ')
    print(classification_report(y_test, kdc_pred))


def print_top_n_predictions(model, test_data, n=10):
    """
    Print the top n predictions of a sample based on the probability the sample belongs to each class

    Parameters:
    rfc (Sklearn.model) - Trained random forest classifier
    test_data (Dataframe) - Subset of the training data used for testing 

    Returns:
    None
    """
    prediciton_probabilities = model.predict_proba(test_data)
    top_n = np.argsort(prediciton_probabilities)[:,:-n-1:-1]
    print(model.classes_[top_n])


def main():
    try:
        df = pd.read_pickle(DATA_FRAME_FILE_NAME)
    except FileNotFoundError:
        symbol_files = read_training_symbol_directory()
        junk_files = read_training_junk_directory()
        df = build_training_data(symbol_files, []) # TODO: Replace empty array with junk files when ready to test both
        os.chdir('../..')
        df.to_pickle(DATA_FRAME_FILE_NAME)
    
    x_train, x_test, y_train, y_test = split_data(df)
    run_random_forest_classifier(x_train, x_test, y_train, y_test)
    run_KDtree_classifier(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    main()
