import numpy as np
import bs4

from points_manipulation import *
from file_manipulation import draw_xml_file

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
