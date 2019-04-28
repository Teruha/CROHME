import numpy as np
import bs4
import matplotlib.pyplot as plt
 
from points_manipulation import *

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

def orientation(p, q, r):
    """
    Determine the orientation of ordered triplet

    Parameters:
    1. p (tuple) - x,y tuple representing a coordinate
    2. q (tuple) - x,y tuple representing a coordinate
    3. r (tuple) - x,y tuple representing a coordinate

    Return:
    1. orientation (int) - the orientation of the triplet, where 0 is colinear, 1 is clockwise, 2 is counterclockwise
    """
    # p[0] is the x coordinate
    # p[1] is the y coordinate
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    
    if val == 0:
        return 0 # colinear
    
    return 1 if val > 0 else 2
    

def on_segment(p, q, r):
    """
    Given three colinear points p, q, r, the function checks if point q lies on line segment 'pr'

    Parameters:
    1. p (tuple) - x,y tuple representing a coordinate
    2. q (tuple) - x,y tuple representing a coordinate
    3. r (tuple) - x,y tuple representing a coordinate

    Return:
    1. on_segement (boolean) - returns true or false depending on if point 'q' lies on 'pr'
    """
    if q[0] <= max([p[0], r[0]]) and q[0] >= min([p[0], r[0]]) and q[1] <= max([p[1], r[1]]) and q[1] >= min([p[1], r[1]]):
        return True
    return False


def do_lines_intersect(p1, p2, q1, q2):
    """
    The main function that returns true if line segment p1-q1 and p2-q2 intersect. 

    Parameters:
    1. p1 (tuple) - x,y tuple representing a coordinate
    2. q1 (tuple) - x,y tuple representing a coordinate
    3. p2 (tuple) - x,y tuple representing a coordinate
    4. q2 (tuple) - x,y tuple representing a coordinate

    Returns:
    1. do_lines_intersect (boolean) - returns true or false if these two line segements intersect
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if (o1 != o2 and o3 != o4):
        return True
    
    if (o1 == 0) and on_segment(p1, p2, q1):
        return True
    if (o2 == 0) and on_segment(p1, q2, q1):
        return True
    if (o3 == 0) and on_segment(p2, p1, q2):
        return True
    if (o4 == 0) and on_segment(p2, q1, q2):
        return True
    return False

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

def extract_covariance_between_x_y(trace_dict):
    """
    Calculate the covariance between x and y

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    covariance (float) - Covariance between x and y
    """
    cov = 0
    n = 0
    for _, points in trace_dict.items():
        x_coors, y_coors = separate_x_y_coors_from_points(points)
        mean_x = np.mean(x_coors) 
        mean_y = np.mean(y_coors)
        for i in range(len(x_coors)):
            n += 1
            cov += ((x_coors[i] - mean_x) * (y_coors[i] - mean_y))
    return (cov/n)

def extract_num_intersecting_lines(trace_dict):
    """
    Calculate the number of intersecting traces

    Parameters: 
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    num_intersections (int) - the number of intersections present in a symbol
    """
    num_intersections = 0
    if len(trace_dict) == 1:
        return num_intersections
    for trace_1, points_1 in trace_dict.items():
        for trace_2, points_2 in trace_dict.items():
            if trace_1 != trace_2 and do_lines_intersect(points_1[0], points_2[0], points_1[-1], points_2[-1]):
                num_intersections += 1
    return num_intersections
    

def get_crossings_in_boundary(trace_dict, start_x, end_x, start_y, end_y, axis):
    """
    Takes 9 lines passing through a boundary aligned to a specified axis
    Calculates number of times traces in a symbol intersects those lines

    Parameters:
    1. trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points
    2. start_x (int) - starting x coordinate
    3. end_x (int) - ending x coordinate
    4. start_y (int) - starting y coordinate
    5. end_y (int) - ending y coordinate
    6. axis (int) - The axis representing the alignment (0 for x-axis, 1 for y-axis)

    Returns:
    1. num_intersections (float) - number of intersections
    """
    count = 0
    if axis == 0:
        coordinates_along_axis = np.linspace(start_x, end_x, 9)
    else: 
        coordinates_along_axis = np.linspace(start_y, end_y, 9)
    for _, points in trace_dict.items():
        x_coors, y_coors = separate_x_y_coors_from_points(points)
        for i in range(len(x_coors) -1):
            p1 = (x_coors[i], y_coors[i])
            p2 = (x_coors[i+1], y_coors[i+1])

            for p in coordinates_along_axis:
                if axis == 0:
                    coor_1 = (p, start_y)
                    coor_2 = (p, end_y)
                else:
                    coor_1 = (start_x, p)
                    coor_2 = (end_x, p)
                if do_lines_intersect(p1, p2, coor_1, coor_2):
                    count += 1
    return count/9

def extract_crossings(trace_dict):
    """
    Gets the crossing feature for traces in a symbol. 
    Avg. crossings in each span is returned.

    """
    for _, points in trace_dict.items():
        x_coors, y_coors = separate_x_y_coors_from_points(points)
        max_x = max(x_coors)
        max_y = max(y_coors)
        min_x = min(x_coors)
        min_y = min(y_coors)
    x_boundary = np.linspace(min_x, max_x, num=6)
    y_boundary = np.linspace(min_y, max_y, num=6)

    x_crossings = []
    y_crossings = []
    for i in range(1, len(x_boundary)):
        x_crossings.append(get_crossings_in_boundary(trace_dict, x_boundary[i-1], x_boundary[i], min_y, max_y, 0))
    for i in range(1, len(y_boundary)):
        y_crossings.append(get_crossings_in_boundary(trace_dict, min_x, max_x, y_boundary[i-1], y_boundary[i], 1))

    return [x_crossings, y_crossings]



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

def extract_features(trace_dict, unique_id, draw_input_data=False):
    """
    Extracts features from a single data file

    Parameters:
    file (string) - file name to read from current directory

    Returns:
    row (dict) - dictionary mapping the features to the data for a particular sample 
    """
    trace_dict = normalize_drawing(smooth_points(remove_consecutive_duplicate_points(trace_dict)))
    if draw_input_data:
         draw_xml_file(trace_dict)
    num_points, num_strokes = extract_num_points_and_strokes(trace_dict)
    directions = extract_directions(trace_dict)
    if len(directions) == 0:
        directions.append(0)
    initial_direction = directions[0]
    end_direction = directions[-1]
    covariance = extract_covariance_between_x_y(trace_dict)

    curvature = extract_curvature(trace_dict)
    aspect_ratio = extract_aspect_ratio(trace_dict)
    frequencies = extract_frequencies(trace_dict)
    crossings = extract_crossings(trace_dict)
    num_intersecting_lines = extract_num_intersecting_lines(trace_dict)

    row = {'UI': unique_id, 'NUM_POINTS': num_points, 'NUM_STROKES': num_strokes, 'NUM_DIRECTIONS': len(directions), 
        'INITIAL_DIRECTION': initial_direction, 'END_DIRECTION': end_direction, 'CURVATURE': curvature,
        'ASPECT_RATIO': aspect_ratio, 'COVARIANCE': covariance, 'NUM_INTERSECTING_LINES': num_intersecting_lines}

    for i, f_x in enumerate(frequencies[0]):
        row['f_x_{0}'.format(i)] = f_x
    for i, f_y in enumerate(frequencies[1]):
        row['f_y_{0}'.format(i)] = f_y
    for i, c_x in enumerate(crossings[0]):
        row['c_x_{0}'.format(i)] = c_x
    for i, c_y in enumerate(crossings[1]):
        row['c_y_{0}'.format(i)] = c_y
    row['SYMBOL_REPRESENTATION'] = None
    row['TRACES'] = None
    return row


