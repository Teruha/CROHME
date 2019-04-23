import math
import numpy as np
import itertools
import sys

from points_manipulation import separate_x_y_coors_from_points
from file_manipulation import create_lg_files, load_files_to_dataframe, split_data
from CROHME import test_random_forest_classifier, train_random_forest_classifier

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
    
    if (o1 == 0 and on_segment(p1, p2, q1)):
        return True
    if (o2 == 0) and on_segment(p1, q2, q1):
        return True
    if (o3 == 0) and on_segment(p2, p1, q2):
        return True
    if (o4 == 0 and on_segment(p2, q1, q2)):
        return True
    return False

def determine_intersecting_traces(trace_dict):
    """
    Determines if two line segments in a trace group intersect with each other, 
    if they do, there's a good chance we have two tracegroups that represent a 
    mathematical symbols requiring two strokes 

    Parameters:
    1. trace_dict (dict: {int -> arr}) - dictionary of trace_ids to coordinates

    Returns:
    1. intersect (list) - list of tuples determining which lines intersect with one another
    """
    intersecting_lines = []
    for t1 in trace_dict:
        for t2 in trace_dict:
            if t1 != t2:
                line1_p1 = trace_dict[t1][0]
                line1_q1 = trace_dict[t1][-1]
                line2_p2 = trace_dict[t2][0]
                line2_q2 = trace_dict[t2][-1]
                if do_lines_intersect(line1_p1, line2_p2, line1_q1, line2_q2):
                    intersecting_lines.append((t1, t2))
    return intersecting_lines

def bounding_box(trace_points):
    """
    Obtain the minimum and maximum x, y coordinates that define the bounding box for a single trace.

    Parameters:
    trace_points (list) - list of tuples representing x, y coordinates

    Returns:
    bb (list) - [minimum x value, maximum x value, minimum y value, maximum y value;]
    """

    x_coors, y_coors = separate_x_y_coors_from_points(trace_points)

    # get the max and min values for all x and y
    min_x = np.amin(x_coors)
    max_x = np.amax(x_coors)
    min_y = np.amin(y_coors)
    max_y = np.amax(y_coors)

    bb = [min_x, max_x, min_y, max_y]

    return bb


# TODO: create functions for overlap x and overlap y
def bounding_box_overlap(trace_points_1, trace_points_2):
    """
    Determine if two bounding boxes overlap in a 2 dimensional space.

    Parameters:
    trace_points_1 (list) - list of tuples representing x, y coordinates
    trace_points_2 (list) - list of tuples representing x, y coordinates

    Returns:
    overlap (boolean) - True if overlap, False if no overlap
    """

    bb1 = bounding_box(trace_points_1)  # min_x, max_x, min_y, max_y
    bb2 = bounding_box(trace_points_2)

    # (bb1_min_x < bb2_max_x) and (bb2_min_x < bb1_max_x) and (bb1_min_y < bb2_max_y) and (bb2_min_y < bb1_max_y)
    overlap = (bb1[0] < bb2[1]) and (bb2[0] < bb1[1]) and \
              (bb1[2] < bb2[3]) and (bb2[2] < bb1[3])

    return overlap


def bounding_box_overlap_xdir(trace_points_1, trace_points_2):
    """
    Determine if two boxes overlap after being projected into a 1 dimensional space. Namely the x-axis.

    Parameters:
    trace_points_1 (list) - list of tuples representing x, y coordinates
    trace_points_2 (list) - list of tuples representing x, y coordinates

    Returns:
    overlap (boolean) - True if overlap, False if no overlap
    """

    bb1 = bounding_box(trace_points_1)  # min_x, max_x, min_y, max_y
    bb2 = bounding_box(trace_points_2)

    # (bb1_min_x < bb2_max_x) and (bb2_min_x < bb1_max_x)
    overlap = (bb1[0] < bb2[1]) and (bb2[0] < bb1[1])

    return overlap


def bounding_box_overlap_ydir(trace_points_1, trace_points_2):
    """
    Determine if two boxes overlap after being projected into a 1 dimensional space. Namely the y-axis.

    Parameters:
    trace_points_1 (list) - list of tuples representing x, y coordinates
    trace_points_2 (list) - list of tuples representing x, y coordinates

    Returns:
    overlap (boolean) - True if overlap, False if no overlap
    """

    bb1 = bounding_box(trace_points_1)  # min_x, max_x, min_y, max_y
    bb2 = bounding_box(trace_points_2)

    # (bb1_min_y < bb2_max_y) and (bb2_min_y < bb1_max_y)
    overlap = (bb1[2] < bb2[3]) and (bb2[2] < bb1[3])

    return overlap


# TODO: may want to consider interpolating more points to get a more accurate density histogram.
def density_histogram(trace_dict):
    """
    Extract the number of points in each of the equally spaced vertical bins. This will hopefully give us some
    context so we can tell which strokes/trace groups go together. The bins are calculated across all of the traces.

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    frequencies (list) - [# points vertical bin]
    """
    num_bins = 10

    # get the max and min values for all x
    max_x, min_x = -np.inf, np.inf

    for _, points in trace_dict.items():
        x_coors, y_coors = separate_x_y_coors_from_points(points)
        max_x = np.amax(x_coors) if np.amax(x_coors) > max_x else max_x
        min_x = np.amin(x_coors) if np.amin(x_coors) < min_x else min_x

    # find the boundaries for each of the 5 bins
    range_x = max_x - min_x
    intervals_x = np.linspace(0, range_x, num_bins+1)
    intervals_x = [value + min_x for value in intervals_x]  # must add the min_x value back to the binning values

    freq_x = [0] * num_bins

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

    return freq_x

def get_merging_threshold(trace_dict):
    """
    Get a reasonable threshold for merging given the traces and the maximum x,y and minimum x,y and 
    the number of traces

    Parameters:
    1. min_x (list) - maximum x value of a trace_dict
    2. min_y (int) - maximum y value of a trace_dict
    3. max_x (int) - minimum x value of a trace_dict
    4. max_y (int) - minimum y value of a trace_dict
    5. num_traces (int) - number of traces in the trace group

    Returns:
    1. threshold (float) - the minimum threshold required for use to consider two traces 'mergeable'
    """

    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    for _, points in trace_dict.items():
        xs, ys = separate_x_y_coors_from_points(points)
        max_x = max(xs) if max(xs) > max_x else max_x 
        min_x = min(xs) if min(xs) < min_x else min_x
        max_y = max(ys) if max(ys) > max_y else max_y
        min_y = min(ys) if min(ys) < min_y else min_y

    max_range = 0
    min_range = 0
    if max_x - min_x <= max_y - min_y:
        max_range = max_y - min_y
        min_range = max_x - min_x
    else:
        max_range = max_x - min_x
        min_range = max_y - min_y
    return (max_range-min_range)/(len(trace_dict) + 1) # play around with this value 

def calculate_center_of_mass(points):
    """
    Given a particular trace, calculate the center of mass

    Parameters:
    1. points (list) -list of coordinates that represent the trace

    Returns:
    1. com_x (float) - center of mass of the x coordinates
    2. com_y (float) - center of mass of the y coordinates
    """
    xs, ys = separate_x_y_coors_from_points(points)
    return (sum(xs)/len(xs), sum(ys)/len(ys))

def can_center_of_mass_of_traces_merge(points_1, points_2, threshold):
    """
    Determine if two traces from the same trace group can be merged by calculating each trace's 
    center of mass
    
    Parameters:
    1. points_1 (list) - list of coordinates that represent the trace
    2. points_2 (list) - list of coordinates that represent the trace
    3. threshold (float) - maximum distance that we determine these strokes to be 'mergeable'  

    Returns:
    1. can_be_merged (boolean) - boolean value based on calculations of the two traces merging together or not
    """
    trace_1_center = calculate_center_of_mass(points_1)
    trace_2_center = calculate_center_of_mass(points_2)
    if threshold >= math.sqrt((trace_1_center[0] - trace_2_center[0])**2 + (trace_1_center[1] - trace_2_center[1])**2):
        return True
    return False

def can_closest_traces_merge(points_1, points_2, threshold):
    """
    Calculate the closest two points between strokes and determine if they can be merged based on the threshold 

    Parameters:
    1. points_1 (list) - list of coordinates that represent the trace
    2. points_2 (list) - list of coordinates that represent the trace
    3. threshold (float) - maximum distance that we determine these strokes to be 'mergeable'

    Returns:
    1. can_be_merged (boolean) - boolean value based on calculations of the two traces merging together or not
    """
    min_dist = np.inf
    for p1 in points_1:
        for p2 in points_2:
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            min_dist = min_dist if dist > min_dist else dist
    return threshold >= dist 

def determine_mergeable_traces(trace_dict):
    """
    Given a group of traces, determine which ones can be merged via center of mass

    Parameters:
    1. trace_dict (dict: {int -> arr}) - dictionary of trace_ids to coordinates

    Returns:
    1. mergeable_traces (list) - list of mergeable 
    """
    mergeable_traces = []
    threshold = get_merging_threshold(trace_dict)
    for trace_id_1, points_1 in trace_dict.items():
        for trace_id_2, points_2 in trace_dict.items():
            mergeable = (can_center_of_mass_of_traces_merge(points_1, points_2, threshold) or (can_closest_traces_merge(points_1, points_2, threshold)))
            if trace_id_1 != trace_id_2 and mergeable:
                mergeable_traces.append((trace_id_1, trace_id_2))
    return mergeable_traces
                
def perform_segmentation(trace_dict):
    """
    Given a group of traces, attempt to group traces into groups that belong together, forming recognizable symbols 

    Parameters:
    1. trace_dict (dict: {int -> arr}) -  dictionary of trace_ids to coordinates

    Returns:
    1. new_trace_dicts (list) - returns a list of newly formed traces group that will have the features extracted.
                                This is a list of dictionaries with the key as a tuple of trace_ids and the value as the list of coordinates 
                                representing their respective trace_id (it is a list of lists). 

                                TODO: Is this really how we want to represent it? There must be a better way. 
                                Return to this (list of list of tracegroups? idk)
    """
    intersecting_traces = determine_intersecting_traces(trace_dict)
    mergable_traces = determine_mergeable_traces(trace_dict)
    
    return list(set(intersecting_traces).union(mergable_traces)) # as of right now, using intersection does not work, using union instead

def merge_tuples(tups):
    """
    Merges tuples in a way that makes all tuples with common elements form one set
    ex. merge_tuples([(1, 2), (3, 4), (1, 4)]) = {1, 2, 3, 4}

    Parameters:
    tups (list) - list of tuples

    Returns:
    groups (set) - list of sets
    """
    groups = [set(t) for t in tups]
    while True:
        for a, b in itertools.combinations(groups, 2):
            # if the groups can be merged
            if len(a & b):
                # construct new groups list
                groups = [g for g in groups if g != a and g != b]
                groups.append(a | b)

                # break the for loop and restart
                break
        else:
            # the for loop ended naturally, so no overlapping groups were found
            break
    return groups

def segment_trace_dicts(trace_dict):
    """
    Takes the results from the segmentation performed above and forms new trace_dicts based on the 
    traces calculated to belong together

    Parameters:
    1. trace_dict (dict) - unsegmented traces  

    Returns:
    1. trace_dicts (dicts) - list of newly segmented trace_dicts
    """
    segmented_groups = perform_segmentation(trace_dict)
    trace_dicts = []
    # remember that multiple trace ids can make up 1 symbol, and we want a trace_dict to represent a single symbol
    segmented_groups = merge_tuples(segmented_groups)
    segmented_groups_set = set()
    for group in segmented_groups:
        for trace_id in group:
            segmented_groups_set.add(trace_id)

    for trace_id in trace_dict:
        if trace_id not in segmented_groups_set:
            new_trace_dict = { trace_id: trace_dict[trace_id] }
            trace_dicts.append(new_trace_dict)

    for combined_traces in segmented_groups:
        new_trace_dict = {}
        for trace_id in combined_traces:
            new_trace_dict[trace_id] = trace_dict[trace_id]
        trace_dicts.append(new_trace_dict)
    
    return trace_dicts

def segmentation_main():
    """
    Differs from classification in a few ways. We need to read in the .inkML files that need to be segmented.
    Then when we parse the segmentations, we need to extract the features out of the traces and classify them.

    Parameters:
    None (for now)

    Returns:
    None
    """
    if len(sys.argv) == 1:
        print('USAGE: [[python3]] CROHME.PY [training_dir] [testing_dir] [(-tr)ain|(-te)st|(-b)oth]')
        print('Ex. 1: python3 CROHME.PY [training_symbols_dir OR .pkl file] [testing_symbols_dir OR .pkl file] -b')
        print('Ex. 2: python3 CROHME.PY [training_symbols_dir OR .pkl file] -tr')
        print('Ex. 2: python3 CROHME.PY [testing_symbols_dir OR .pkl file]  -te')
    elif len(sys.argv) == 3 or len(sys.argv) == 4:
        if sys.argv[-1] == '-tr': # train the model, this means we are creating a new one
            df = load_files_to_dataframe(sys.argv[1], segment_data_func=segment_trace_dicts)
            x_train = None
            if 'TRACES' in list(df.columns):
                x_train = df.drop(list(['SYMBOL_REPRESENTATION', 'UI' ,'TRACES'], axis=1))
            else:
                x_train = df.drop(list(['SYMBOL_REPRESENTATION', 'UI']), axis=1)
            y_train = df['SYMBOL_REPRESENTATION']
            train_random_forest_classifier(x_train, y_train)
        elif sys.argv[-1] == '-te': # test the model, this means it already exists
            df = load_files_to_dataframe(sys.argv[1], segment_data_func=segment_trace_dicts)
            dropped_x_test = df.drop(list(['SYMBOL_REPRESENTATION', 'UI' ,'TRACES']), axis=1) # Keep this
            y_test = df['SYMBOL_REPRESENTATION']
            predictions = test_random_forest_classifier(dropped_x_test, y_test, 200)
            create_lg_files(df, predictions)
        elif sys.argv[-1] == '-b': # test and train the model, this means we need to recreate the model and test it
            df, df2 = load_files_to_dataframe(sys.argv[1], sys.argv[2], True)
            x_train, _, y_train, _ = split_data(df, 0.00)
            _, x_test, _, y_test = split_data(df2, 0.99)
            train_random_forest_classifier(x_train, y_train)
            test_random_forest_classifier(x_test, y_test)
        else:
            print('ERROR: NO FLAG OR INVALID FLAG SPECIFIED.')
    else:
        print('INVALID PARAMETERS, PLEASE RUN FILE WITH NO PARAMETERS TO SEE USAGE.')


if __name__ == '__main__':
    segmentation_main()
   
    
