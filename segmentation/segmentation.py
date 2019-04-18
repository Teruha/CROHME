
import numpy as np
from classification.points_manipulation import separate_x_y_coors_from_points


def determine_intersecting_segments(trace_group):
    """
    Determines if two line segments in a trace group intersect with each other, 
    if they do, there's a good chance we have two tracegroups that represent a 
    mathematical symbols requiring two strokes 

    Parameters:
    1. trace_group (dict) -

    Returns:
    1. intersect (boolean) - boolean determining if two lines intersect
    """
    

def orientation(p, q, r):
    # p[0] is the x coordinate
    # p[1] is the y coordinate
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    
    if val == 0:
        return 0 # colinear
    
    return 1 if val > 0 else 2
    

def on_segment(p, q, r):
    if q[0] <= max([p[0], r[0]]) and q[0] >= min([p[0], r[0]]) and q[1] <= max([p[1], r[1]]) and q[1] >= min([p[1], r[1]]):
        return True
    return False


def do_lines_intersect(p1, p2, q1, q2):
    
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