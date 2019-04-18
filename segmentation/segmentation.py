import numpy as np

# TODO: reused from project 1... should probably place in separate file again.
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

################################


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


# TODO: may want to consider interpolating more points to get a more accurate density histogram.
def densityHistogram(trace_dict):

    """
    Extract the number of points in each of the equally spaced vertical bins. This will hopefully give us some
    context so we can tell which strokes/trace groups go together. The bins are calculated across all of the traces.

    Parameters:
    trace_dict (dict: {int -> arr}) - dictionary of trace id to array of points

    Returns:
    frequencies (list) - [# points vertical bins, # points horizontal bins]
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