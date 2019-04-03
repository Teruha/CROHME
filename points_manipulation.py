import numpy as np

from scipy import interpolate

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
