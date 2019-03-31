import os
import csv
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
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
            x, y = coor.split()
            x, y = int(x), -int(y)
            points.append((x, y))
        # Draw line segments between points on the plot, 
        # to see points set the "marker" parameter to "+" or "o"
        for i in range(len(points)-1):
            plt.plot((points[i][0], points[i+1][0]), (points[i][1], points[i+1][1]), color='black')

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

def read_input_data(file):
    """
    Reads a single .inkML file
    

    Parameters:
    file (string) - file name to read from current directory

    Returns:
    None
    """
    with open(file, 'r') as f:

        soup = BeautifulSoup(f, features="lxml")
        # you can iterate nd get whatever tag <> is needed
        for node in soup.find_all('annotation'):
            print(node)
        
        draw_xml_file(soup.findAll("trace"))
        extract_num_points_and_strokes(soup.findAll("trace"))

        plt.show()


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
                    training_symbol_files.append(dirname +'/'+ f)
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

def main():
    symbol_files = read_training_symbol_directory()
    junk_files = read_training_junk_directory()


if __name__ == '__main__':
    main()
