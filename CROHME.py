from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

import csv

# path to training data (locally)
folderPath = '/Users/rachaelthormann/Desktop/Classes Spring 2019/Pattern Recognition/Project 1/'
iso_4'/Users/qh/Documents/python_projects/csci-737/project-part-1/task2-trainSymb2014/trainingSymbols/iso4.inkml'

# FEATURE POSSIBILITIES
# number of trace points, normalized values ...

"""
def createInputCSV():

    f = open(folderPath + 'InputTrainingSymbol.CSV','w')

    for i in range(0, 85802):
        f.write('iso' + str(i) + '.inkml\n')

    f.close()

    return
"""
for values in soup.findAll("trace"):
    trace_groups = str(values.contents[0])
    points = []
    for i, coor in enumerate(trace_groups.replace('\n', '').split(',')):
        x, y = coor.split()
        x, y = int(x), -int(y)
        points.append((x, y))
#         plt.plot(x,y, 'ro')
#         plt.annotate(i, xy=(x, y), xytext=(5, 4), ha='left', textcoords='offset points')
    for i in range(len(points)-1):
        plt.plot((points[i][0], points[i+1][0]), (points[i][1], points[i+1][1]), color='black')

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

def readInputData(file):
    with open(file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in readCSV:
            if (i == 4):
                print(row[0])

                with open(folderPath + 'task2-trainSymb2014/trainingSymbols/' + row[0], 'r') as f:

                    soup = BeautifulSoup(f, features="lxml")
                    # you can iterate nd get whatever tag <> is needed
                    for node in soup.find_all('annotation'):
                        print(node)
                    
                    draw_xml_file(soup.findAll("trace"))
                    extract_num_points_and_strokes(soup.findAll("trace"))
                    


                    plt.show()

            i += 1

    return


def main():
    readInputData(folderPath + 'InputTrainingSymbol.CSV')

if __name__ = '__main__':
    main()