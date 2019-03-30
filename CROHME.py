from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

import csv

# path to training data (locally)
folderPath = '/Users/rachaelthormann/Desktop/Classes Spring 2019/Pattern Recognition/Project 1/'

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

                    # print trace values and plot
                    for values in soup.findAll("trace"):
                        # print(values)

                        trace_groups = str(values.contents[0])
                        for coor in trace_groups.replace('\n', '').split(','):
                            x, y = coor.split()
                            x, y = int(x), int(y)
                            # print((x, y))
                            plt.plot(x, -y, "ro")

                    plt.show()

            i += 1

    return


def main():
    readInputData(folderPath + 'InputTrainingSymbol.CSV')

    return


main()