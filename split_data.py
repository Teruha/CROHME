"""
Authors: Rachael Thormann and Qadir Haqq
"""
import bs4
from os import listdir
from os.path import join
from points_manipulation import get_coordinates_from_trace


# change this to reflect where your training files are being stored
FILEPATH = '/Users/rachaelthormann/Desktop/Train/inkml'


def get_training_directory():
    """
    Loop through all folders within training directory and create a list of all inkml files.

    Return:
    1. files (list) - list of the full paths of all training inkml files
    """
    files = []

    # loop through each folder in given directory
    for directory in listdir(FILEPATH):
        if '.' not in directory:
            for file in listdir(join(FILEPATH, directory)):

                path = join(join(FILEPATH, directory), file)  # get full path

                # make sure you are looking at an inkml file
                if path.endswith('.inkml'):
                    files.append(path)

    return files


def get_trace_groups(file):
    """
    Create trace_dict given an inkml file.

    Parameters:
    1. file (str) - file name

    Return:
    1. trace_dict (dict) - dictionary of trace id's to the list of coordinates they represent
    """

    with open(file, 'r') as f:
        soup = bs4.BeautifulSoup(f, features='lxml')

        trace_dict = {}

        for trace in soup.findAll('trace'):
            trace_dict[trace['id']] = get_coordinates_from_trace(trace)

        return trace_dict


def get_symbol_counts():
    """
    Get the counts for all symbols in all inkml files given.

    Return:
    1. symbol_count (dict) : dictionary of of symbols to their counts within all files
    """

    file_list = get_training_directory()

    symbol_count = dict()

    for file in file_list:
        print('Extracting Info: ', file)

        # extract the trace groups
        tracegroups = get_trace_groups(file)

        # loop through the trace groups to count the symbols
        for trace_group in tracegroups.keys():

            if symbol_count.get(trace_group) is None:
                symbol_count[trace_group] = 1
            else:
                symbol_count[trace_group] += 1

    return symbol_count


def test_train_split(file_list, symbol_count):
    """
    Split the data into training and testing sets based on the priors for each symbol.

    Parameters:
    1. file_list (list): list of inkml files
    2. symbol_count (dict) : symbol counts in all inkml files

    Return:
    1. training_files (list) : list of files that should be used in training set
    2. testing_files (list) : list of files that should be used in testing set
    """

    # create two lists.. one for training and one for testing
    training_files = []
    testing_files = []

    train_set_symbol_count = dict()

    # first get the total symbol count for all files
    for file in file_list:

        # print('Training/Testing split:', file)

        temp = dict()

        tracegroups = get_trace_groups(file)

        # for each file place the symbol counts in a new temporary dictionary
        for trace_group in tracegroups.keys():

            if temp.get(trace_group) is None:
                temp[trace_group] = 1
            else:
                temp[trace_group] += 1

        # now loop through each symbol present within that file
        for key in temp.keys():

            # get the count for each symbol plus the current symbol count in the training set
            if train_set_symbol_count.get(key) is None:
                count = temp.get(key)
            else:
                count = temp.get(key) + train_set_symbol_count.get(key)

            # we want a 70/30 split so if the count for a symbol is less than 70% of total symbol count
            # place in training set
            if count < (symbol_count.get(key) * 0.7):
                training_files.append(file)
                if train_set_symbol_count.get(key) is None:
                    train_set_symbol_count[key] = 1
                else:
                    train_set_symbol_count[key] += 1

            # otherwise
            # place in testing set
            else:
                testing_files.append(file)
            break

    return training_files, testing_files


def create_csv(file_list, file):
    """
    Create a CSV file with the given name. This file contains a list pf paths to the training or testing files.

    :param file_list: list of inkml files in set
    :param file: file to write to
    :return: None
    """

    with open(file, "w") as outfile:
        for entry in file_list:

            # only take the last 3 pieces of the path
            entry1 = "/".join(entry.split('/')[4:])

            outfile.write(entry1)
            outfile.write("\n")

    return


def main():
    """
    Program to split the data into training and testing sets using the priors of the symbol.

    :return:  None
    """

    file_list = get_training_directory()

    # first get the count of symbols in all files
    symbol_count = get_symbol_counts()

    # now split the files into two sets based on these priors
    training_files, testing_files = test_train_split(file_list, symbol_count)

    create_csv(training_files, 'training.csv')
    create_csv(testing_files, 'testing.csv')

    print('# of Training Files', len(training_files))
    print('# of Testing Files', len(testing_files))


if __name__ == '__main__':
    main()
