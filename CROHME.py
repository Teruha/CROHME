import os
import numpy as np
import pandas as pd
import constants as CONST

from file_manipulation import read_training_junk_directory, read_training_symbol_directory
from data_manipulation import build_training_data, split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix
from joblib import load, dump

def run_random_forest_classifier(x_train, x_test, y_train, y_test, n_estimators=100, criterion=CONST.RFC_IMPURITY_CRITERION[0]):
    """
    Run the Random Forest Classifier on the processed data and output the results.
    NOTE: This function creates a new pickle file if no such file exists or reads an existing one

    Parameters:
    x_train (pandas.series) - Training samples building the model
    x_test (pandas.series) - Testing samples 
    y_train (pandas.series) - Training samples building the model
    y_test (pandas.series) - Testing samples

    Returns:
    None
    """
    rfc = None
    file_name = '{0}_{1}_{2}.pkl'.format(CONST.RFC_MODEL_FILE_NAME, n_estimators, criterion)
    try:
        with open(file_name, 'rb') as f: 
            print('Loading RFC from memory.')
            rfc = load(f)
    except FileNotFoundError:
        rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
        rfc.fit(x_train, y_train)
        with open(file_name, 'wb') as f:
            dump(rfc, f, compress=True)

    rfc_pred = rfc.predict(x_test)
    print_top_n_predictions(rfc, x_test)

    print('Random Forest Classifier results:')
    if len(x_train) < 1000:
        print('Confusion Matrix: ')
        print(confusion_matrix(y_test, rfc_pred))
    print('Classification Report: ')
    print(classification_report(y_test, rfc_pred))

    
def run_KDtree_classifier(x_train, x_test, y_train, y_test):
    """
    Run the KD Tree Classifier on the processed data and output the results.
    NOTE: This function creates a new pickle file if no such file exists or reads an existing one

    Parameters:
    x_train (pandas.series) - Training samples building the model
    x_test (pandas.series) - Testing samples 
    y_train (pandas.series) - Training samples building the model
    y_test (pandas.series) - Testing samples

    Returns:
    None
    """
    kdc = None
    try: 
        with open(CONST.KD_TREE_MODEL_NAME, 'rb') as f: 
            print('Loading KDC from memory.')
            kdc = load(f)
    except FileNotFoundError:
        kdc = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
        kdc.fit(x_train, y_train)
        with open(CONST.KD_TREE_MODEL_NAME, 'wb') as f:
            dump(kdc, f,compress=True)
    
    kdc_pred = kdc.predict(x_test)
    print_top_n_predictions(kdc, x_test)

    print('KDTree Classifier results:')
    if len(x_train) < 1000:
        print('Confusion Matrix: ')
        print(confusion_matrix(y_test, kdc_pred))
    print('Classification Report: ')
    print(classification_report(y_test, kdc_pred))

def print_top_n_predictions(model, test_data, n=10):
    """
    Print the top n predictions of a sample based on the probability the sample belongs to each class

    Parameters:
    rfc (Sklearn.model) - Trained random forest classifier
    test_data (Dataframe) - Subset of the training data used for testing 

    Returns:
    None
    """
    prediciton_probabilities = model.predict_proba(test_data)
    top_n = np.argsort(prediciton_probabilities)[:,:-n-1:-1]
    print(model.classes_[top_n])

def main():
    try:
        df = pd.read_pickle(CONST.DATA_FRAME_FILE_NAME)
    except FileNotFoundError:
        symbol_files = read_training_symbol_directory()
        junk_files = read_training_junk_directory()
        df = build_training_data(symbol_files, []) # TODO: Replace empty array with junk files when ready to test both
        os.chdir('../..')
        df.to_pickle(CONST.DATA_FRAME_FILE_NAME)
    
    x_train, x_test, y_train, y_test = split_data(df)
    run_random_forest_classifier(x_train, x_test, y_train, y_test, 100, CONST.RFC_IMPURITY_CRITERION[0])
    run_KDtree_classifier(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    main()
