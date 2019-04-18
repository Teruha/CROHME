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

    if len(x_test) != 0:
        dropped_x_test = x_test.drop(list(['SYMBOL_REPRESENTATION', 'UI']), axis=1)
        rfc_pred = rfc.predict(dropped_x_test)
        print_top_n_predictions(rfc, dropped_x_test, 10, True, x_test)
        with open(CONST.PREDICTION_GROUND_TRUTH_CSV, 'w+') as f:
            for _, row in x_test.iterrows():
                f.write('{0},{1}\n'.format(row['UI'], row['SYMBOL_REPRESENTATION']))
        
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
    
    if len(x_test) != 0:
        dropped_x_test = x_test.drop(list(['SYMBOL_REPRESENTATION', 'UI']), axis=1)
        kdc_pred = kdc.predict(dropped_x_test)
        print_top_n_predictions(kdc, dropped_x_test, 10, False, x_test)
        print('KDTree Classifier results:')
        if len(x_train) < 1000:
            print('Confusion Matrix: ')
            print(confusion_matrix(y_test, kdc_pred))
        print('Classification Report: ')
        print(classification_report(y_test, kdc_pred))

def print_top_n_predictions(model, test_data, n=10, is_rfc=False, x_test=None):
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

    if is_rfc:
        with open(CONST.RFC_RECOGNITION_RESULTS_CSV, 'w+') as f:
            unique_identifiers = x_test['UI']
            for u, guess in zip(unique_identifiers, model.classes_[top_n]):
                preds = ','.join(guess)
                f.write('{0},{1}\n'.format(u, preds))
    else:
        with open(CONST.KD_TREE_RECOGNITION_RESULTS_CSV, 'w+') as f:
            unique_identifiers = x_test['UI']
            for u, guess in zip(unique_identifiers, model.classes_[top_n]):
                preds = ','.join(guess)
                f.write('{0},{1}\n'.format(u, preds))

    # print(model.classes_[top_n])

def main():
    data_file_to_load = CONST.DATA_FRAME_FILE_NAME
    try:
        df = pd.read_pickle(data_file_to_load)
        print('Loaded ' + data_file_to_load + ' from memory.')
    except FileNotFoundError:
        symbol_files = read_training_symbol_directory()
        junk_files = read_training_junk_directory()
        df = build_training_data(symbol_files, junk_files) # TODO: Replace empty array with junk files when ready to test both
        os.chdir('../..')
        df.to_pickle(data_file_to_load)
    
    x_train, x_test, y_train, y_test = split_data(df)
    run_random_forest_classifier(x_train, x_test, y_train, y_test, 200, CONST.RFC_IMPURITY_CRITERION[0])
    run_KDtree_classifier(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    main()
