import os
import numpy as np
import pandas as pd
import sys
import classification.constants as CONST

from classification.file_manipulation import get_inkml_files
from classification.data_manipulation import build_training_data, split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix
from joblib import load, dump

def train_random_forest_classifier(x_train, y_train, n_estimators=100, criterion=CONST.RFC_IMPURITY_CRITERION[0]):
    """
    Train and return a random forest classifier

    Parameters:
    1. x_train (pandas.series) - Training samples building the model
    2. y_train (pandas.series) - Training samples building the model
    3. n_estimators (int) - The size of the random forest ensemble
    4. criterion (str) - The criterion for which the Random Forest Classifier uses to calculate impurity  

    Returns:
    1. rfc (sklearn.model) - Random Forest Classifier Scikit learn classifier
    """
    x_train.drop(['UI','TRACES'], axis=1, inplace=True)
    rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    rfc.fit(x_train, y_train)
    file_name = '{0}_{1}_{2}.pkl'.format(CONST.RFC_MODEL_FILE_NAME, n_estimators, criterion)
    with open(file_name, 'wb') as f:
        dump(rfc, f, compress=True)
    return rfc

def test_random_forest_classifier(x_test, y_test, n_estimators=100, criterion=CONST.RFC_IMPURITY_CRITERION[0]):
    """
    Test the random forest classifier given test data

    Parameters:
    1. x_test (pandas.series) - Testing samples 
    2. y_test (pandas.series) - Testing samples

    Returns:
    1. None
    """
    file_name = '{0}_{1}_{2}.pkl'.format(CONST.RFC_MODEL_FILE_NAME, n_estimators, criterion)
    with open(file_name, 'rb') as f: 
        print('Loading RFC from memory.')
        rfc = load(f)
        rfc_pred = rfc.predict(x_test)
        if len(x_test) < 1000:
            print('Confusion Matrix: ')
            print(confusion_matrix(y_test, rfc_pred))
        print('Classification Report: ')
        print(classification_report(y_test, rfc_pred))
        return rfc_pred

def run_random_forest_classifier(x_train, x_test, y_train, y_test, n_estimators=100, criterion=CONST.RFC_IMPURITY_CRITERION[0]):
    """
    Run the Random Forest Classifier on the processed data and output the results.
    NOTE: This function creates a new pickle file if no such file exists or reads an existing one

    Parameters:
    x_train (pandas.series) - Training samples building the model
    
    y_train (pandas.series) - Training samples building the model
    

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

def load_files_to_dataframe(dir_name, second_dir=None, segmentation=False, save=True):
    df = None
    df2 = None
    if '.pkl' in dir_name:
        df = pd.read_pickle(dir_name)
    else: # read file(s) from input directory
        df = build_training_data(get_inkml_files(dir_name), segment_data_files=segmentation)
        if save:
            df.to_pickle(dir_name + '.pkl')
    if not second_dir:
        return df

    if '.pkl' in second_dir:
        df2 = pd.read_pickle(second_dir)
    else: # read file(s) from input directory
        df2 = build_training_data(get_inkml_files(second_dir), segment_data_files=segmentation)
        if save:
            df2.to_pickle(second_dir + '.pkl')
    return df, df2

def classification_main():
    if len(sys.argv) == 1:
        print('USAGE: [[python3]] CROHME.PY [training_dir] [testing_dir] [(-tr)ain|(-te)st|(-b)oth]')
        print('Ex. 1: python3 CROHME.PY [training_symbols_dir OR .pkl file] [testing_symbols_dir OR .pkl file] -b')
        print('Ex. 2: python3 CROHME.PY [training_symbols_dir OR .pkl file] -tr')
        print('Ex. 2: python3 CROHME.PY [testing_symbols_dir OR .pkl file]  -te')
    elif len(sys.argv) == 3 or len(sys.argv) == 4:
        if sys.argv[-1] == '-tr': # train the model, this means we are creating a new one
            df = load_files_to_dataframe(sys.argv[1])
            x_train, _, y_train, _ = split_data(df)
            train_random_forest_classifier(x_train, y_train)
        elif sys.argv[-1] == '-te': # test the model, this means it already exists
            df = load_files_to_dataframe(sys.argv[1])
            _, x_test, _, y_test = split_data(df)
            test_random_forest_classifier(x_test, y_test)
        elif sys.argv[-1] == '-b': # test and train the model, this means we need to recreate the model and test it
            df, df2 = load_files_to_dataframe(sys.argv[1], sys.argv[2])
            x_train, _, y_train, _ = split_data(df, 0.00)
            _, x_test, _, y_test = split_data(df2, 1)
            train_random_forest_classifier(x_train, y_train)
            test_random_forest_classifier(x_test, y_test)
        else:
            print('ERROR: NO FLAG OR INVALID FLAG SPECIFIED.')
    else:
        print('INVALID PARAMETERS, PLEASE RUN FILE WITH NO PARAMETERS TO SEE USAGE.')


if __name__ == '__main__':
    classification_main()
