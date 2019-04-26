import os
import numpy as np
import pandas as pd
import sys
import constants as CONST

from file_manipulation import get_inkml_files, load_files_to_dataframe, build_training_data, split_data
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
    if not os.path.isdir('models'):
        os.mkdir('models')
    rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    rfc.fit(x_train, y_train) 
    curr_path = os.path.dirname(os.path.abspath(__file__))
    models_path = curr_path + '/models/'
    if not os.path.exists(models_path):
        os.mkdir(models_path)
    file_name = '{0}_{1}_{2}.pkl'.format(CONST.RFC_MODEL_FILE_NAME, n_estimators, criterion)
    original_folder = os.getcwd()
    os.chdir(models_path)
    if os.path.exists(file_name):
        os.remove(file_name)
    os.chdir(original_folder)
    with open(models_path + file_name, 'wb') as f:
        dump(rfc, f, compress=True)
        print('Saved model "{0}" to {1}'.format(file_name, models_path))
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
    curr_path = os.path.dirname(os.path.abspath(__file__))
    models_path = curr_path + '/models/'
    with open(models_path + file_name, 'rb') as f: 
        print('Loading RFC from memory.')
        rfc = load(f)
        rfc_pred = rfc.predict(x_test)
        # if len(x_test) < 1000:
        #     print('Confusion Matrix: ')
        #     print(confusion_matrix(y_test, rfc_pred))
        # print('Classification Report: ')
        # print(classification_report(y_test, rfc_pred))
        # print_top_n_predictions(rfc, 10, True, x_test)
        
        return rfc_pred

def print_top_n_predictions(model, n=10, is_rfc=False, x_test=None):
    """
    Print the top n predictions of a sample based on the probability the sample belongs to each class

    Parameters:
    rfc (Sklearn.model) - Trained random forest classifier

    Returns:
    None
    """
    prediciton_probabilities = model.predict_proba(x_test)
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
    print(model.classes_[top_n])

def classification_main():
    if len(sys.argv) == 1:
        print('USAGE: [[python3]] CROHME.PY [training_dir] [testing_dir] [(-tr)ain|(-te)st|(-b)oth]')
        print('Ex. 1: python3 CROHME.PY [training_symbols_dir OR .pkl file] [testing_symbols_dir OR .pkl file] -b')
        print('Ex. 2: python3 CROHME.PY [training_symbols_dir OR .pkl file] <ground_truth_file> -tr')
        print('Ex. 2: python3 CROHME.PY [testing_symbols_dir OR .pkl file]  -te')
    elif len(sys.argv) == 3 or len(sys.argv) == 4:
        if sys.argv[-1] == '-tr': # train the model, this means we are creating a new one
            if len(sys.argv) == 4:
                df = load_files_to_dataframe(sys.argv[1], sys.argv[2]) # with ground truth files
            else:
                df = load_files_to_dataframe(sys.argv[1]) # without ground truth files
            x_train = df.drop(list(['SYMBOL_REPRESENTATION', 'UI', 'TRACES']), axis=1) 
            y_train = df['SYMBOL_REPRESENTATION']
            # x_train, _, y_train, _ = split_data(df)
            train_random_forest_classifier(x_train, y_train, 200)
        elif sys.argv[-1] == '-te': # test the model, this means it already exists
            df = load_files_to_dataframe(sys.argv[1])
            _, x_test, _, y_test = split_data(df)
            test_random_forest_classifier(x_test, y_test, 200)
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
