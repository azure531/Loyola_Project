import pandas as pd
import glob
from preprocess_data import PreprocessDataset
from context_information import ContextInformation
from partial_information_classifier import PartialInformationClassifier
from decision_classifier import DecisionClassifier
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, confusion_matrix, classification_report
import pprint as pp
import pickle
import os


class EarlyTextClassifier:
    def __init__(self, etc_kwargs, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs, unique_labels=None,
                 dictionary=None, verbose=True):
        self.dataset_path = etc_kwargs['dataset_path']  # Path to your dataset
        self.initial_step = etc_kwargs['initial_step']
        self.step_size = etc_kwargs['step_size']
        self.preprocess_kwargs = preprocess_kwargs
        self.cpi_kwargs = cpi_kwargs
        self.context_kwargs = context_kwargs
        self.dmc_kwargs = dmc_kwargs
        self.cpi_kwargs['initial_step'] = self.initial_step
        self.cpi_kwargs['step_size'] = self.step_size
        self.context_kwargs['initial_step'] = self.initial_step
        self.context_kwargs['step_size'] = self.step_size
        self.dictionary = dictionary
        self.ci = None
        self.cpi = None
        self.dmc = None
        self.unique_labels = unique_labels
        self.is_loaded = False
        self.verbose = verbose

        self.load_model()

    def verboseprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def has_same_parameters(self, model):
        if (self.dataset_path == model.dataset_path) and \
                (self.initial_step == model.initial_step) and \
                (self.step_size == model.step_size) and \
                (self.preprocess_kwargs == model.preprocess_kwargs) and \
                (self.cpi_kwargs['train_dataset_percentage'] == model.cpi_kwargs['train_dataset_percentage']) and \
                (self.cpi_kwargs['test_dataset_percentage'] == model.cpi_kwargs['test_dataset_percentage']) and \
                (self.cpi_kwargs['doc_rep'] == model.cpi_kwargs['doc_rep']) and \
                (self.cpi_kwargs['cpi_clf'].get_params() == model.cpi_kwargs['cpi_clf'].get_params()) and \
                (self.context_kwargs == model.context_kwargs) and \
                (self.dmc_kwargs['train_dataset_percentage'] == model.dmc_kwargs['train_dataset_percentage']) and \
                (self.dmc_kwargs['test_dataset_percentage'] == model.dmc_kwargs['test_dataset_percentage']) and \
                (self.dmc_kwargs['dmc_clf'].get_params() == model.dmc_kwargs['dmc_clf'].get_params()):
            return True
        else:
            return False

    def copy_attributes(self, model):
        self.ci = model.ci
        self.cpi = model.cpi
        self.dmc = model.dmc

    def load_model(self):
        possible_files = glob.glob(f'models/*.pickle')
        for file in possible_files:
            with open(file, 'rb') as f:
                loaded_model = pickle.load(f)
                if self.has_same_parameters(loaded_model):
                    self.verboseprint('Model already trained. Loading it.')
                    self.copy_attributes(loaded_model)
                    self.is_loaded = True

    def print_params_information(self):
        print("Dataset path: {}".format(self.dataset_path))
        print('-' * 80)
        print('Pre-process params:')
        pp.pprint(self.preprocess_kwargs)
        print('-' * 80)
        print('CPI params:')
        pp.pprint(self.cpi_kwargs)
        print('-' * 80)
        print('Context Information params:')
        pp.pprint(self.context_kwargs)
        print('-' * 80)
        print('DMC params:')
        pp.pprint(self.dmc_kwargs)
        print('-' * 80)

    def preprocess_dataset(self):
        self.verboseprint('Pre-processing dataset')
        
        # Load the dataset from CSV
        df = pd.read_csv(self.dataset_path)
        
        # Assuming your CSV has 'text' and 'label' columns for features and targets
        # Adjust according to your actual column names
        X = df['text'].values  # Replace 'text' with your actual text column name
        y = df['label'].values  # Replace 'label' with your actual label column name

        # Initialize and preprocess dataset
        prep_data = PreprocessDataset(self.preprocess_kwargs, self.verbose)
        self.dictionary = prep_data.build_dict(X)

        X_numeric = prep_data.transform_into_numeric_array(X)
        self.unique_labels = np.unique(y)

        self.verboseprint(f'X.shape: {X_numeric.shape}')
        self.verboseprint(f'y.shape: {y.shape}')

        return X_numeric, y

    def fit(self, Xtrain, ytrain):
        if not self.is_loaded:
            self.verboseprint('Training EarlyTextClassifier model')
            self.ci = ContextInformation(self.context_kwargs, self.dictionary, self.verbose)
            self.ci.get_training_information(Xtrain, ytrain)

            self.cpi = PartialInformationClassifier(self.cpi_kwargs, self.dictionary, self.verbose)
            cpi_Xtrain, cpi_ytrain, cpi_Xtest, cpi_ytest = self.cpi.split_dataset(Xtrain, ytrain)

            self.verboseprint(f'cpi_Xtrain.shape: {cpi_Xtrain.shape}')
            self.verboseprint(f'cpi_ytrain.shape: {cpi_ytrain.shape}')

            self.cpi.fit(cpi_Xtrain, cpi_ytrain)
            cpi_predictions, cpi_percentages = self.cpi.predict(cpi_Xtest)

            dmc_X, dmc_y = self.ci.generate_dmc_dataset(cpi_Xtest, cpi_ytest, cpi_predictions)

            self.dmc = DecisionClassifier(self.dmc_kwargs, self.verbose)
            dmc_Xtrain, dmc_ytrain, dmc_Xtest, dmc_ytest = self.dmc.split_dataset(dmc_X, dmc_y)

            self.verboseprint(f'dmc_Xtrain.shape: {dmc_Xtrain.shape}')
            self.verboseprint(f'dmc_ytrain.shape: {dmc_ytrain.shape}')

            self.dmc.fit(dmc_Xtrain, dmc_ytrain)
            dmc_prediction, _ = self.dmc.predict(dmc_Xtest)
        else:
            self.verboseprint('EarlyTextClassifier model already trained')

    def predict(self, Xtest):
        self.verboseprint('Predicting with the EarlyTextClassifier model')
        cpi_predictions, cpi_percentages = self.cpi.predict(Xtest)
        dmc_X, dmc_y = self.ci.generate_dmc_dataset(Xtest, None, cpi_predictions)
        dmc_prediction, prediction_time = self.dmc.predict(dmc_X)
        return cpi_percentages, cpi_predictions, dmc_prediction, prediction_time, dmc_y

    def time_penalization(self, k, penalization_type, time_threshold):
        if penalization_type == 'Losada-Crestani':
            return 1.0 - ((1.0 + np.exp(k - time_threshold)) ** (-1))
        return 0.0

    def score(self, y_true, cpi_prediction, cpi_percentages, prediction_time, penalization_type, time_threshold, costs,
              print_output=True):
        y_pred = []
        k = []
        num_docs = len(y_true)
        for i in range(num_docs):
            t = prediction_time[i]
            p = cpi_percentages[prediction_time[i]]
            y_pred.append(cpi_prediction[t, i])
            k.append(p)
        y_pred = np.array(y_pred)
        k = np.array(k)

        error_score = np.zeros(num_docs)
        if len(self.unique_labels) > 2:
            for idx in range(num_docs):
                if y_true[idx] == y_pred[idx]:
                    error_score[idx] = self.time_penalization(k[idx], penalization_type, time_threshold) * costs['c_tp']
                else:
                    error_score[idx] = costs['c_fn'] + np.sum(y_true == y_true[idx]) / num_docs
        else:
            for idx in range(num_docs):
                if (y_true[idx] == 1) and (y_pred[idx] == 1):
                    error_score[idx] = self.time_penalization(k[idx], penalization_type, time_threshold) * costs['c_tp']
                elif (y_true[idx] == 1) and (y_pred[idx] == 0):
                    error_score[idx] = costs['c_fn']
                elif (y_true[idx] == 0) and (y_pred[idx] == 1):
                    error_score[idx] = costs['c_fp']
                else:
                    error_score[idx] = costs['c_tn']
        
        total_error = np.sum(error_score)
        if print_output:
            print("Recall: ", recall_score(y_true, y_pred, average='weighted'))
            print("Precision: ", precision_score(y_true, y_pred, average='weighted'))
            print("F1 Score: ", f1_score(y_true, y_pred, average='weighted'))
            print("Accuracy: ", accuracy_score(y_true, y_pred))
            print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred))
            print("Classification Report: \n", classification_report(y_true, y_pred))

        return total_error
