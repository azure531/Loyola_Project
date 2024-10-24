import sys
sys.path.append("../src")  # Adds higher directory to python modules path.

from early_text_classifier import EarlyTextClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('poster')

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

# Update the dataset path and name according to your dataset structure
etc_kwargs = {
    'dataset_path': '../dataset/offensive_messages.csv',  # Path to your dataset
    'dataset_name': 'offensive_messages',  # Dataset name
    'initial_step': 1,
    'step_size': 1
}

preprocess_kwargs = {
    'min_word_length': 2,
    'max_number_words': 10000  # Adjust based on your data
}

# Classifier for the partial information classifier (CPI)
cpi_clf = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=False)
cpi_kwargs = {
    'train_dataset_percentage': 0.75,
    'test_dataset_percentage': 0.25,
    'doc_rep': 'term_frec',  # Change if you have a different representation
    'cpi_clf': cpi_clf
}

# Context parameters
context_kwargs = {
    'number_most_common': 25  # Adjust this number based on your needs
}

# Classifier for the document model classifier (DMC)
dmc_clf = LogisticRegression(C=2, solver='liblinear', n_jobs=1, random_state=0)
dmc_kwargs = {
    'train_dataset_percentage': 0.75,
    'test_dataset_percentage': 0.25,
    'dmc_clf': dmc_clf
}

# Create an instance of EarlyTextClassifier with the updated parameters
etc = EarlyTextClassifier(etc_kwargs, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs)

# Print parameters information
etc.print_params_information()

# Preprocess the dataset
Xtrain, ytrain, Xtest, ytest = etc.preprocess_dataset()

# Fit the model on the training data
etc.fit(Xtrain, ytrain)

# Make predictions
cpi_perc, cpi_pred, dmc_pred, pred_time, dmc_ytest = etc.predict(Xtest, ytest)

# Calculate accuracy for CPI and DMC models
num_steps = len(cpi_perc)
x = cpi_perc
accuracy_cpi = np.sum(cpi_pred == ytest, axis=1) / ytest.size
y_cpi = np.zeros(num_steps)
y_dmc = np.zeros(num_steps)

for idx in range(num_steps):
    y_cpi[idx] = np.sum(cpi_pred[idx] == ytest) / ytest.size
    y_dmc[idx] = np.sum(dmc_pred[idx] == dmc_ytest[idx]) / dmc_ytest[idx].size

# Plotting accuracy vs percentage of document read
plt.plot(x, y_cpi, label='CPI Model')
plt.plot(x, y_dmc, label='DMC Model')
plt.xlabel('Percentage of Document Read')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy vs Document Read Percentage')
plt.show()

# Scoring the model with penalization parameters
penalization_type = 'Losada-Crestani'  # Change this based on your requirements
time_threshold = 30  # Adjust this threshold as needed
costs = {
    'c_tp': 1.0,  # True positives cost
    'c_fn': 1.0,  # False negatives cost
    'c_fp': 1.0   # False positives cost
}

# Score the model based on the predictions
etc.score(ytest, cpi_pred, cpi_perc, pred_time, penalization_type, time_threshold, costs)

# Saving model
