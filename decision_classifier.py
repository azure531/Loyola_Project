import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Change this if you prefer another classifier


class DecisionClassifier:
    def __init__(self, dmc_kwargs, verbose=True):
        self.train_dataset_percentage = dmc_kwargs['train_dataset_percentage']
        self.test_dataset_percentage = dmc_kwargs['test_dataset_percentage']
        self.clf = dmc_kwargs['dmc_clf']
        self.verbose = verbose
        self.vectorizer = TfidfVectorizer(max_features=10000)  # Adjust the number of features as needed

    def verboseprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def load_data(self, file_path):
        # Load the dataset
        self.verboseprint("Loading dataset from", file_path)
        data = pd.read_csv(file_path)
        X = data['Text'].values
        y = data['oh_label'].values  # Assuming 'oh_label' is your target variable

        # Vectorize the text data
        self.verboseprint("Vectorizing text data")
        X_vectorized = self.vectorizer.fit_transform(X).toarray()  # Convert sparse to dense array
        return X_vectorized, y

    def split_dataset(self, X, y):
        self.verboseprint("Splitting preprocessed dataset for the DecisionClassifier")
        num_docs = X.shape[0]
        num_training = int(np.round(num_docs * self.train_dataset_percentage))
        num_test = int(np.round(num_docs * self.test_dataset_percentage))
        if num_docs < num_training + num_test:
            self.verboseprint("The training-test splits must sum to one or less.")
        return X[:num_training], y[:num_training], X[num_training:], y[num_training:]

    def flatten_dataset(self, X, y):
        return X, y  # Flattening not needed for dense arrays

    def fit(self, Xtrain, ytrain):
        self.verboseprint("Training DecisionClassifier")
        Xtrain, ytrain = self.flatten_dataset(Xtrain, ytrain)
        self.clf.fit(Xtrain, ytrain)

    def predict(self, Xtest):
        self.verboseprint("Predicting with DecisionClassifier")
        predictions = self.clf.predict(Xtest)
        return predictions

    def evaluate(self, X_test, y_test):
        self.verboseprint("Evaluating DecisionClassifier")
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        self.verboseprint(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    # Parameters for the DecisionClassifier
    dmc_kwargs = {
        'train_dataset_percentage': 0.8,  # 80% for training
        'test_dataset_percentage': 0.2,    # 20% for testing
        'dmc_clf': LogisticRegression()     # Change this to your desired classifier
    }

    # Create an instance of DecisionClassifier
    classifier = DecisionClassifier(dmc_kwargs)

    # Load data
    X, y = classifier.load_data('offensive_messages.csv')
    
    # Split dataset
    X_train, y_train, X_test, y_test = classifier.split_dataset(X, y)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    classifier.evaluate(X_test, y_test)
