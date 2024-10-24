import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier  # Importing the correct classifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class PartialInformationClassifier:
    def __init__(self, cpi_kwargs, dictionary, verbose=True):
        self.random_state = np.random.RandomState(1234)
        self.train_dataset_percentage = cpi_kwargs['train_dataset_percentage']
        self.test_dataset_percentage = cpi_kwargs['test_dataset_percentage']
        self.doc_rep = cpi_kwargs['doc_rep']
        self.dictionary = dictionary
        self.initial_step = cpi_kwargs['initial_step']
        self.step_size = cpi_kwargs['step_size']
        self.clf = cpi_kwargs['cpi_clf']
        self.verbose = verbose

    def verboseprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def split_dataset(self, Xtrain, ytrain):
        self.verboseprint("Splitting preprocessed dataset for the PartialInformationClassifier")
        ss = ShuffleSplit(train_size=self.train_dataset_percentage, test_size=self.test_dataset_percentage,
                          random_state=self.random_state)
        idx_train, idx_test = next(ss.split(X=Xtrain, y=ytrain))
        cpi_Xtrain, cpi_ytrain = Xtrain[idx_train], ytrain[idx_train]
        cpi_Xtest, cpi_ytest = Xtrain[idx_test], ytrain[idx_test]
        return cpi_Xtrain, cpi_ytrain, cpi_Xtest, cpi_ytest

    def get_document_representation(self, data):
        num_docs = len(data)
        num_features = len(self.dictionary) + 1  # Considering the UNKNOWN token.
        i = []  # Initialize i as an empty list
        j = []  # Initialize j as an empty list
        v = []  # Initialize v as an empty list

        if self.doc_rep == 'term_frec':
            for idx, row in enumerate(data):
                unique, counts = np.unique(row, return_counts=True)

                # Remove any invalid tokens
                index_to_delete = np.where((unique < 0) | (unique >= num_features))
                unique = np.delete(unique, index_to_delete)
                counts = np.delete(counts, index_to_delete)

                # Ensure we are still within the range
                if unique.size > 0:
                    i.extend([idx] * len(unique))
                    j.extend(unique.tolist())
                    v.extend(counts.tolist())
                else:
                    print(f"Warning: No valid tokens in document index {idx}")

        # Create sparse matrix if there are any values to process
        if len(i) > 0 and len(j) > 0 and len(v) > 0:
            sparse_matrix = sparse.coo_matrix((v, (i, j)), shape=(num_docs, num_features)).tocsr()
        else:
            sparse_matrix = sparse.coo_matrix((num_docs, num_features)).tocsr()  # Create an empty sparse matrix

        return sparse_matrix

    def fit(self, Xtrain, ytrain):
        self.verboseprint("Training PartialInformationClassifier")
        Xtrain = self.get_document_representation(Xtrain)
        self.verboseprint(f'cpi_Xtrain_representation.shape: {Xtrain.shape}')
        self.clf.fit(Xtrain, ytrain)

    def predict(self, Xtest, ytest):
        self.verboseprint("Predicting with PartialInformationClassifier")
        num_docs = len(Xtest)
        _, docs_len = np.where(Xtest == -1)  # Finding the end of the document
        percentages = []
        preds = []
        accuracies = []

        for p in range(self.initial_step, 101, self.step_size):
            docs_partial_len = np.round(docs_len * p / 100).astype(int)
            max_length = np.max(docs_partial_len)
            partial_Xtest = -2 * np.ones((num_docs, max_length + 1), dtype=int)

            for idx, pl in enumerate(docs_partial_len):
                partial_Xtest[idx, 0:pl] = Xtest[idx, 0:pl]
                partial_Xtest[idx, pl] = -1  # Marking the end of the partial document

            partial_Xtest = self.get_document_representation(partial_Xtest)
            predictions_test = self.clf.predict(partial_Xtest)

            # Store results
            percentages.append(p)
            preds.append(predictions_test)
            accuracies.append(accuracy_score(ytest, predictions_test))  # Calculate accuracy

        return preds, percentages, accuracies

    def visualize_predictions(self, percentages, accuracies):
        plt.figure(figsize=(10, 5))
        plt.plot(percentages, accuracies, marker='o')
        plt.title('Accuracy vs Percentage of Document Length Used')
        plt.xlabel('Percentage of Document Length Used')
        plt.ylabel('Accuracy')
        plt.xticks(np.arange(0, 101, 10))
        plt.grid()
        plt.show()


# Example parameters and usage
cpi_kwargs = {
    'train_dataset_percentage': 0.8,
    'test_dataset_percentage': 0.2,
    'doc_rep': 'term_frec',
    'initial_step': 10,
    'step_size': 10,
    'cpi_clf': DecisionTreeClassifier()  # Replace with an actual classifier instance
}

dictionary = {}  # Replace with your actual dictionary

# Initialize the classifier
classifier = PartialInformationClassifier(cpi_kwargs, dictionary)

# Assuming you have your data prepared
# Example dummy data (replace with your actual data)
Xtrain = np.array([[1, 2, -1], [1, 3, -1], [2, 3, -1]])
ytrain = np.array([0, 1, 0])
Xtest = np.array([[1, 2, -1], [2, 3, -1]])
ytest = np.array([0, 1])

# Fit the model
classifier.fit(Xtrain, ytrain)

# Predict and visualize
preds, percentages, accuracies = classifier.predict(Xtest, ytest)
classifier.visualize_predictions(percentages, accuracies)
