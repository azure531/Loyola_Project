import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PreprocessDataset:
    def __init__(self, preprocess_kwargs, verbose=True):
        self.min_word_length = preprocess_kwargs['min_word_length']
        self.max_number_words = preprocess_kwargs['max_number_words']
        self.representation = 'word_tf'
        self.dictionary = None
        self.unique_labels = None
        self.verbose = verbose

    def verboseprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def read_raw_dataset(self, path):
        """
        Read raw dataset and returns a list of tuples (label, document) for every document in the dataset.
        """
        dataset = []
        dataframe = pd.read_csv(path)

        for _, row in dataframe.iterrows():
            try:
                label = row['oh_label']  # Use 'oh_label' for labels
                document = row['Text']    # Use 'Text' for the message content
            except KeyError as e:
                print(f'Error while reading dataset: {e}')
                continue
            
            dataset.append((label, document))
        
        return dataset

    def build_dict(self, path):
        """
        Returns a dictionary with the words of the dataset.
        """
        dataset = self.read_raw_dataset(path)
        documents = [x[1] for x in dataset]

        self.verboseprint('Building dictionary')
        wordcount = dict()
        for ss in documents:
            words = ss.strip().lower().split()
            for w in words:
                if len(w) < self.min_word_length:
                    continue
                if w not in wordcount:
                    wordcount[w] = 1
                else:
                    wordcount[w] += 1

        counts = list(wordcount.values())
        keys = list(wordcount.keys())

        self.verboseprint(np.sum(counts), ' total words ', len(keys), ' unique words')

        sorted_idx = np.argsort(counts)[::-1]
        worddict = dict()

        for idx, ss in enumerate(sorted_idx):
            if (self.max_number_words != 'all') and (self.max_number_words <= idx):
                self.verboseprint(f'Considering only {self.max_number_words} unique terms plus the UNKOWN token')
                break
            worddict[keys[ss]] = idx + 1  # leave 0 (UNK)
        
        self.dictionary = worddict

        # Visualization of word frequencies
        self.visualize_word_frequencies(wordcount)

        return worddict

    def visualize_word_frequencies(self, wordcount):
        """Visualize the top 20 most frequent words."""
        top_words = sorted(wordcount.items(), key=lambda x: x[1], reverse=True)[:20]
        words, counts = zip(*top_words)

        plt.figure(figsize=(10, 6))
        plt.bar(words, counts, color='skyblue')
        plt.xticks(rotation=45)
        plt.title('Top 20 Most Frequent Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def transform_into_numeric_array(self, path):
        """
        Transform a list of documents into a numpy array of shape (num_docs, max_length+1).
        """
        dataset = self.read_raw_dataset(path)
        num_docs = len(dataset)

        seqs = [None] * num_docs
        max_length = 0
        for idx, line in enumerate(dataset):
            document = line[1]
            words = document.strip().lower().split()
            seqs[idx] = [self.dictionary[w] if w in self.dictionary else 0 for w in words]
            length_doc = len(words)
            if max_length < length_doc:
                max_length = length_doc

        preprocess_dataset = -2 * np.ones((num_docs, max_length + 1), dtype=int)
        for idx in range(num_docs):
            length_doc = len(seqs[idx])
            preprocess_dataset[idx, 0:length_doc] = seqs[idx]
            preprocess_dataset[idx, length_doc] = -1

        # Visualization of the shape of the numeric array
        self.visualize_numeric_array_shape(preprocess_dataset)

        return preprocess_dataset

    def visualize_numeric_array_shape(self, numeric_array):
        """Visualize the shape of the numeric array."""
        plt.figure(figsize=(10, 6))
        plt.hist([len(doc) for doc in numeric_array], bins=30, color='lightgreen', edgecolor='black')
        plt.title('Distribution of Document Lengths in Numeric Array')
        plt.xlabel('Document Length')
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def get_labels(self, path):
        """
        Read raw dataset and returns a tuple (final_labels, unique_labels).
        """
        labels = []
        ul = [] if self.unique_labels is None else self.unique_labels
        dataframe = pd.read_csv(path)

        for _, row in dataframe.iterrows():
            try:
                label = row['oh_label']
            except KeyError as e:
                print(f'Error while reading dataset: {e}')
                continue
            labels.append(label)
            if (self.unique_labels is None) and (label not in ul):
                ul.append(label)

        ul.sort()
        num_documents = len(labels)
        final_labels = np.empty([num_documents], dtype=int)
        for idx, l in enumerate(labels):
            final_labels[idx] = ul.index(l)

        self.unique_labels = ul
        return final_labels, ul

# Usage
if __name__ == "__main__":
    preprocess_kwargs = {
        'min_word_length': 3,
        'max_number_words': 'all'
    }
    
    processor = PreprocessDataset(preprocess_kwargs, verbose=True)
    
    # Path to your dataset
    dataset_path = 'offensive_messages.csv'  # Update this to the correct path

    # Build dictionary
    word_dict = processor.build_dict(dataset_path)

    # Transform into numeric array
    numeric_array = processor.transform_into_numeric_array(dataset_path)

    # Get labels
    final_labels, unique_labels = processor.get_labels(dataset_path)

    print("Word Dictionary:", word_dict)
    print("Numeric Array Shape:", numeric_array.shape)
    print("Final Labels:", final_labels)
    print("Unique Labels:", unique_labels)
