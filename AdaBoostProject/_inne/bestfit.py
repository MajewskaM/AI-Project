### same for adaboost

from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(source_directory):
    features_list = []
    labels = []

    # iterate over the files in the source directory
    for filename in os.listdir(source_directory):
        # check if the file is a text file - desired ending with 'x24.txt'
        if filename.endswith("x24.txt"):
            new_file_dir = os.path.join(source_directory, filename)
            with open(new_file_dir, 'r') as f:
                num_examples = int(f.readline())
                num_pixels = int(f.readline())

                for _ in range(num_examples):
                    example = list(map(float, f.readline().split()))
                    features = example[:num_pixels]
                    labels.append(int(example[num_pixels + 2]))
                    features_list.append(features)

    return np.array(features_list), np.array(labels)


# current working directory
working_directory = os.getcwd()
data_source_directory = os.path.join(working_directory, 'famous48')


# loading data: extracting features and labels of categories
# data_directory = os.path.join()
X, y = load_data(data_source_directory)


# splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Define parameter grid with different learning rates for AdaBoost
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 1]  # Learning rate for AdaBoost
}

# Create AdaBoostClassifier with SGDClassifier as base estimator
sgd_base_clf = SGDClassifier()  # SGDClassifier as the base estimator
ada_sgd_clf = AdaBoostClassifier(estimator=sgd_base_clf, algorithm='SAMME')  # AdaBoostClassifier with SGDClassifier

# Perform grid search cross-validation
grid_search = GridSearchCV(ada_sgd_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print("Best parameters found:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)
