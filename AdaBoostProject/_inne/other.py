from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import numpy as np
import os

# Define function to load data
def load_data(source_directory):
    features_list = []
    labels = []

    for filename in os.listdir(source_directory):
        if filename.endswith("x24.txt"):
            new_file_dir = os.path.join(source_directory, filename)
            with open(new_file_dir, 'r') as f:
                num_examples = int(f.readline())
                num_pixels = int(f.readline())
                data = np.loadtxt(f, max_rows=num_examples)
                features_list.extend(data[:, :num_pixels])
                labels.extend(data[:, -1])

    return np.array(features_list), np.array(labels)

# Define data directories
working_directory = os.getcwd()
data_source_directory = os.path.join(working_directory, 'famous48')

# Load data
X, y = load_data(data_source_directory)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Create and train AdaBoost model with SGDClassifier
clf = make_pipeline(StandardScaler(),
                    AdaBoostClassifier(estimator=SGDClassifier(max_iter=1000, tol=1e-3), n_estimators=300, algorithm='SAMME'))
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
