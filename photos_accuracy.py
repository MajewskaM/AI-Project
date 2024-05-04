import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

# Load dataset from specified "source" directory
def load_data(source_directory, num_photos):
    features_list = []
    labels = []

    # Iterate over all the files in the source directory
    for filename in os.listdir(source_directory):
        # For this project, files with source data end with: 'x24.txt'
        if filename.endswith("x24.txt"):
            new_file_dir = os.path.join(source_directory, filename)
            with open(new_file_dir, 'r') as f:
                num_examples = int(f.readline())
                num_pixels = int(f.readline())

                for _ in range(num_examples):
                    example = list(map(float, f.readline().split()))
                    label = int(example[num_pixels + 2])
                    features = example[:num_pixels]
                    labels.append(label)
                    features_list.append(features)

    features_list = np.array(features_list)
    labels = np.array(labels)

    # Select subset of photos
    indices = np.random.choice(len(features_list), num_photos, replace=False)
    subset_features = features_list[indices]
    subset_labels = labels[indices]

    return subset_features, subset_labels

# Define source directory
working_directory = os.getcwd()
data_source_directory = os.path.join(working_directory, 'famous48')

# Define different numbers of photos
num_photos_list = [10, 20, 30, 40, 48]

# Train AdaBoost classifier and evaluate accuracy for each subset size
accuracies = []
for num_photos in num_photos_list:
    # Load subset of dataset
    X, y = load_data(data_source_directory, num_photos)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    # Train AdaBoost classifier
    ada_clf = AdaBoostClassifier(estimator=SGDClassifier, n_estimators=300, algorithm='SAMME')
    ada_clf.fit(X_train, y_train)

    # Evaluate accuracy on testing set
    y_pred = ada_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plot accuracy for each subset size
plt.plot(num_photos_list, accuracies, marker='o')
plt.xlabel('Number of Photos')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Photos')
plt.grid(True)
plt.show()
