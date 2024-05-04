import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.svm as svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# load dataset from specified "source" directory
def load_data(source_directory):
    features_list = []
    labels = []

    # iterate over all the files in the source directory
    for filename in os.listdir(source_directory):

        # for this project files with source data ends with: 'x24.txt'
        if filename.endswith("x24.txt"):
            new_file_dir = os.path.join(source_directory, filename)
            with open(new_file_dir, 'r') as f:
                num_examples = int(f.readline())
                num_pixels = int(f.readline())

                for _ in range(num_examples):
                    example = list(map(float, f.readline().split()))
                    label = int(example[num_pixels + 2])

                    # filter out labels of 10 people
                    if label <= 9:
                        features = example[:num_pixels]
                        labels.append(label)
                        features_list.append(features)

    return np.array(features_list), np.array(labels)

# training classifiers and evaluating model
def evaluateResults(model, X_test, y_test, X_train, y_train, cv, name):
    # fitting the model to the training data
    model.fit(X_train, y_train)
    
    cv_train_score = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"On average, {name} model has f1 score of {cv_train_score.mean():3f} +/- {cv_train_score.std():.3f} on the training set.")

    # predict the class of test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy for each individual photo
    individual_accuracies = (y_pred == y_test)

    # Calculate accuracy per each person (label)
    label_accuracy = defaultdict(list)
    for i, accuracy in enumerate(individual_accuracies):
        label = y_test[i]
        label_accuracy[label].append(accuracy)

    # Calculate accuracy per each person as a percentage
    accuracy_percentage_per_person = {}
    for label, accuracies in label_accuracy.items():
        total_photos = len(accuracies)
        correct_predictions = sum(accuracies)
        accuracy_percentage = (correct_predictions / total_photos) * 100
        accuracy_percentage_per_person[label] = accuracy_percentage

    # Print the accuracy per each person (label) in percentage
    print("Accuracy per each person (label) in percentage:")
    for label, accuracy_percentage in accuracy_percentage_per_person.items():
        print(f"Person {label}: {accuracy_percentage:.2f}%")

    # Overall accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)
    print("Model Evaluation - accuracy:", overall_accuracy)

    prepareReport(y_test, y_pred)

# preparing report about the model predictions accuracy
def prepareReport(y_test, y_pred):
    
    print("#Classification report")
    print(classification_report(y_test, y_pred))

    print("#Confusion matrix")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    #ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()


# get current working directory
working_directory = os.getcwd()
data_source_directory = os.path.join(working_directory, 'famous48')

# loading data: extracting features and labels of categories
X, y = load_data(data_source_directory)

# Calculate the mean number of photos of each person (labels 0-9)
mean_photos_per_person = len(y) / 10
print("Mean number of photos of each person:", mean_photos_per_person)

# splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)

print("Shape of training data: ", X_train.shape)
print("Shape of training label: ", y_train.shape)
print("Shape of test data: ", X_test.shape)
print("Shape of test label: ", y_test.shape)

# Define the parameter grid for LogisticRegression
param_grid_lr = {
    'logisticregression__C': [0.1, 1, 10, 100],
    'logisticregression__penalty': ['l1', 'l2', 'elasticnet'],
    'logisticregression__l1_ratio':[0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.13,0.14,0.15,0.2]
}
# Create a pipeline with scaler and logistic regression
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='saga', tol=0.1))

# Perform grid search for LogisticRegression
grid_search_lr = GridSearchCV(pipe, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1, verbose = 2).fit(X_train, y_train)

# Print the best parameters
print("Best parameters for LogisticRegression:", grid_search_lr.best_params_)
print("Best Score for LogisticRegression:", grid_search_lr.best_score_)

# Evaluate the results
evaluateResults(grid_search_lr.best_estimator_, X_test, y_test, X_train, y_train, ShuffleSplit(n_splits=10, test_size=0.1, random_state=42), "LogisticRegression")