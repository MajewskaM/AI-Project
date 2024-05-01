import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# Load dataset from specified "source" directory
def load_data(source_directory):
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

                    # Filter out labels of 10 people
                    if label <= 9:
                        features = example[:num_pixels]
                        labels.append(label)
                        features_list.append(features)

    return np.array(features_list), np.array(labels)

# Training classifiers and evaluating model
def evaluate_results(model, X_test, y_test, X_train, y_train, cv, name):
    # Fitting the model to the training data
    model.fit(X_train, y_train)
    
    cv_train_score = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"On average, {name} model has f1 score of {cv_train_score.mean():3f} +/- {cv_train_score.std():.3f} on the training set.")

    # Predict the class of test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Evaluation - accuracy:", accuracy)

    prepare_report(y_test, y_pred)

# Preparing report about the model predictions accuracy
def prepare_report(y_test, y_pred):
    
    print("#Classification report")
    print(classification_report(y_test, y_pred))

    print("#Confusion matrix")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    #ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

# Perform grid search cross-validation
def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

# Get current working directory
working_directory = os.getcwd()
data_source_directory = os.path.join(working_directory, 'famous48')

# Loading data: extracting features and labels of categories
X, y = load_data(data_source_directory)

# Calculate the mean number of photos of each person (labels 0-9)
mean_photos_per_person = len(y) / 10
print("Mean number of photos of each person:", mean_photos_per_person)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)

print("Shape of training data: ", X_train.shape)
print("Shape of training label: ", y_train.shape)
print("Shape of test data: ", X_test.shape)
print("Shape of test label: ", y_test.shape)

# Base classifiers to be used in AdaBoost
base_classifiers = [LogisticRegression(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), SGDClassifier()]

# Perform grid search for AdaBoostClassifier
param_grid_ada = {'n_estimators': [10, 50, 100, 300, 500], 'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 1]}

# Plotting
fig, ax = plt.subplots()
accuracy_scores = []

for classifier in base_classifiers:
    ada_clf = AdaBoostClassifier(estimator=classifier, algorithm='SAMME')
    grid_search_ada = perform_grid_search(ada_clf, param_grid_ada, X_train, y_train)
    accuracy_scores.append(grid_search_ada.best_score_)

# Plotting the accuracy scores
x_labels = ['Logistic Regression', 'SVM', 'Decision Tree', 'RandomForestClassifier', 'SGD Classifier']
ax.bar(x_labels, accuracy_scores)
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Different Base Classifiers in AdaBoost')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
