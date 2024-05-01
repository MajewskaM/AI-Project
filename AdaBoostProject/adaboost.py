import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


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

    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Evaluation - accuracy:", accuracy)

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

base_classifiers = [LogisticRegression(), SVC(), DecisionTreeClassifier(), LinearRegression(), SGDClassifier()]

# using the decision tree as the base weak learner
# dt = DecisionTreeClassifier(max_depth=2)
sgdc = SGDClassifier()


model = AdaBoostClassifier(n_estimators=50, algorithm='SAMME')
#data_pipeline = Pipeline([("classifier", AdaBoostClassifier(estimator=dt, n_estimators=50, algorithm='SAMME'))])
cross_validation_split = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
evaluateResults(model, X_test, y_test, X_train, y_train, cross_validation_split, "AdaBoostClassifier")
