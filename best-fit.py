import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
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
mean_photos_per_person = len(y) / 48
print("Mean number of photos of each person:", mean_photos_per_person)

# splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)

print("Shape of training data: ", X_train.shape)
print("Shape of training label: ", y_train.shape)
print("Shape of test data: ", X_test.shape)
print("Shape of test label: ", y_test.shape)

# define the model with default hyperparameters
model = AdaBoostClassifier()

# define the grid of values to search
grid = dict()
grid['n_estimators'] = [10, 50, 100, 500]
grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#dla calego
# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
logistic_regression = LogisticRegression(max_iter=1000, solver='saga', tol=0.1, C=1, penalty='l2')
linear_svc = LinearSVC(C=0.01)
decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=4)
random_forest = RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=500)
sgdc_classifier = SGDClassifier(alpha=0.000774263682681127, l1_ratio=0.14, loss='log_loss', penalty='elasticnet')
base_classifiers = {logistic_regression:'Logistic Regression', linear_svc:'LinearSVC', decision_tree:'Decision Tree', random_forest:'Random Forest', sgdc_classifier:'SGD Classifier'}

# Perform grid search for AdaBoostClassifier
param_grid_ada = {'n_estimators': [10, 50, 100, 500], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]}

# AdaBoost accuarcy first 10 people, random 10 people
# Plotting
fig, ax = plt.subplots()
accuracy_scores = []

for classifier, name in base_classifiers.items():
    print("Classifier:", name)
    ada_clf = AdaBoostClassifier(estimator=classifier, algorithm='SAMME')
    grid_search_ada = GridSearchCV(ada_clf, param_grid_ada, cv=5, scoring='accuracy', n_jobs=-1).fit(X_train, y_train)
    print("Best Ada Boost Parameters for " + name + ": " + str(grid_search_ada.best_params_))
    print("Score: " + str(grid_search_ada.best_score_))
    print()
    accuracy_scores.append(grid_search_ada.best_score_)



# Plotting the accuracy scores
x_labels = ['Logistic Regression', 'LinearSVC', 'Decision Tree', 'Random Forest', 'SGD Classifier']
ax.bar(x_labels, accuracy_scores)
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Different Base Classifiers in AdaBoost')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# incerfase number of photos
# probowac z innym learning_rate=0.1 ten najlepszy 
model = AdaBoostClassifier(estimator=, n_estimators=300, algorithm='SAMME')

# 10 people
# Classifier: Logistic Regression
# Best Ada Boost Parameters for Logistic Regression: {'learning_rate': 0.0001, 'n_estimators': 10}
#lr_ada = AdaBoostClassifier(estimator=logistic_regression, learning_rate=0.0001, algorithm='SAMME', n_estimators=10)
# Best Ada Boost Parameters for LinearSVC: {'learning_rate': 0.01, 'n_estimators': 500}

# Classifier: Decision Tree
# Best Ada Boost Parameters for Decision Tree: {'learning_rate': 1.0, 'n_estimators': 500}

# Classifier: Random Forest
# Best Ada Boost Parameters for Random Forest: {'learning_rate': 0.001, 'n_estimators': 500}

# Classifier: SGD Classifier
# Best Ada Boost Parameters for SGD Classifier: {'learning_rate': 1.0, 'n_estimators': 500}
    

# Create a pipeline with StandardScaler and AdaBoostClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', AdaBoostClassifier(base_estimator=dt_base_estimator))
])

#data_pipeline = Pipeline([("classifier", AdaBoostClassifier(estimator=dt, n_estimators=50, algorithm='SAMME'))])
cross_validation_split = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
evaluateResults(model, X_test, y_test, X_train, y_train, cross_validation_split, "AdaBoostClassifier")
