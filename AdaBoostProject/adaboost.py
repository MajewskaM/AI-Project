from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from rearrange_files import rearrange_files
import numpy as np
import os


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

# define the file names
x24x24_filename = 'x24x24.txt'
y24x24_filename = 'y24x24.txt'
z24x24_filename = 'z24x24.txt'

# creating directory with rearranged files
rearranged_data_directory = os.path.join(working_directory, 'data')
# rearrange_files(data_source_directory, rearranged_data_directory, x24x24_filename)
# rearrange_files(data_source_directory, rearranged_data_directory, y24x24_filename)
# rearrange_files(data_source_directory, rearranged_data_directory, z24x24_filename)

# loading data: extracting features and labels of categories
# data_directory = os.path.join()
X, y = load_data(data_source_directory)
print(X.shape)
print(X.dtype)
print(y)


# splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# 2. we need to normalize our dataset? preprocessing sprobowac
# preprocessing: Standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

print("Shape of training data: ", X_train.shape)
print("Shape of training label: ", y_train.shape)
print("Shape of test data: ", X_test.shape)
print("Shape of test label: ", y_test.shape)

# base_classifiers = [LogisticRegression(), SVC(), DecisionTreeClassifier(), LinearRegression()]

# using the decision tree as the base weak learner.

# weak_learner = DecisionTreeClassifier(max_leaf_nodes=8)
# cl = LogisticRegression(multi_class='multinomial')
# multi_class='ovr' 0.5497
# bez arg 0.5321

# sv = SVC()
# after Nystroem transformation
# clf = LinearSVC(dual="auto")
# sprobowac do innych funkcji z fandom state = 1
# feature_map_nystroem = Nystroem(gamma=.2, random_state=42, n_components=576)
# data_transformed = feature_map_nystroem.fit_transform(X_train) #X
# clf.fit(data_transformed, y_train)
# LinearSVC(dual='auto')
# rint(clf.score(data_transformed, y_train))
# svL = LinearSVC()

sgd = SGDClassifier()
# NN - nearest neightbor classifier
# NC - nearest center classifier
# Bayesian classifier
# Fisher discrimiant analysis
# decidion trees
dt = DecisionTreeClassifier(max_depth=3)
# Adaptive Boosting - AdaBoost AB
# Support vector machines (SVM)
# Artficial neural networks (ANN)
# Deep learning (DL)

# o tym bedzie na nastepnym wykladzie

# dt = LogisticRegression()
# data_pipeline = Pipeline([("classifier", AdaBoostClassifier(estimator=dt, n_estimators=200, algorithm='SAMME'))])
# data_pipeline.fit(X_train, y_train)


# generate AdaBoost model with the SAMME algorithm - for multiclass
model = AdaBoostClassifier(estimator=sgd, n_estimators=300, algorithm='SAMME')

# fitting the model to the training data
#model.fit(X_train, y_train)

# # predict the class of test data
# y_pred = model.predict(X_test)

# # calculate the accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Model Evaluation: accuracy:", accuracy)
# # For this code: accuracy: 0.7763157894736842


from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Cross-validation setup
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42)

# Define functions for training classifiers and evaluation
def train_classifiers(estimator, X_train, y_train, cv, name):
    estimator.fit(X_train, y_train)
    cv_train_score = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"On average, {name} model has f1 score of {cv_train_score.mean():3f} +/- {cv_train_score.std():.3f} on the training set.")
    y_pred = estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Evaluation: accuracy:", accuracy)
    

def eval(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)

    print("#Classification report")
    print(classification_report(y_test, y_pred))

    print("#Confusion matrix")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    #ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

# SVC
from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.3, max_depth=3, n_estimators = 300, subsample=0.5)
#xgb_params
# xgb_tuned

# Create pipeline with AdaBoostClassifier
data_pipeline = Pipeline([("classifier", AdaBoostClassifier(estimator=xgb, algorithm='SAMME'))])

# for decision tree stumps
#GRID SEARCH

# estimators = Pipeline([('vectorizer', CountVectorizer()),
#                        ('transformer', TfidfTransformer()),
#                        ('classifier', AdaBoostClassifier(learning_rate=1))])
# Train classifiers and evaluate
train_classifiers(data_pipeline, X_train, y_train.ravel(), cv, "AdaBoostClassifier")
eval(data_pipeline, X_test, y_test)

#eval(sgd, X_test, y_test)





# REPORT
# powyciagac jakies cechy, jakei biblioteki sa potrzebne,
# biblioteki koncepcja oipis i jakie sa wyniki, ryskunki wykresy jakie prawdopo,
# wykres z danych innych
# do 95%
# normlaizacja
# opisy do raportu z decision tree
