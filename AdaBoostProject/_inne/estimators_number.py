# explore adaboost ensemble number of trees effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
import os
import numpy as np


# get a list of models to evaluate
def get_models():
    models = dict()
    # define number of trees to consider
    n_trees = [10, 50, 100, 300, 500, 700, 1000, 5000]
    for n in n_trees:
        models[str(n)] = AdaBoostClassifier(estimator=DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1), n_estimators=n, algorithm='SAMME')
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    #pred = model.predict(X_test)
	# define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    

    # # calculate the accuracy
    # scores = accuracy_score(y_test, y_pred)
    return scores

# define dataset
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	# evaluate the model
	#scores = evaluate_model(model, X, y)
    model.fit(X_train, y_train)

    # predict the class of test data
    y_pred = model.predict(X_test)

    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
	# store the results
    results.append(accuracy)
    names.append(name)
	# summarize the performance along the way
    print('>%s %.3f (%.3f)' % (name, mean(accuracy), std(accuracy)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()