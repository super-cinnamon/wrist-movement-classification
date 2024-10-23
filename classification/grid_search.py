import itertools
import os
import json
import logging

import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from models.mlp.MLP import MultiLayerPerceptron
from models.mlp.MLP import grid_search as mlp_gs

logging.basicConfig(level=logging.INFO)

MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', "data")

dataset_filename = "processed_dataset.csv"
dataset_path = os.path.join(DATA_PATH, dataset_filename)
dataset = pd.read_csv(dataset_path)


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop("class", axis=1),
    dataset["class"],
    test_size=0.2,
    random_state=1234
)

knn_param_grid = {
    'n_neighbors': [1, 3, 5, 7],
    'metric': ['minkowski', 'manhattan', 'cosine', 'haversine'],
    'weights': ['uniform', 'distance']
}

logistic_regression_param_grid = {
    'solver': ['lbfgs', 'liblinear', 'newton-cholesky'],
    'max_iter': [50, 100, 150, 200, 250, 300, 500],
    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    'C': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],  # noqa
    'class_weight': [None, 'balanced']
}

bayes_param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
}

svm_param_grid = {
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'C': [0.01, 0.1, 1, 10],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
    'coef0': [-1, 0, 1, 2, 3],
    'shrinking': [False],
    'class_weight': [None, 'balanced']
}

decision_tree_param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 2, 3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7],
    'min_weight_fraction_leaf': [0.01, 0.02, 0.03, 0.04, 0.05],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

random_forest_param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 3, 5, 6, 7, 8],
    'min_samples_split': [2, 4, 5, 7, 8],
    'min_samples_leaf': [1, 2, 5, 7],
    'min_weight_fraction_leaf': [0.01, 0.03, 0.05],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced'],
    'n_estimators': [20, 50, 100, 150, 200],
    'bootstrap': [False, True],
    'warm_start': [False, True]
}

xgboost_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150, 200, 300],
    'min_child_weight': [1, 2, 3, 4],
    'gamma': [0, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 2, 3, 4],
    'reg_alpha': [0, 0.1, 0.2, 0.3],
    'reg_lambda': [0, 0.1, 0.2, 0.3]
}

mlp_param_grid = {
    "hidden_size": [4, 8, 16, 32, 64],
    "activation": ['relu', 'leaky_relu', 'tanh', 'swish', 'gelu', 'selu'],
    "optimizer": ['adam', 'rmsprop', 'sgd'],
    "batch_size": [4, 8, 16, 32],
    "regularization": [0.0001, 0.001, 0.01],
    "epochs": [5, 10, 20, 30, 50, 80, 100],
    "lr": [0.001, 0.01, 0.1],
    "loss": ['mse', 'hinge', 'log']
}


# loop through all and create gridsearch
grids = ["knn", "logistic_regression", "naive_bayes", "svm", "decision_tree", "random_forest", "xgboost", "mlp"]   # noqa
param_grids = [knn_param_grid, logistic_regression_param_grid,
               bayes_param_grid, svm_param_grid, decision_tree_param_grid,
               random_forest_param_grid, xgboost_param_grid, mlp_param_grid]


def get_best_params():
    for i in range(len(grids)):
        logging.info(f"Running {grids[i]}")
        model_path = grids[i]
        if model_path == "knn":
            model = KNeighborsClassifier()
        elif model_path == "logistic_regression":
            model = LogisticRegression()
        elif model_path == "naive_bayes":
            model = GaussianNB()
        elif model_path == "svm":
            model = SVC()
        elif model_path == "decision_tree":
            model = DecisionTreeClassifier()
        elif model_path == "random_forest":
            model = RandomForestClassifier()
        elif model_path == "xgboost":
            model = XGBClassifier()
        elif model_path == "mlp":
            model = MultiLayerPerceptron(input_size=X_train.shape[1])

        if model_path == "mlp":
            hyperparameter_combinations = list(itertools.product(*mlp_param_grid.values()))  # noqa
            best_params = mlp_gs(
                X_train, y_train, X_test, y_test,
                hyperparameter_combinations,
                X_train.shape[1],
                os.path.join(MODELS_PATH, model_path)
            )

        else:
            grid_search = GridSearchCV(model, param_grids[i], cv=5, scoring='f1')   # noqa
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_

        # save best model and hyperparametrs into files
        if not os.path.exists(os.path.join(MODELS_PATH, model_path)):
            os.makedirs(os.path.join(MODELS_PATH, model_path))
        with open(os.path.join(MODELS_PATH, model_path, 'best_params.json'), 'w') as f:  # noqa
            json.dump(best_params, f, indent=4)
