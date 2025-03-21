import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class CustomNB:
    def __init__(self, **kwargs):
        self.nb = GaussianNB(**kwargs)
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # Ensure inputs are NumPy arrays
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.nb.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        # Ensure input is a NumPy array
        X_test = np.array(X_test)
        return self.nb.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Regular evaluation function that returns precision, recall, F1-score, accuracy, and confusion matrix.
        """
        y_pred = self.predict(X_test)
        metrics = {
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        return metrics

    def incremental_evaluate(self, X_test, y_test):
        """
        Incremental evaluation function that adds wrongly predicted instances to the training set
        and evaluates the model after all test instances have been predicted.
        """
        # Ensure inputs are NumPy arrays
        X_train_incremental = self.X_train.copy()
        y_train_incremental = self.y_train.copy()
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        for i in range(len(X_test)):
            x_instance = X_test[i].reshape(1, -1)
            y_instance = y_test[i]
            y_pred = self.nb.predict(x_instance)[0]

            if y_pred != y_instance:
                # Add the wrongly predicted instance to the training set
                X_train_incremental = np.vstack((X_train_incremental, x_instance))
                y_train_incremental = np.append(y_train_incremental, y_instance)
                # Refit the model with the updated training set
                self.nb.fit(X_train_incremental, y_train_incremental)

        # Evaluate using the regular evaluation function
        return self.evaluate(X_test, y_test)


class CustomKNN:
    def __init__(self, **kwargs):
        self.knn = KNeighborsClassifier(**kwargs)
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # Ensure inputs are NumPy arrays
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.knn.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        # Ensure input is a NumPy array
        X_test = np.array(X_test)
        return self.knn.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Regular evaluation function that returns precision, recall, F1-score, accuracy, and confusion matrix.
        """
        y_pred = self.predict(X_test)
        metrics = {
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        return metrics

    def incremental_evaluate(self, X_test, y_test):
        """
        Incremental evaluation function that adds wrongly predicted instances to the training set
        and evaluates the model after all test instances have been predicted.
        """
        # Ensure inputs are NumPy arrays
        X_train_incremental = self.X_train.copy()
        y_train_incremental = self.y_train.copy()
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        for i in range(len(X_test)):
            x_instance = X_test[i].reshape(1, -1)
            y_instance = y_test[i]
            y_pred = self.knn.predict(x_instance)[0]

            if y_pred != y_instance:
                # Add the wrongly predicted instance to the training set
                X_train_incremental = np.vstack((X_train_incremental, x_instance))
                y_train_incremental = np.append(y_train_incremental, y_instance)
                # Refit the model with the updated training set
                self.knn.fit(X_train_incremental, y_train_incremental)

        # Evaluate using the regular evaluation function
        return self.evaluate(X_test, y_test)