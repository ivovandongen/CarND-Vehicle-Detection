import time
import multiprocessing

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from saved_object import SavedObject
from data_preparation import DataPreparation, HogConfig
from tracer_decorator import traced


class Classifier(SavedObject):
    """
    Encapsulates the classifier and training thereof
    """

    SAVE_FILE = "classifier.p"

    def __init__(self):
        self.svc = None

    @traced
    def _fit(self, X_train, y_train):
        # Setup the parameter space to explore
        param_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            # {'C': [10, 1000], 'gamma': [0.0001], 'kernel': ['rbf']}
        ]

        # Create the SVM
        svc = SVC()

        # Use multiple cpus to speed things up, but be conservative as this crashes a lot
        cpus = multiprocessing.cpu_count() // 2
        cpus = max(1, cpus)
        print("Using", cpus, "CPUs")

        # Use grid search to explore the parameter space
        svc = GridSearchCV(estimator=svc, param_grid=param_grid, n_jobs=cpus, verbose=10)
        svc.fit(X_train, y_train)

        self.svc = svc

    @traced
    def report(self, X_test, y_test):
        return {
            'score': round(self.svc.score(X_test, y_test), 4),
            'params': self.svc.get_params()
        }

    @traced
    def predict(self, X):
        return self.svc.predict(X)

    @staticmethod
    def _instance():
        test_data = DataPreparation.default()
        print("Preparing classifier")
        classifier = Classifier()
        classifier._fit(test_data.X_train, test_data.y_train)
        return classifier

    @staticmethod
    def default():
        """
        Use this method to obtain an instance
        :return: the trained classifier
        """

        return SavedObject._create(Classifier._instance, Classifier.SAVE_FILE)


def main():
    classifier = Classifier.default()
    test_data = DataPreparation.default()
    report = classifier.report(test_data.X_test, test_data.y_test)
    print("Classifier score:", report['score'], "params:", report["params"])


if __name__ == '__main__':
    main()
