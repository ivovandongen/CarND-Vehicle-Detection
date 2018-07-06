import time
import multiprocessing

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from saved_object import SavedObject
from test_data import TestData


class Classifier(SavedObject):
    """
    Encapsulates the classifier and training thereof
    """

    SAVE_FILE = "classifier.p"

    def __init__(self):
        self.svc = None

    def _fit(self, X_train, y_train):
        # Setup the parameter space to explore
        param_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]

        # Create the SVM
        svc = SVC()

        # Check the training time for the SVC
        t = time.time()

        # Use multiple cpus to speed things up, but be conservative as this crashes a lot
        cpus = 2 if multiprocessing.cpu_count() > 4 else 1
        print("Using", cpus, "CPUs")

        # Use grid search to explore the parameter space
        svc = GridSearchCV(estimator=svc, param_grid=param_grid, n_jobs=cpus, verbose=10)
        svc.fit(X_train, y_train)

        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        self.svc = svc

    def report(self, X_test, y_test):
        return round(self.svc.score(X_test, y_test), 4)

    @staticmethod
    def _instance():
        test_data = TestData.default()
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

        obj = SavedObject._create(Classifier._instance, Classifier.SAVE_FILE)
        test_data = TestData.default()
        print("Classifier score", obj.report(test_data.X_test, test_data.y_test))


if __name__ == '__main__':
    Classifier.default()