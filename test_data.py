import numpy as np
import glob
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from saved_object import SavedObject
from feature_extraction import extract_features


class TestData(SavedObject):
    """
    Encapsulates loading and preparation of the test data set
    """

    SAVE_FILE = "test_data.p"

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _init(self):
        import warnings
        warnings.simplefilter("error")

        # Divide up into cars and notcars
        # Read in car and non-car images
        cars = glob.glob('training_data/vehicles/*/*.png')
        notcars = glob.glob('training_data/non-vehicles/*/*.png')

        ### TODO: Tweak these parameters and see how the results change.
        colorspace = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

        t = time.time()
        car_features = extract_features(cars, cspace=colorspace, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                        hog_channel=hog_channel)
        notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                           hog_channel=hog_channel)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

        t = time.time()
        # Fit a per-column scaler
        X_scaler = StandardScaler(copy=False).fit(X_train)  # partial_fit
        # Apply the scaler to X
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to scale data...')

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    @staticmethod
    def _instance():
        print("Preparing test data")
        data = TestData()
        data._init()
        return data

    @staticmethod
    def default():
        """
        Use this method to obtain a prepared instance
        :return: the prepared test set
        """
        return SavedObject._create(TestData._instance, TestData.SAVE_FILE)
