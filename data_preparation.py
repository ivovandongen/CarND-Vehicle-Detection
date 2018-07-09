import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from saved_object import SavedObject
from feature_extraction import extract_features
from tracer_decorator import traced


class HogConfig:

    def __init__(self, colorspace='YUV', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
        """
        Configuration for HOG
        :param colorspace: Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        :param orient:
        :param pix_per_cell:
        :param cell_per_block:
        :param hog_channel: Can be 0, 1, 2, or "ALL"
        """
        self.colorspace = colorspace
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel


class DataPreparation(SavedObject):
    """
    Encapsulates loading and preparation of the test data set and preparation
    """

    SAVE_FILE = "data_preparation.p"

    def __init__(self, hog_config):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.hog_config = hog_config
        self.scaler = None

    @traced
    def _get_features(self):
        import warnings
        warnings.simplefilter("error")

        # Divide up into cars and notcars
        # Read in car and non-car images
        cars = glob.glob('training_data/vehicles/*/*.png')
        notcars = glob.glob('training_data/non-vehicles/*/*.png')

        car_features = extract_features(cars, cspace=self.hog_config.colorspace,
                                        orient=self.hog_config.orient,
                                        pix_per_cell=self.hog_config.pix_per_cell,
                                        cell_per_block=self.hog_config.cell_per_block,
                                        hog_channel=self.hog_config.hog_channel)
        notcar_features = extract_features(notcars, cspace=self.hog_config.colorspace,
                                           orient=self.hog_config.orient,
                                           pix_per_cell=self.hog_config.pix_per_cell,
                                           cell_per_block=self.hog_config.cell_per_block,
                                           hog_channel=self.hog_config.hog_channel)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        return X, y

    @traced
    def _prepare_scaler(self):
        assert self.X_train is not None
        self.scaler = StandardScaler(copy=False).fit(self.X_train)  # partial_fit

    @traced
    def _split_test_data(self, X, y):
        # Split up data into randomized training and test sets
        rand_state = 42
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,
                                                                                random_state=rand_state)

    @traced
    def _scale_data(self):
        # Apply the scaler to X
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    @traced
    def _init(self):
        # Get features and split test set
        X, y = self._get_features()
        self._split_test_data(X, y)

        # Fit a per-column scaler
        self._prepare_scaler()
        self._scale_data()

        print('Using:', self.hog_config.orient, 'orientations', self.hog_config.pix_per_cell,
              'pixels per cell and', self.hog_config.cell_per_block, 'cells per block')
        print('Feature vector length:', len(self.X_train[0]))

    @traced
    def prepare_images(self, images):
        features = extract_features(images, cspace=self.hog_config.colorspace,
                                    orient=self.hog_config.orient,
                                    pix_per_cell=self.hog_config.pix_per_cell,
                                    cell_per_block=self.hog_config.cell_per_block,
                                    hog_channel=self.hog_config.hog_channel)
        features = np.array(features).astype(np.float64)
        features = self.scaler.transform(features)

        return features

    @staticmethod
    def _instance():
        print("Preparing test data")
        data = DataPreparation(HogConfig())
        data._init()
        return data

    @staticmethod
    def default():
        """
        Use this method to obtain a prepared instance
        :return: the prepared test set
        """
        return SavedObject._create(DataPreparation._instance, DataPreparation.SAVE_FILE)


def main():
    DataPreparation.default()


if __name__ == '__main__':
    main()
