from glob import glob
import numpy as np

from data_preparation import DataPreparation, HogConfig
from classifier import Classifier


def main():
    data_prep = DataPreparation.default()
    classifier = Classifier.default()
    test_files = glob('training_data/vehicles/*/*.png')
    np.random.shuffle(np.array(test_files))
    prepared = data_prep.prepare_images(test_files[0:1000])
    results = classifier.predict(prepared)
    print("Results", results)
    print("Error", (len(results[results == 0]) / len(results)), "%")


if __name__ == '__main__':
    main()