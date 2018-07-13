import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog


def convert_color(image, cspace):
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif cspace == 'LAB':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif cspace == 'GRAY':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            raise Exception("Cannot convert to color space: " + cspace)
    else:
        converted = np.copy(image)

    return converted


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                                  transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualize=vis, feature_vector=feature_vec)
        return features


def extract_features_from_img(image, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channels=[0],
                              hog_feature_vec=True, spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    feature_image = convert_color(image, cspace)

    # Apply bin_spatial() to get spatial color features
    bin_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() to get color histogram features
    hist_features = color_hist(feature_image, hist_bins, hist_range)

    try:
        gray_image = convert_color(image, 'GRAY')
        hog_features = np.ravel(get_hog_features(gray_image,
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=hog_feature_vec))
    except Warning as e:
        print("ERROR!!!", e)
        return None

    return np.concatenate([bin_features, hist_features, hog_features])
    # return np.concatenate([hog_features, bin_features])


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channels=[0], hog_feature_vec=True, spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        extracted = extract_features_from_img(image, cspace, orient, pix_per_cell, cell_per_block, hog_channels,
                                              hog_feature_vec, spatial_size, hist_bins, hist_range)
        if extracted is not None:
            features.append(extracted)

    # Return list of feature vectors
    return features
