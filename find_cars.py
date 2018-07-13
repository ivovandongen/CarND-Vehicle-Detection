import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from glob import glob
from scipy.ndimage.measurements import label

from classifier import Classifier
from data_preparation import DataPreparation, HogConfig
from feature_extraction import get_hog_features, bin_spatial, color_hist, convert_color
from tracer_decorator import traced


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


# Define a single function that can extract features using hog sub-sampling and make predictions
@traced
def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, colorspace,
              hog_channels, spatial_size, hist_bins):
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, xstart:xstop, :]
    ctrans_tosearch = convert_color(img_tosearch, colorspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 8 ** 2
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    img_hog_features = []
    for channel in hog_channels:
        img_hog_features.append(
            get_hog_features(ctrans_tosearch[:, :, channel], orient, pix_per_cell, cell_per_block, feature_vec=False))

    boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG for this patch
            patch_hog_features = []
            for img_hog_features_channel in img_hog_features:
                patch_hog_features.append(
                    img_hog_features_channel[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel())
            hog_features = np.hstack(patch_hog_features)

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            test_prediction = svc.predict(test_features)
            # if test_prediction == 1 or test_prediction is not 1:
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                boxes.append(((xbox_left + xstart, ytop_draw + ystart), (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))

    return boxes


if __name__ == '__main__':

    # get attributes of our svc object
    classifier = Classifier.linear()
    data_prep = DataPreparation.default()
    svc = classifier.svc
    X_scaler = data_prep.scaler
    orient = data_prep.hog_config.orient
    pix_per_cell = data_prep.hog_config.pix_per_cell
    cell_per_block = data_prep.hog_config.cell_per_block
    colorspace = data_prep.hog_config.colorspace
    hog_channels = data_prep.hog_config.hog_channels
    spatial_size = data_prep.hog_config.spatial_size
    hist_bins = data_prep.hog_config.hist_bins

    for file in glob('test_images/vlc*.png'):
        img = mpimg.imread(file)

        search_grid = [
            (400, 496, 200, 1080, 1),
            (400, 560, 100, 1180, 1.5),
            (400, 600, 100, 1180, 2),
            (400, 688, 0, 1280, 3),
            (336, 720, 0, 1280, 4),
        ]

        boxes = []

        for y_start, y_stop, x_start, x_stop, scale in search_grid:
            boxes.extend(find_cars(img, y_start, y_stop, x_start, x_stop, scale,
                          svc, X_scaler,
                          orient, pix_per_cell, cell_per_block,
                          colorspace, hog_channels,
                          spatial_size, hist_bins))

        # Draw raw boxes
        raw_box_img = np.copy(img)
        for box in boxes:
            cv2.rectangle(raw_box_img, box[0], box[1], (0, 0, 255), 6)

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        # Add heat to each box in box list

        heat = add_heat(heat, boxes)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        # plt.imshow(draw_img)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 9))
        ax1.imshow(draw_img)
        ax1.set_title('Car Positions')

        ax2.imshow(raw_box_img)
        ax2.set_title('Raw boxes')

        ax3.imshow(heatmap, cmap="hot")
        ax3.set_title('Heat Map')

        plt.savefig('output_images/cars_' + file.split('/')[-1])
