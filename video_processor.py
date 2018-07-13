import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from collections import deque

from tracer_decorator import traced, enable_tracing
from classifier import Classifier
from data_preparation import DataPreparation, HogConfig
from find_cars import find_cars, add_heat, apply_threshold, draw_labeled_bboxes


class VideoProcessor:
    """
    Class to help with processing a video
    """

    def __init__(self, input_file, output_file, data_prep, classifier):
        self.input_file = input_file
        self.output_file = output_file
        self.data_prep = data_prep
        self.classifier = classifier
        self.heat_buffer = deque(maxlen=5)
        self.labels = None
        self.current_frame = 0


    @traced
    def process(self, sub_clip=None, frame_divisor=3):
        """
        Process the video clip
        :param frame_divisor: process one in every x frames (set to 0 or lower to disable
        :param sub_clip: optionally specify a sub clip (start, end)
        :return: None
        """
        self.current_frame = 0

        def handle_frame(img, t):
            img = self._process_image(img, self.current_frame, frame_divisor)
            self.current_frame += 1
            return img

        clip = VideoFileClip(self.input_file)
        if sub_clip:
            clip = clip.subclip(sub_clip[0], sub_clip[1])
        out_clip = clip.fl(lambda gf, t: handle_frame(gf(t), t))
        out_clip.write_videofile(self.output_file, audio=False)

    @traced
    def _extract_labels(self, img):
        # get attributes of our svc object
        classifier = self.classifier
        data_prep = self.data_prep
        svc = classifier.svc
        X_scaler = data_prep.scaler
        orient = data_prep.hog_config.orient
        pix_per_cell = data_prep.hog_config.pix_per_cell
        cell_per_block = data_prep.hog_config.cell_per_block
        colorspace = data_prep.hog_config.colorspace
        hog_channels = data_prep.hog_config.hog_channels
        spatial_size = data_prep.hog_config.spatial_size
        hist_bins = data_prep.hog_config.hist_bins

        search_grid = [
            (400, 496, 200, 1080, 1),
            (400, 560, 0, 1280, 1.5),
            (400, 600, 0, 1280, 2),
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

        # heat = self.last_heat if self.last_heat is not None else np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat, boxes)

        threshold = 1
        heatmap = np.copy(heat)

        if len(self.heat_buffer) > 0:
            # Add the past heat buffers with this
            # frame's heatbuffer to create more stable
            # bounding boxes.
            #
            # Also helps prevent false positives as
            # the threshold is increased for every
            # frame added from the past
            for past_heat in self.heat_buffer:
                heatmap += past_heat
                threshold += 1

        # Apply threshold to help remove false positives
        heatmap = apply_threshold(heatmap, threshold)

        # Store for later
        self.heat_buffer.appendleft(heat)

        # Find final boxes from heatmap using label function
        return label(heatmap)

    @traced
    def _process_image(self, img, frame, frame_divisor):
        if frame_divisor <= 0 or frame % frame_divisor is 0 or self.labels is None:
            # Minimize the amount of frames to process
            self.labels = self._extract_labels(img)

        # Draw boxes on the original image
        return draw_labeled_bboxes(img, self.labels)


def main():
    enable_tracing(False)
    input_file = "test_videos/project_video.mp4"
    output_file = 'output_videos/processed_project_video.mp4'
    processor = VideoProcessor(input_file=input_file, output_file=output_file,
                               classifier=Classifier.rbf(),
                               data_prep=DataPreparation.default())

    print("Processing video", input_file, output_file)
    # processor.process(sub_clip=(12, 15))
    # processor.process(sub_clip=(21, 26))
    # processor.process(sub_clip=(5, 10), frame_divisor=4)
    processor.process(frame_divisor=4)


if __name__ == '__main__':
    main()
