import numpy as np

from ultralytics import RTDETR

from traffic_counter.plugin_video_processing import WEIGHTS_DIR
from traffic_counter.plugin_video_processing.detectors.abstract_detector_adapter import (
    DetectorAdapter,
)
from traffic_counter.plugin_video_processing.tracks_exporter import CLASSES


class RTDETRAdapter(DetectorAdapter):
    def __init__(self, weights="rtdetr-x.pt"):
        self.detector = RTDETR(model=WEIGHTS_DIR + weights)

    def _convert_dets(self, dets):
        return np.array(dets[0].boxes.data.cpu())

    def detect(self, img):
        dets = self.detector(img, classes=[int(key) for key in CLASSES.keys()])
        return self._convert_dets(dets)
