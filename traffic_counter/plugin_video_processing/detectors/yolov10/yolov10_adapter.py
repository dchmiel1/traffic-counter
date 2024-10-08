import numpy as np

from ultralytics import YOLO

from traffic_counter.plugin_video_processing import WEIGHTS_DIR
from traffic_counter.plugin_video_processing.detectors.abstract_detector_adapter import (
    DetectorAdapter,
)


class YOLOv10Adapter(DetectorAdapter):
    def __init__(self, weights="yolov10x.pt"):
        self.detector = YOLO(model=WEIGHTS_DIR + weights)

    def _convert_dets(self, dets):
        return np.array(dets[0].boxes.data.cpu())

    def detect(self, img):
        dets = self.detector(img)
        return self._convert_dets(dets)
