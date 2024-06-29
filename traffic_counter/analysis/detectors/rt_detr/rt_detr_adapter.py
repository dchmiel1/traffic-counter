import numpy as np

from ultralytics import RTDETR

from traffic_counter.abstract_detector_adapter import DetectorAdapter


class RTDETRAdapter(DetectorAdapter):
    weights_dir = "weights/rtdetr/"

    def __init__(self, weights="rtdetr-x.pt"):
        self.detector = RTDETR(model=self.weights_dir + weights)

    def _convert_dets(self, dets):
        return np.array(dets[0].boxes.data.cpu())

    def detect(self, img):
        dets = self.detector(img)
        return self._convert_dets(dets)
