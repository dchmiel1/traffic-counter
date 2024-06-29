import numpy as np

from traffic_counter.abstract_detector_adapter import DetectorAdapter
from traffic_counter.detectors.yolov6.yolov6.core.inferer import (
    Inferer as YOLOv6Inferer,
)


class YOLOv6Adapter(DetectorAdapter):
    weights_dir = "weights/yolov6/"

    def __init__(
        self,
        weights="yolov6l6.pt",
        device=0,
        conf_thres=0.4,
        iou_thres=0.45,
        agnostic_nms=True,
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.detector = YOLOv6Inferer(
            None,
            weights=self.weights_dir + weights,
            device=device,
            yaml="traffic_counter/detectors/yolov6/data/coco.yaml",
            img_size=[640, 640],
            half=True,
        )

    def _convert_dets(self, dets):
        return np.array(dets.cpu())

    def detect(self, img):
        dets = self.detector.infer_on_image(
            img, self.conf_thres, self.iou_thres, None, self.agnostic_nms, 1000
        )
        return self._convert_dets(dets)
