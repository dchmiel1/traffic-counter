import numpy as np

from traffic_counter.plugin_video_processing import WEIGHTS_DIR
from traffic_counter.plugin_video_processing.detectors.abstract_detector_adapter import (
    DetectorAdapter,
)
from traffic_counter.plugin_video_processing.detectors.co_detr.mmdet.apis import (
    inference_detector,
    init_detector as init_codetr_detector,
)
from traffic_counter.plugin_video_processing.tracks_exporter import CLASSES


class CODETRAdapter(DetectorAdapter):
    config_dir = "traffic_counter/plugin_video_processing/detectors/co_detr/projects/configs/co_dino/"

    def __init__(
        self,
        weights="co_dino_5scale_swin_large_3x_coco.pth",
        config="co_dino_5scale_swin_large_3x_coco.py",
        device="cuda:0",
    ):
        self.detector = init_codetr_detector(
            config=self.config_dir + config,
            checkpoint=WEIGHTS_DIR + weights,
            device=device,
        )

    def _convert_dets(self, dets):
        converted_dets = []
        for class_id, class_dets in enumerate(dets):
            if str(class_id) not in CLASSES:
                continue
            for det in class_dets:
                converted_det = np.append([det[0:5]], [class_id])
                converted_dets.append(converted_det)
        return np.array(converted_dets)

    def detect(self, img):
        dets = inference_detector(self.detector, img)
        return self._convert_dets(dets)
