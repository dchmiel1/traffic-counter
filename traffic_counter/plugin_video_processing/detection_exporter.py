from copy import deepcopy
from pathlib import Path

import cv2

from traffic_counter.plugin_video_processing.tracks_exporter import CLASSES

OTDET_BASE = {
    "metadata": {
        "classes": CLASSES,
    },
    "data": {"detections": []},
}


class DetectionExporter:
    def __init__(self, detector_name, video_file: Path):
        self.detector_name = detector_name
        self.video_file = video_file
        self.detections = {}
        self._retrieve_metadata()

    def _retrieve_metadata(self):
        vid = cv2.VideoCapture(self.video_file)
        self.metadata = {}
        self.metadata["filename"] = self.video_file.name.split(".")[0]
        self.metadata["filetype"] = "." + self.video_file.name.split(".")[1]
        self.metadata["width"] = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.metadata["height"] = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.metadata["expected_duration"] = 0
        self.metadata["recorded_fps"] = vid.get(cv2.CAP_PROP_FPS)
        self.metadata["actual_fps"] = vid.get(cv2.CAP_PROP_FPS)
        self.metadata["number_of_frames"] = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.metadata["length"] = "0:00:00.000000"

    @property
    def otdet(self):
        otdet = deepcopy(OTDET_BASE)
        otdet["metadata"]["model"] = self.detector_name
        otdet["metadata"]["video"] = self.metadata
        for frame_id, dets in self.detections.items():
            record = {
                frame_id: [
                    {
                        "x": float(det_record[0]),
                        "y": float(det_record[1]),
                        "w": float(det_record[2]),
                        "h": float(det_record[3]),
                        "confidence": float(det_record[4]),
                        "class": int(det_record[5]),
                    }
                    for det_record in dets
                ]
            }
            otdet["data"]["detections"].append(record)
        return otdet

    def update(self, frame_id, dets):
        self.detections[frame_id] = dets
