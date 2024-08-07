import bz2
import datetime
import json
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import cv2

CLASSES = {
    # "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    # "4": "airplane",
    "5": "bus",
    # "6": "train",
    "7": "truck",
    # "8": "boat",
    # "9": "traffic light",
    # "10": "fire hydrant",
    # "11": "stop sign",
    # "12": "parking meter",
    # "13": "bench",
    # "14": "bird",
    # "15": "cat",
    # "16": "dog",
    # "17": "horse",
    # "18": "sheep",
    # "19": "cow",
    # "20": "elephant",
    # "21": "bear",
    # "22": "zebra",
    # "23": "giraffe",
    # "24": "backpack",
    # "25": "umbrella",
    # "26": "handbag",
    # "27": "tie",
    # "28": "suitcase",
    # "29": "frisbee",
    # "30": "skis",
    # "31": "snowboard",
    # "32": "sports ball",
    # "33": "kite",
    # "34": "baseball bat",
    # "35": "baseball glove",
    # "36": "skateboard",
    # "37": "surfboard",
    # "38": "tennis racket",
    # "39": "bottle",
    # "40": "wine glass",
    # "41": "cup",
    # "42": "fork",
    # "43": "knife",
    # "44": "spoon",
    # "45": "bowl",
    # "46": "banana",
    # "47": "apple",
    # "48": "sandwich",
    # "49": "orange",
    # "50": "broccoli",
    # "51": "carrot",
    # "52": "hot dog",
    # "53": "pizza",
    # "54": "donut",
    # "55": "cake",
    # "56": "chair",
    # "57": "couch",
    # "58": "potted plant",
    # "59": "bed",
    # "60": "dining table",
    # "61": "toilet",
    # "62": "tv",
    # "63": "laptop",
    # "64": "mouse",
    # "65": "remote",
    # "66": "keyboard",
    # "67": "cell phone",
    # "68": "microwave",
    # "69": "oven",
    # "70": "toaster",
    # "71": "sink",
    # "72": "refrigerator",
    # "73": "book",
    # "74": "clock",
    # "75": "vase",
    # "76": "scissors",
    # "77": "teddy bear",
    # "78": "hair drier",
    # "79": "toothbrush",
}

OTTRK_BASE = {
    "metadata": {
        "otdet_version": "1.3",
        "detection": {
            "otvision_version": "0.0",
            "model": {"classes": CLASSES},
        },
        "ottrk_version": "1.1",
        "tracking": {
            "otvision_version": "0.0",
            "tracking_run_id": "e2062db8-229c-4e1b-8101-99d5818afe70",
            "frame_group": 0,
        },
    },
    "data": {"detections": []},
}

# TODO: refactor
START = datetime.datetime.now()
timestamp = time.mktime(START.timetuple())


@dataclass
class TrackRecord:
    cls: str
    conf: float
    bbox: list  # x, y, w ,h
    frame_id: int
    track_id: int


class TracksExporter(ABC):
    def __init__(self, tracker, video_file: Path):
        self.tracker = tracker
        self.video_file = video_file
        self.tracking_records = []
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
        self.metadata["recorded_start_date"] = timestamp
        self.metadata["length"] = "0:00:00.000000"

    def _is_track_valid(self, track):
        if not track.history_observations:
            return False

        if str(int(track.cls)) not in CLASSES:
            return False

        if any(x < 0 for x in track.history_observations[-1]):
            return False

        return True

    @abstractmethod
    def update():
        pass

    @property
    def ottrk(self):
        ottrk = deepcopy(OTTRK_BASE)
        ottrk["metadata"]["detection"]["model"]["classes"] = CLASSES
        ottrk["metadata"]["video"] = self.metadata

        for r in self.tracking_records:
            ottrk["data"]["detections"].append(
                {
                    "class": CLASSES[str(int(r.cls))],
                    "confidence": r.conf,
                    "x": r.bbox[0],
                    "y": r.bbox[1],
                    "w": r.bbox[2] - r.bbox[0],
                    "h": r.bbox[3] - r.bbox[1],
                    "frame": r.frame_id,
                    "occurrence": timestamp
                    + (r.frame_id * 1 / ottrk["metadata"]["video"]["actual_fps"]),
                    "track-id": r.track_id,
                }
            )
        return ottrk

    def save_ottrk(self):
        ottrk = self.ottrk

        # TODO
        filename = self.video_file.rsplit(".")[0] + ".ottrk"
        with open("_" + filename, "w") as f:
            f.write(json.dumps(ottrk, indent=4))

        with bz2.open(filename, "wt", encoding="UTF-8") as f:
            f.write(json.dumps(ottrk))

    def save_mp4():
        pass


class DeepOCSORTTracksExporter(TracksExporter):
    def update(self, frame_id):
        for tr in self.tracker.active_tracks:
            if not self._is_track_valid(tr):
                continue

            tracking_record = TrackRecord(
                tr.cls, tr.conf, deepcopy(tr.history_observations[-1]), frame_id, tr.id
            )
            self.tracking_records.append(tracking_record)


class BoTSORTTracksExporter(TracksExporter):
    def update(self, frame_id):
        for tr in self.tracker.active_tracks:
            if not self._is_track_valid(tr):
                continue

            tracking_record = TrackRecord(
                tr.cls, tr.conf, deepcopy(tr.history_observations[-1]), frame_id, tr.id
            )
            self.tracking_records.append(tracking_record)


class SMILETrackTracksExporter(TracksExporter):
    def update(self, frame_id):
        for tr in self.tracker.tracked_stracks:
            if not self._is_track_valid(tr):
                continue

            bbox = [float(obs) for obs in tr.history_observations[-1]]
            tracking_record = TrackRecord(
                tr.cls, float(tr.score), deepcopy(bbox), frame_id, tr.track_id
            )
            self.tracking_records.append(tracking_record)
