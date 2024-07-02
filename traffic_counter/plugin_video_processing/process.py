from typing import Callable
from pathlib import Path

import cv2
from boxmot import BoTSORT, DeepOCSORT
from boxmot.trackers.basetracker import BaseTracker

from traffic_counter.plugin_video_processing.detectors.co_detr.co_detr_adapter import (
    CODETRAdapter as CODETR,
)
from traffic_counter.plugin_video_processing.detectors.yolov6.yolov6_adapter import (
    YOLOv6Adapter as YOLOv6,
)
from traffic_counter.plugin_video_processing.detectors.abstract_detector_adapter import (
    DetectorAdapter,
)
from traffic_counter.plugin_video_processing.detectors.rt_detr import (
    rt_detr_adapter as RTDETR,
)
from traffic_counter.plugin_video_processing.trackers.smiletrack.smiletrack import (
    SMILEtrack,
)
from traffic_counter.plugin_video_processing.tracks_exporter import (
    BoTSORTTracksExporter,
    DeepOCSORTTracksExporter,
    SMILETrackTracksExporter,
    TracksExporter,
)
from traffic_counter.plugin_ui.customtkinter_gui.video_processing_progress_bar_window import (
    VideoProcessingProgressBarWindow,
)

CO_DETR_NAME = "CO-DETR"
RT_DETR_NAME = "RT-DETR"
YOLOV6_NAME = "YOLOv6"

DEEP_OC_SORT_NAME = "DeepOCSORT"
BOT_SORT_NAME = "BoT-SORT"
SMILETRACK_NAME = "SmileTrack"

detectors = {CO_DETR_NAME: CODETR, RT_DETR_NAME: RTDETR, YOLOV6_NAME: YOLOv6}
trackers = {
    DEEP_OC_SORT_NAME: DeepOCSORT,
    BOT_SORT_NAME: BoTSORT,
    SMILETRACK_NAME: SMILEtrack,
}
results_exporter = {
    DEEP_OC_SORT_NAME: DeepOCSORTTracksExporter,
    BOT_SORT_NAME: BoTSORTTracksExporter,
    SMILETRACK_NAME: SMILETrackTracksExporter,
}


def get_detector(detector_name: str) -> DetectorAdapter:
    det_class = detectors.get(detector_name)
    if det_class is None:
        raise Exception(f"Invalid detector '{detector_name}'")

    return det_class()


def get_tracker(tracker_name: str) -> BaseTracker:
    tracker_class = trackers.get(tracker_name)
    if tracker_class is None:
        raise Exception(f"Invalid tracker '{tracker_name}'")

    return tracker_class(
        model_weights=Path("weights/trackers/osnet_x0_25_msmt17.pt"),
        device="cuda:0",
        fp16=False,
    )


def get_results_exporter(
    tracker: BaseTracker, video: str, tracker_name: str
) -> TracksExporter:
    exporter_class = results_exporter.get(tracker_name)
    if exporter_class is None:
        raise Exception(f"Exporter for '{tracker_name}' not found")

    return exporter_class(tracker, video)


def process(
    video_path: Path,
    detector_name: str,
    tracker_name: str,
    progress_bar: VideoProcessingProgressBarWindow,
    data_handler: Callable,
):
    detector = get_detector(detector_name)
    tracker = get_tracker(tracker_name)
    exporter = get_results_exporter(tracker, video_path, tracker_name)
    vid = cv2.VideoCapture(video_path)
    frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_id = 0
    while True:
        frame_id += 1
        ret, im = vid.read()
        if not ret:
            break

        dets = detector.detect(im)
        tracker.update(dets, im)
        exporter.update(frame_id)
        print(frame_id)
        progress_bar.update(frame_id / frame_count)

    vid.release()
    data_handler(video_path, exporter.ottrk)
