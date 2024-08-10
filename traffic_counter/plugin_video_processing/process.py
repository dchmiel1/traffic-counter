import os
from typing import Callable
from pathlib import Path

import cv2
import numpy as np
from boxmot import BoTSORT, DeepOCSORT
from boxmot.trackers.basetracker import BaseTracker

from traffic_counter.plugin_video_processing.detection_exporter import DetectionExporter
from traffic_counter.plugin_video_processing.detectors.co_detr.co_detr_adapter import (
    CODETRAdapter as CODETR,
)
from traffic_counter.plugin_video_processing.detectors.yolov6.yolov6_adapter import (
    YOLOv6Adapter as YOLOv6,
)
from traffic_counter.plugin_video_processing.detectors.abstract_detector_adapter import (
    DetectorAdapter,
)
from traffic_counter.plugin_video_processing.detectors.rt_detr.rt_detr_adapter import (
    RTDETRAdapter as RTDETR,
)
from traffic_counter.plugin_video_processing.trackers.deep_ocsort_plus.deep_ocsort_plus import (
    DeepOCSortPlus,
)
from traffic_counter.plugin_video_processing.trackers.smiletrack.mc_SMILEtrack import (
    SMILEtrack,
)
from traffic_counter.plugin_video_processing.trackers.plot_override import (
    plot_results,
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
from traffic_counter.plugin_parser.json_parser import parse_json, write_json

CO_DETR_NAME = "CO-DETR"
RT_DETR_NAME = "RT-DETR"
YOLOV6_NAME = "YOLOv6"

DEEP_OC_SORT_NAME = "DeepOCSORT"
BOT_SORT_NAME = "BoT-SORT"
SMILETRACK_NAME = "SmileTrack"
DEEP_OC_SORT_PLUS_NAME = "DeepOCSORT+"

detectors = {
    CO_DETR_NAME: CODETR,
    RT_DETR_NAME: RTDETR,
    YOLOV6_NAME: YOLOv6,
}
trackers = {
    DEEP_OC_SORT_NAME: DeepOCSORT,
    BOT_SORT_NAME: BoTSORT,
    SMILETRACK_NAME: SMILEtrack,
    DEEP_OC_SORT_PLUS_NAME: DeepOCSortPlus,
}
results_exporter = {
    DEEP_OC_SORT_NAME: DeepOCSORTTracksExporter,
    BOT_SORT_NAME: BoTSORTTracksExporter,
    SMILETRACK_NAME: SMILETrackTracksExporter,
    DEEP_OC_SORT_PLUS_NAME: DeepOCSORTTracksExporter,
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
    tracker: BaseTracker, video: Path, tracker_name: str
) -> TracksExporter:
    exporter_class = results_exporter.get(tracker_name)
    if exporter_class is None:
        raise Exception(f"Exporter for '{tracker_name}' not found")

    return exporter_class(tracker, video)


def initialize_video_writer(vid_reader, filename):
    fps = vid_reader.get(cv2.CAP_PROP_FPS)
    width = vid_reader.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(filename, fourcc, int(fps), (int(width), int(height)))


def process(
    video_path: Path,
    detector_name: str,
    tracker_name: str,
    progress_bar: VideoProcessingProgressBarWindow,
    data_handler: Callable,
    save_processed_video: bool,
    save_detections: bool = True,
):
    video_path = Path(video_path)

    detector = get_detector(detector_name)
    detection_exporter = DetectionExporter(detector_name, video_path)
    tracker = get_tracker(tracker_name)
    track_exporter = get_results_exporter(tracker, video_path, tracker_name)
    vid_reader = cv2.VideoCapture(video_path)
    frame_count = vid_reader.get(cv2.CAP_PROP_FRAME_COUNT)

    base_path = f"{str(video_path).rsplit('.', 1)[0]}_{detector_name}_{tracker_name}"
    detection_data_path = f"{str(video_path).rsplit('.', 1)[0]}_{detector_name}.otdet"
    detection_data = parse_json(detection_data_path) if os.path.isfile(detection_data_path) else None

    if save_processed_video:
        vid_writer = initialize_video_writer(vid_reader, base_path + ".mp4")

    frame_id = 0
    while True:
        frame_id += 1
        ret, im = vid_reader.read()
        if not ret:
            break

        if detection_data:
            frame_dets = detection_data["data"]["detections"][str(frame_id)]
            dets = np.array([[det["x"], det["y"], det["w"], det["h"], det["confidence"], det["class"]] for det in frame_dets])
        else:
            dets = detector.detect(im)
        detection_exporter.update(frame_id, dets)
        tracker.update(dets, im)
        track_exporter.update(frame_id)

        if save_processed_video:
            plot_results(tracker, im, show_trajectories=True)
            vid_writer.write(im)

        print(frame_id)
        progress_bar.update(frame_id / frame_count)

    vid_reader.release()
    if save_processed_video:
        vid_writer.release()
    if save_detections and not os.path.isfile(detection_data_path):
        write_json(detection_exporter.otdet, detection_data_path)
    data_handler(track_exporter.ottrk, base_path, save_processed_video)
