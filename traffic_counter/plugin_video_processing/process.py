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
from traffic_counter.plugin_video_processing.detectors.rt_detr.rt_detr_adapter import (
    RTDETRAdapter as RTDETR,
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
    video_path: str,
    detector_name: str,
    tracker_name: str,
    progress_bar: VideoProcessingProgressBarWindow,
    data_handler: Callable,
    save_processed_video: bool,
):
    video_path = Path(video_path)

    detector = get_detector(detector_name)
    tracker = get_tracker(tracker_name)
    exporter = get_results_exporter(tracker, video_path, tracker_name)
    vid_reader = cv2.VideoCapture(video_path)
    frame_count = vid_reader.get(cv2.CAP_PROP_FRAME_COUNT)

    if save_processed_video:
        processed_video_filename = "_processed.".join(str(video_path).rsplit(".", 1))
        vid_writer = initialize_video_writer(vid_reader, processed_video_filename)
    else:
        processed_video_filename = None

    frame_id = 0
    while True:
        frame_id += 1
        ret, im = vid_reader.read()
        if not ret:
            break

        dets = detector.detect(im)
        tracker.update(dets, im)
        exporter.update(frame_id)

        if save_processed_video:
            tracker.plot_results(im, show_trajectories=True)
            vid_writer.write(im)

        print(frame_id)
        progress_bar.update(frame_id / frame_count)

    vid_reader.release()
    if save_processed_video:
        vid_writer.release()
    data_handler(exporter.ottrk, str(video_path), processed_video_filename)
