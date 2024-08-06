from typing import Callable
from pathlib import Path

import cv2
import numpy as np
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
from traffic_counter.plugin_video_processing.trackers.deep_ocsort_plus.deep_ocsort_plus import (
    DeepOCSortPlus,
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


# this function is a copy of boxmot.trackers.basetracker.BaseTracker.plot_results method
def plot_results(tracker, img: np.ndarray, show_trajectories: bool) -> np.ndarray:
    """
    Visualizes the trajectories of all active tracks on the image. For each track,
    it draws the latest bounding box and the path of movement if the history of
    observations is longer than two. This helps in understanding the movement patterns
    of each tracked object.

    Parameters:
    - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.

    Returns:
    - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
    """

    # if values in dict
    if tracker.per_class_active_tracks:
        for k in tracker.per_class_active_tracks.keys():
            active_tracks = tracker.per_class_active_tracks[k]
            for a in active_tracks:
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = tracker.plot_box_on_img(img, box, a.conf, a.cls, a.id)
                        if show_trajectories:
                            img = tracker.plot_trackers_trajectories(img, a.history_observations, a.id)
    else:
        for a in tracker.active_tracks:
            if a.history_observations:
                if len(a.history_observations) > 2 and a.frozen == False:
                    box = a.history_observations[-1]
                    img = tracker.plot_box_on_img(img, box, a.conf, a.cls, a.id)
                    if show_trajectories:
                        img = tracker.plot_trackers_trajectories(img, a.history_observations, a.id)

    return img

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
            plot_results(tracker, im, show_trajectories=True)
            vid_writer.write(im)

        print(frame_id)
        progress_bar.update(frame_id / frame_count)

    vid_reader.release()
    if save_processed_video:
        vid_writer.release()
    data_handler(exporter.ottrk, str(video_path), processed_video_filename)
