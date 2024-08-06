import cv2
import numpy as np


# this function modifies boxmot.trackers.basetracker.BaseTracker.plot_trackers_trajectories method
def plot_trackers_trajectories(
    tracker, img: np.ndarray, observations: list, id: int
) -> np.ndarray:
    """
    Draws the trajectories of tracked objects based on historical observations. Each point
    in the trajectory is represented by a circle, with the thickness increasing for more
    recent observations to visualize the path of movement.

    Parameters:
    - img (np.ndarray): The image array on which to draw the trajectories.
    - observations (list): A list of bounding box coordinates representing the historical
    observations of a tracked object. Each observation is in the format (x1, y1, x2, y2).
    - id (int): The unique identifier of the tracked object for color consistency in visualization.

    Returns:
    - np.ndarray: The image array with the trajectories drawn on it.
    """
    for i, box in enumerate(observations):
        if i == 0:
            continue
        p1 = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        box2 = observations[i - 1]
        p2 = (int((box2[0] + box2[2]) / 2), int((box2[1] + box2[3]) / 2))
        img = cv2.line(img, p1, p2, color=tracker.id_to_color(int(id)), thickness=2)
    return img


# this function overrides and modifies boxmot.trackers.basetracker.BaseTracker.plot_results method
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
                            img = plot_trackers_trajectories(
                                tracker, img, a.history_observations, a.id
                            )
    else:
        for a in tracker.active_tracks:
            if a.history_observations:
                if len(a.history_observations) > 2 and a.frozen == False:
                    box = a.history_observations[-1]
                    img = tracker.plot_box_on_img(img, box, a.conf, a.cls, a.id)
                    if show_trajectories:
                        img = plot_trackers_trajectories(
                            tracker, img, a.history_observations, a.id
                        )

    return img
