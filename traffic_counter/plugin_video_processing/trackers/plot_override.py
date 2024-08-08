import colorsys
import cv2
import hashlib
import numpy as np


# this function modifies boxmot.trackers.basetracker.BaseTracker.id_to_color method
def id_to_color(id: int, saturation: float = 0.75, value: float = 0.95) -> tuple:
        """
        Generates a consistent unique BGR color for a given ID using hashing.

        Parameters:
        - id (int): Unique identifier for which to generate a color.
        - saturation (float): Saturation value for the color in HSV space.
        - value (float): Value (brightness) for the color in HSV space.

        Returns:
        - tuple: A tuple representing the BGR color.
        """

        # Hash the ID to get a consistent unique value
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()
        
        # Convert the first few characters of the hash to an integer
        # and map it to a value between 0 and 1 for the hue
        hue = int(hash_digest[:8], 16) / 0xffffffff
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert RGB from 0-1 range to 0-255 range and format as hexadecimal
        rgb_255 = tuple(int(component * 255) for component in rgb)
        hex_color = '#%02x%02x%02x' % rgb_255
        # Strip the '#' character and convert the string to RGB integers
        rgb = tuple(int(hex_color.strip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert RGB to BGR for OpenCV
        bgr = rgb[::-1]
        
        return bgr

# this function modifies boxmot.trackers.basetracker.BaseTracker.plot_box_on_img method
def plot_box_on_img(img: np.ndarray, box: tuple, conf: float, cls: int, id: int) -> np.ndarray:
    """
    Draws a bounding box with ID, confidence, and class information on an image.

    Parameters:
    - img (np.ndarray): The image array to draw on.
    - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
    - conf (float): Confidence score of the detection.
    - cls (int): Class ID of the detection.
    - id (int): Unique identifier for the detection.

    Returns:
    - np.ndarray: The image array with the bounding box drawn on it.
    """

    thickness = 2
    fontscale = 0.5

    img = cv2.rectangle(
        img,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        id_to_color(id),
        thickness
    )
    img = cv2.putText(
        img,
        f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
        (int(box[0]), int(box[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontscale,
        id_to_color(id),
        thickness
    )
    return img

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
        img = cv2.line(img, p1, p2, color=id_to_color(int(id)), thickness=2)
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

    active_tracks = tracker.active_tracks if hasattr(tracker, "active_tracks") else tracker.tracked_stracks
    for a in active_tracks:
        if a.history_observations:
            if len(a.history_observations) > 2 and (not hasattr(a, "frozen") or a.frozen == False):
                box = a.history_observations[-1]
                img = plot_box_on_img(img, box, a.conf if hasattr(a, "conf") else a.score, a.cls, a.id if hasattr(a, "id") else a.track_id)
                if show_trajectories:
                    img = plot_trackers_trajectories(
                        tracker, img, a.history_observations, a.id if hasattr(a, "id") else a.track_id
                    )

    return img
