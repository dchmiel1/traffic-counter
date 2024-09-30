# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import torch
from collections import deque

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc.sof import SOF
from boxmot.motion.kalman_filters.botsort_kf import KalmanFilter
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import (
    embedding_distance,
    fuse_score,
    iou_distance,
    linear_assignment,
)
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils import PerClassDecorator


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, det, feat=None, feat_history=50):
        # wait activate
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(self.cls, self.conf)
        self.history_observations = deque([], maxlen=50)

        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        self.last_size = deque([], maxlen=3)
        self.last_valid_x_params = deque([], maxlen=3)
        self.last_valid_y_params = deque([], maxlen=3)

        self.last_is_disappearing = deque([], maxlen=3)
        self.last_diffs = deque([], maxlen=3)

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, conf):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += conf
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, conf])
                self.cls = cls
        else:
            self.cls_hist.append([cls, conf])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    def modify_prediction(self):
        try:
            # obtain saved params
            valid_mean_width, valid_mean_vel_x = self.mean_params(
                self.last_valid_x_params
            )
            valid_mean_height, valid_mean_vel_y = self.mean_params(
                self.last_valid_y_params
            )
            mean_width, mean_height = self.mean_params(self.last_size)
        except ValueError:
            return None, None

        last_prediction = xywh2xxyy(self.mean[:4])

        # prepare data
        valid_vels = [valid_mean_vel_x, valid_mean_vel_y]
        valid_size = [valid_mean_width, valid_mean_height]
        last_size = [mean_width, mean_height]
        diffs = [sum(x) / len(self.last_diffs) for x in zip(*self.last_diffs)]
        skip = [sum(x) < 1 for x in zip(*self.last_is_disappearing)]

        # move prediction right/down
        def go_forward(coords, valid_size, last_size):
            coords[1] = coords[0] + valid_size
            coords[0] = coords[1] - last_size
            return coords

        # move prediction left/up
        def go_back(coords, valid_size, last_size):
            coords[0] = coords[1] - valid_size
            coords[1] = coords[0] + last_size
            return coords

        new_xxyy = []
        for i, coords in enumerate([(last_prediction[:2]), (last_prediction[2:4])]):
            if skip[i]:
                new_xxyy.extend(coords)
                continue

            if abs(valid_vels[i]) < 2:
                # 1st case - object not moving or moving very slowly
                modify_coords = go_forward if diffs[i] < 0 else go_back
            elif valid_vels[i] * diffs[i] > 0 and abs(diffs[i]) > abs(valid_vels[i]):
                # 2nd case - object covered by faster obstacle
                modify_coords = go_forward if valid_vels[i] < 0 else go_back
            else:
                # 3rd case - object went behind obstacle
                modify_coords = go_forward if valid_vels[i] > 0 else go_back

            new_xxyy.extend(modify_coords(coords, valid_size[i], last_size[i]))

        self.last_is_disappearing.clear()
        self.last_diffs.clear()
        return xxyy2xyxy(new_xxyy), (valid_mean_vel_x, valid_mean_vel_y)

    def modify_mean(self, mean):
        if any([sum(x) > 1 for x in zip(*self.last_is_disappearing)]):
            bbox, vels = self.modify_prediction()
            if bbox is not None and vels is not None:
                mean[0] = (bbox[0] + bbox[2]) / 2
                mean[1] = (bbox[1] + bbox[3]) / 2
                mean[2] = bbox[2] - bbox[0]
                mean[3] = bbox[3] - bbox[1]
                mean[4] = vels[0]
                mean[5] = vels[1]
        mean[6] = 0
        mean[7] = 0

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    st.modify_mean(multi_mean[i])
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

        self.update_cls(new_track.cls, new_track.conf)

    def mean_params(self, params):
        return (sum(p) / len(params) for p in zip(*params))

    def save_params(self):
        curr_bbox = self.xyxy
        width = self.mean[2]
        height = self.mean[3]

        last_bbox = self.history_observations[-1]
        x_diff = (curr_bbox[2] + curr_bbox[0]) / 2 - (last_bbox[2] + last_bbox[0]) / 2
        y_diff = (curr_bbox[3] + curr_bbox[1]) / 2 - (last_bbox[3] + last_bbox[1]) / 2

        self.last_diffs.append((x_diff, y_diff))
        self.last_size.append((width, height))

    def check_if_disappearing(self):
        try:
            mean_width, mean_height = self.mean_params(self.last_size)
        except (ValueError, KeyError):
            return

        width = self.mean[2]
        height = self.mean[3]

        # check if size decreases and save the result
        is_disappearing = (mean_width * 0.85 > width, mean_height * 0.85 > height)
        self.last_is_disappearing.append(is_disappearing)

        # save valid size and velocity if not disappearing
        if not is_disappearing[0] and mean_width <= width:
            self.last_valid_x_params.append((width, float(self.mean[4])))
        if not is_disappearing[1] and mean_height <= height:
            self.last_valid_y_params.append((height, float(self.mean[5])))

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )
        self.check_if_disappearing()
        self.save_params()

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        self.update_cls(new_track.cls, new_track.conf)

    @property
    def xyxy(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # kf (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret


class BoTSORTPlus(BaseTracker):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        per_class=False,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.85,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "sof",
        frame_rate=30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
    ):
        super(BoTSORTPlus, self).__init__()
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        self.with_reid = with_reid
        if self.with_reid:
            rab = ReidAutoBackend(weights=model_weights, device=device, half=fp16)
            self.model = rab.get_backend()

        self.cmc = SOF()
        self.fuse_first_associate = fuse_first_associate

    @PerClassDecorator
    def update(self, dets, img, embs=None):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])

        # Remove bad detections
        confs = dets[:, 4]

        # find second round association detections
        second_mask = np.logical_and(
            confs > self.track_low_thresh, confs < self.track_high_thresh
        )
        dets_second = dets[second_mask]

        # find first round association detections
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]

        """Extract embeddings """
        # appearance descriptor extraction
        if self.with_reid:
            if embs is not None:
                features_high = embs
            else:
                # (Ndets x X) [512, 1024, 2048]
                features_high = self.model.get_features(dets_first[:, 0:4], img)

        if len(dets) > 0:
            """Detections"""
            if self.with_reid:
                detections = [
                    STrack(det, f) for (det, f) in zip(dets_first, features_high)
                ]
            else:
                detections = [STrack(det) for (det) in np.array(dets_first)]
        else:
            detections = []

        """ Add newly detected tracklets to active_tracks"""
        unconfirmed = []
        active_tracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.cmc.apply(img, dets_first)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high conf detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low conf detection boxes"""
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [STrack(dets_second) for dets_second in dets_second]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh

        ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.conf < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_count)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_age:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        output_stracks = [track for track in self.active_tracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.id)
            output.append(t.conf)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)

        outputs = np.asarray(outputs)
        return outputs


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.id] = t
    for t in tlistb:
        tid = t.id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


def xywh2xxyy(xywh):
    """
    Convert bounding box coordinates from (x_c, y_c, width, height) format to
    (x1, x2, y1, y2) format where (x1, y1) is the top-left corner and (x2, y2)
    is the bottom-right corner.

    Args:
        xywh (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        xxyy (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, x1, y1, y2) format.
    """

    xxyy = xywh.clone() if isinstance(xywh, torch.Tensor) else np.copy(xywh)
    xxyy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2  # top left x
    xxyy[..., 1] = xywh[..., 0] + xywh[..., 2] / 2  # bottom right x
    xxyy[..., 2] = xywh[..., 1] - xywh[..., 3] / 2  # top left y
    xxyy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2  # bottom right y
    return xxyy


def xxyy2xyxy(xxyy):
    """
    Convert bounding box coordinates from (x1, x2, y1, y2) format to
    (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and (x2, y2)
    is the bottom-right corner.

    Args:
        xxyy (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        xyxy (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    return [xxyy[0], xxyy[2], xxyy[1], xxyy[3]]
