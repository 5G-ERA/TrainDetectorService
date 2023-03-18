
import cv2
import numpy as np
from math import sqrt


class OpticalFlowKLT():
    """Obejct for storing optical flow information.

    This object stores detected keypoints and their respective tracks.
    Keypoints are obtained using cv2.goodFeaturesToTrack method and
    optical flow is calculated with cv2.calcOpticalFlowPyrLK.
    """

    def __init__(self, min_points, min_tracks_for_motion, min_track_len_for_motion, motion_percent_threshold, 
                motion_voting_ratio, max_track_len=50, detect_new_pt_interval=5, max_new_points=250):
        """Initialize optical flow object.

        Args:
            min_points (int): Minimum number of existing point tracks
                (otherwise adding new points is triggered).
            min_tracks_for_motion (int): Minimum number of point 
                tracks for motion estimation.
            min_track_len_for_motion (int): Minimum point track length 
                to be considered for motion estimation.
            motion_percent_threshold (float): Motion threshold given 
                as percentage of larger image side.
            motion_voting_ratio (float): Percent of moving points to
                consider the object as moving.
            max_track_len (int): Maximum point track length (older 
                points are removed from track.)
            detect_new_pt_interval (int): Interval between detecting
                and adding new points.
            max_new_points (int): Maximum number of newly added points.
        
        """

        self.tracks = []
        self.frame_idx = 0
        self.prev_frame_gray = None
        self.prev_roi = None
        self.good_threshold = 1

        self.min_points = min_points  # 10
        self.min_tracks_for_motion = min_tracks_for_motion  # 5
        self.min_track_len_for_motion = min_track_len_for_motion  # 10
        self.motion_percent_threshold = motion_percent_threshold  # 0.005  # given as % of larger frame side
        self.motion_voting_ratio = motion_voting_ratio  # 0.3  # percent of moving points to consider the object as moving
        
        self.max_track_len = max_track_len  # 50
        self.detect_new_pt_interval = detect_new_pt_interval  # 5, 10
        self.max_new_points = max_new_points  # 250

        self.lk_params = dict(winSize=(19, 19), #(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.feature_params = dict(maxCorners=max_new_points,  # 500
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

    def visualize(self, frame=None):
        """Visualize keypoint tracks.
        
        Args:
            frame (_type_): Image to be used for visualization.
                If no image is supplied, the last gray frame is used. 
        
        Returns:
            Image with visualized keypoint tracks.
        """

        # if no frame is supplied, the last gray frame is used
        if frame is None:
            frame = self.prev_frame_gray.copy()
        vis = frame

        if len(self.tracks) > 0:
            for track in self.tracks:
                x, y = track[-1]
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

        return vis

    def detect(self, next_frame, roi=None):
        """Detect and store optical flow.
        
        Args:
            next_frame (_type_): Next image on which detection is 
                performed.
            roi (list): Image ROI for detection.

        Returns:
            Keypoint tracks.
        """

        # roi is used when adding new points (roi format: [xmin, ymin, xmax, ymax])
        if roi is not None:
            assert len(roi) == 4 and roi[0] <= roi[2] and roi[1] <= roi[3]
        frame_gray = next_frame
        if len(next_frame.shape) == 3:
            frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        if len(self.tracks) > 0:
            img0, img1 = self.prev_frame_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < self.good_threshold
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                if not (self._pt_in_roi(x, y, roi) or self._pt_in_roi(x, y, self.prev_roi)):
                    continue
                tr.append((x, y))
                if len(tr) > self.max_track_len:
                    del tr[0]
                new_tracks.append(tr)
            self.tracks = new_tracks

        if self.frame_idx % self.detect_new_pt_interval == 0 or len(self.tracks) < self.min_points:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            frame_crop = frame_gray
            if roi is not None:
                x1, y1 = max(int(roi[0]), 0), max(int(roi[1]), 0)
                x2, y2 = min(int(roi[2]), frame_gray.shape[1]), min(int(roi[3]), frame_gray.shape[0])
                frame_crop = frame_gray[y1:y2, x1:x2]
                mask = mask[y1:y2, x1:x2]
            p = cv2.goodFeaturesToTrack(frame_crop, mask=mask, **self.feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    if roi is not None:
                        x, y = x + x1, y + y1
                    self.tracks.append([(x, y)])

        self.frame_idx += 1
        self.prev_frame_gray = frame_gray
        self.prev_roi = roi

        return self.tracks

    def is_moving(self):
        """Determine if the object should be cosidered as moving.
        
        Returns:
            int: Information about estimated movement.
                For moving object, 1 is returned.
                For still objetc, 0 is returned.
                In case there is not sufficient information, 
                -1 is returned.
        """

        # returns 1, 0 or -1 (when available information is not enough to determine if motion occured)
        motion_sizes = [self.motion_size(t) for t in self.tracks if len(t) > self.min_track_len_for_motion]
        if len(motion_sizes) < self.min_tracks_for_motion:
            return -1
        movement_limit = max(self.prev_frame_gray.shape) * self.motion_percent_threshold
        movements = [m for m in motion_sizes if m > movement_limit]
        # decide the movement based on direct voting of tracks
        num_moving = len(movements)
        voting_threshold = len(motion_sizes) * self.motion_voting_ratio
        if num_moving >= voting_threshold:
            return 1
        if num_moving >= 3:
            # ignore all still points when deciding about movement
            return 1
        else:
            return 0

    def motion_size(self, track):
        """Find size of the movement of given track.
        
        Movement size is the distance between the first and the last 
        point of the keypoint track.
        """

        if len(track) < 2:
            return 0
        p1, p2 = track[0], track[-1]
        x1, y1 = p1
        x2, y2 = p2
        
        # euclidean distance
        dist = sqrt((x2-x1)**2 + (y2-y1)**2)
        return dist

    def _pt_in_roi(self, x, y, roi):
        """Determine if given point is inside roi."""

        if roi is None:
            return True
        x_min, y_min, x_max, y_max = roi
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        return False
