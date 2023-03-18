
import cv2
import numpy as np

from era_5g_train_detection_standalone.modules.third_party.sort_tracker import SortTracker
from era_5g_train_detection_standalone.modules.optical_flow_KLT import OpticalFlowKLT


class DetectionProcessing():
    """Processing of detected objects in order to determine movement.
    
    Detected objects are first assigned to tracks using SORT tracker.
    Kypoints are then detected and tracked wihin each object track. 
    Movement of keypoints is then used to determine if the particular
    object is moving or not.
    """

    def __init__(self, sort_tracker_params=None, optical_flow_params=None, render_detections=True, verbose=True):
        """Initialize object for processing of detections.

        Args:
            sort_tracker_params (dict, optional): Dictionary with 
                optional SORT tracker parameters.
            optical_flow_params (dict, optional): Dictionary with 
                optional parameters for optical flow calculation.
            render_detections (bool, optional): Create output 
                images with detected movements.
            verbose (bool, optional): Print debug information.
        """

        if sort_tracker_params is not None:
            self.sort_tracker_params = sort_tracker_params
        else:
            self.sort_tracker_params = {
                "max_age": 50,
                "min_hits": 0,
                "iou_threshold": 0.2,
                }
        
        if optical_flow_params is not None:
            self.optical_flow_params = optical_flow_params
        else:
            self.optical_flow_params = {
                "min_points": 10,
                "min_tracks_for_motion": 5,
                "min_track_len_for_motion": 5,
                "motion_percent_threshold": 0.005,  # given as % of larger frame side
                "motion_voting_ratio": 0.25,  # percent of moving points to consider the object as moving
                "max_track_len": 50,
                "detect_new_pt_interval": 5,
                "max_new_points": 250,
                }

        self.sort_tracker = SortTracker(**self.sort_tracker_params)
        self.optical_flow_by_tracks = {}

        self.render_detections = render_detections
        self.verbose = verbose

    def process_detections(self, data):
        """Determines movements from object detections.
        
        Object detections are assigned to tracks and movement is then
        detected using optical flow.

        Args:
            data (dict): A dictionary containing items:
                "image" with original BGR image
                "bboxes" with detected bounding boxes 
                "scores" with detection scores 

        Returns:
            dict: Original data dict with added item "movements" that
                contains information about detected movements and
                optionally also item "output_bgr_image" with 
                visualization (depending on whether the class parameter
                render_detections is set)
        """

        raw_image = data["image"]
        bboxes, scores = data["bboxes"], data["scores"]

        # Prepare input for sort tracker.
        if bboxes.shape[0] == 0:
            boxes_w_scores = np.empty((0,5))
        else:
            boxes_w_scores = np.concatenate((bboxes, scores.reshape((-1, 1))), axis=1)

        # Update SORT tracker.
        sort_tracks = self.sort_tracker.update(boxes_w_scores)

        # Update optical flows.
        output_movements = self.update_optical_flows(sort_tracks, raw_image)

        # Create output image if desired.
        if self.render_detections:
            out_img = self.render_output_image(sort_tracks, raw_image)
            data["output_bgr_image"] = out_img

        data["movements"] = output_movements

        return data

    def update_optical_flows(self, sort_tracks, raw_img):
        """Update optical flows within each of the object tracks.

        Args:
            sort_tracks: Object tracks obtained from SORT tracker
            raw_img: Original image.

        Returns:
            list: List with information about detected movements.
        """

        tracks_ids = sort_tracks[:, 4]

        output_movements = []

        # create new optical flow objects for new tracks
        existing_opt_flows = list(self.optical_flow_by_tracks.keys())
        for track_id in tracks_ids:
            if track_id not in existing_opt_flows:
                self.optical_flow_by_tracks[track_id] = OpticalFlowKLT(**self.optical_flow_params)
        
        # remove optical flows for tracks that do not exist anymore
        for old_flow_id in existing_opt_flows:
            if old_flow_id not in tracks_ids:
                del(self.optical_flow_by_tracks[old_flow_id])

        # update optical flows and determine if the track is moving
        frame_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        for track in sort_tracks:
            self.optical_flow_by_tracks[track[4]].detect(frame_gray, track[0:4])
            
            moving = self.optical_flow_by_tracks[track[4]].is_moving()
            out_movement_data = {"bbox": track.tolist()[0:4], "moving": moving}  # "track_id": track[4],
            output_movements.append(out_movement_data)

        return output_movements

    def render_output_image(self, sort_tracks, raw_img):
        """Create new image and draw detected objects and movements."""

        vis_img = raw_img.copy()

        # BGR colors
        colors = {
            "green": (0, 255, 0), 
            "blue": (255, 0, 0), 
            "red": (0, 0, 255), 
            "orange": (0, 165, 255)
            }
        
        # draw detection rect
        for t in sort_tracks:
            moving = self.optical_flow_by_tracks[t[4]].is_moving()
            if moving == -1:
                color = colors["orange"]
            elif moving == 1:
                color = colors["red"]
            else:
                color = colors["green"]
            thickness = 2
            vis_img = cv2.rectangle(vis_img, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
        
        # draw optical flows
        for opt_flow in self.optical_flow_by_tracks.values():
            opt_flow.visualize(vis_img)
        
        return vis_img

    def clear_tracks(self):
        "Reset internal state."

        # reset sort tracker
        self.sort_tracker = SortTracker(**self.sort_tracker_params)

        # reset optical flows
        self.optical_flow_by_tracks = {}

