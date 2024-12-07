import math

import numpy as np


class SpeedTrackerAnalysis:

    def __init__(self, fps):
        self.object_id = None
        self._positions = {}
        self._current_speed = {}
        self.fps = fps

    def set_object_id(self, object_id):
        self.object_id = object_id
        if self._positions.get(object_id) is None:
            self._positions[object_id] = []
            self._current_speed[object_id] = 0

    def analyze_object_speed(self, xyxy, scene_index):
        self._positions[self.object_id].append(xyxy)
        if scene_index % int(self.fps / 9) == 0:
            speed_detected = self._get_speed(scene_index)
            self._current_speed[self.object_id] = speed_detected
        return self._current_speed[self.object_id]

    def _get_speed(self, index_frame):
        frame_time = index_frame / self.fps
        if index_frame == 0:
            return 0

        print(f"============== Frame time: {frame_time} ============")

        curr_speed = 0

        # check that speed not jumping too much
        if self._positions[self.object_id].__len__() > 2:
            prev_speed = self._current_speed[self.object_id]
            curr_speed = self._calculate_speed(frame_time)
            if abs(prev_speed - curr_speed) > 80:
                return prev_speed

            if curr_speed == 0 and prev_speed > 5:
                return prev_speed

        return curr_speed

    def _calculate_speed(self, time_delta):
        """
        Calculate the speed of an object based on its coordinates at two different time frames.

        Parameters:
        coord_prev (tuple): The (x1, y1, x2, y2) coordinates of the object in the previous frame.
        coord_curr (tuple): The (x1, y1, x2, y2) coordinates of the object in the current frame.
        time_delta (float): The time difference between the two frames in seconds.

        Returns:
        float: The speed of the object in units per second.
        """

        coord_prev = self._positions[self.object_id][self._positions[self.object_id].__len__() - 2]
        coord_curr = self._positions[self.object_id][self._positions[self.object_id].__len__() - 1]

        # Calculate the center of the bounding box for previous and current coordinates
        center_prev = ((coord_prev[0] + coord_prev[2]) / 2, (coord_prev[1] + coord_prev[3]) / 2)
        center_curr = ((coord_curr[0] + coord_curr[2]) / 2, (coord_curr[1] + coord_curr[3]) / 2)

        # Calculate the Euclidean distance between the centers of the bounding boxes
        distance = math.sqrt((center_curr[0] - center_prev[0]) ** 2 + (center_curr[1] - center_prev[1]) ** 2)

        fps_delta = 25
        if self.fps > fps_delta:
            fps_delta = (self.fps / 3.4)

        # Calculate the speed
        speed = (distance / time_delta * fps_delta) * 100
        return speed
