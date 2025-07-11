from functools import wraps
import time
import cv2
import math
import numpy as np
def log_inference_time(func):
    """Decorator to log inference time for any function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"{func.__name__} time: {inference_time:.4f} seconds")
        return result
    return wrapper

class MovementTrailVideo:
    def __init__(self, max_trail_length=50):
        self.trail_points = []
        self.max_trail_length = max_trail_length
    # @log_inference_time
    def draw_trail_opencv(self, frame, current_pos):
        """Draw trail using OpenCV"""
        # Add current position to trail
        self.trail_points.append(current_pos)

        # Limit trail length
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)

        # Draw trail with fading effect
        for i, point in enumerate(self.trail_points):
            if i == 0:
                continue

            # Calculate fade factor (newer points are brighter)
            fade_factor = i / len(self.trail_points)

            # Draw line segment
            thickness = max(1, int(fade_factor * 8))
            alpha = int(fade_factor * 255)

            # Create color (blue to red gradient)
            color = (int(255 * (1 - fade_factor)),
                     int(100 * fade_factor), int(255 * fade_factor))

            cv2.line(frame, self.trail_points[i-1], point, color, thickness)

        # Draw current position as a circle
        # cv2.circle(frame, current_pos, 15, (0, 255, 0), -1)
        # cv2.circle(frame, current_pos, 15, (255, 255, 255), 2)

        return frame


def add_point(idx, array, distance, new_point):
    x_new, y_new = new_point[:2]

    for _idx, obj in enumerate(array):
        # if _idx == idx:
        #     continue
        x_existing, y_existing = obj['point'][:2]
        _distance = math.dist([x_existing, y_existing], [x_new, y_new])

        if _idx == idx:
            if _distance > 300:
                del array[_idx]
            continue

        if _distance == distance:
            del array[_idx]
            return
    array.append({'point': new_point, 'video': MovementTrailVideo()})

class InferenceModel:
    def __init__(self,model='yolo'):
        self.runner = model
        from ultralytics import YOLO
        self.model = YOLO("model_weight/best.pt")
            
    def get_model(self):
        return self.model
  
    def inference(self,frame):
        return self.model.predict(source=frame, verbose=False)
            