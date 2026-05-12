import cv2
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    """
    Hand tracking utility using MediaPipe Hand Landmarker.

    This class detects a single hand in a video stream and extracts the position of the index finger tip
    (landmark 8). It is designed to work in VIDEO mode, meaning it requires timestamps for each frame.
    """

    # Path to the MediaPipe Hand Landmarker model
    def __init__(self, model_path="hand_landmarker.task"):
        """
        Initialize the HandTracker with a MediaPipe model.

        :param: model_path (str): Path to the MediaPipe hand landmarker model file.
        """

        # Configure base options with the model file
        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        # Configure the hand landmarker
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )

        # Create the hand landmarker detector
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Store initial time to generate timestamps later
        self.start_time = time.time()

    def get_index_position(self, opencv_frame):
        """
        Detect the index fingertip position in a given frame.

        :param opencv_frame: Frame captured using OpenCV (BGR format).
        :return:
            tuple[int, int] | None:
                - (x, y): Pixel coordinates of the index fingertip.
                - None: If no hand is detected.
        """

        # Convert BGR (OpenCV format) to RGB (required by MediaPipe)
        rgb_frame = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)

        # Generate timestamp in milliseconds (required for VIDEO mode)
        timestamp_ms = int((time.time() - self.start_time) * 1000)

        # Convert NumPy array to MediaPipe Image object
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Run hand detection on the current frame
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        # If no hands are detected, return None
        if not result.hand_landmarks:
            return None

        # Get frame dimensions
        height, width, _ = opencv_frame.shape

        # Landmark 8 corresponds to the tip of the index finger
        index_tip = result.hand_landmarks[0][8]

        # Convert normalized coordinates (0–1) to pixel coordinates
        x = int(index_tip.x * width)
        y = int(index_tip.y * height)
        z = index_tip.z
        print("Deep: ", z)
        return x, y

    def get_full_result(self, opencv_frame):
        """
        """
        rgb_frame = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect_for_video(mp_image, timestamp_ms)