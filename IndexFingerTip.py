import cv2
import time
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the MediaPipe Hand Landmarker model
MODEL_PATH = "hand_landmarker.task"

# Base configuration for the model
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

# Hand landmarker configuration
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

# Create the hand landmarker detector
detector = vision.HandLandmarker.create_from_options(options)

# Drawing utilities
mp_hand_connections  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

def draw_landmarks_on_image(rgb_image, hand_detection_result):
    """
    Draw detected hand landmarks on an RGB image.

    :param rgb_image: Image in RGB format (height x width x 3).
    :param hand_detection_result: Result object returned by detector.detect_for_video(...)
    :return: A copy of the input image with hand landmarks drawn. If no hands are detected, the image is returned unchanged
    """

    # Create a copy to avoid modifying the original image
    annotated_image = np.copy(rgb_image)

    # If no hands were detected, return the image unchanged
    if not hand_detection_result.hand_landmarks:
        return annotated_image

    height, width, _ = annotated_image.shape

    # Draw landmarks for each detected hand
    for hand_landmarks in hand_detection_result.hand_landmarks:
        for idx, landmark in enumerate(hand_landmarks):

            # Convert normalized coordinates to pixels
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # If it's the index finger, then color it green.
            if idx == 8:
                # Finger as a red color
                color = (255, 0, 0)

                # Draw the point
                cv2.circle(annotated_image, (x, y), 6, color, -1)

    return annotated_image


# Open default camera (0 = built-in webcam)
cap = cv2.VideoCapture(0)

# Used to generate increasing timestamps (required for VIDEO mode)
start_time = time.time()

while True:
    ret, frame = cap.read()

    # If frame was not captured correctly, exit loop
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Generate timestamp in milliseconds (required for VIDEO mode)
    timestamp_ms = int((time.time() - start_time) * 1000)

    # Convert numpy array to MediaPipe Image object
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Perform hand detection
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    # Draw landmarks on the RGB image
    annotated_rgb = draw_landmarks_on_image(rgb_frame, detection_result)

    # Convert back to BGR for OpenCV display
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    # Show result window
    cv2.imshow("Hand Tracking - MediaPipe", annotated_bgr)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()