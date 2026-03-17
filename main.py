import cv2
from hand_tracker import HandTracker
from cursor_controller import CursorController


def main():
    """
    Main entry point for the hand-tracking mouse control application.

    This function initializes the webcam feed, hand tracker, and cursor controller. It continuously captures video
    frames, detects the position of the user's index finger, applies smoothing to reduce jitter, and moves the system
    cursor based  on the detected hand position.

    Controls:
    - Press 'ESC' to exit the application.

    :return:
        None
    """

    # Initialize webcam (device 0 = default camera)
    cap = cv2.VideoCapture(0)

    # Create instances for hand tracking and cursor control
    tracker = HandTracker()
    cursor = CursorController()

    # Store previous smoothed cursor positions
    prev_x, prev_y = 0, 0

    # Smoothing factor (lower = smoother, higher = more responsive)
    alpha = 0.35

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Flip frame horizontally for mirror-like interaction
        frame = cv2.flip(frame, 1)

        # Get the index finger position (x, y) from the tracker
        position = tracker.get_index_position(frame)

        if position:

            x, y = position
            h, w, _ = frame.shape

            # Apply exponential smoothing to reduce cursor jitter
            smooth_x = int(alpha * x + (1 - alpha) * prev_x)
            smooth_y = int(alpha * y + (1 - alpha) * prev_y)

            # Update previous values
            prev_x, prev_y = smooth_x, smooth_y

            # Move cursor relative to screen size
            cursor.move_cursor(smooth_x, smooth_y, w, h)

            # Draw a green circle at the detected finger position
            cv2.circle(frame, (x, y), 10, (0,255,0), -1)

        # Show the processed frame
        cv2.imshow("Hand Mouse", frame)

        # Exit when ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()