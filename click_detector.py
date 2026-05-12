import math
import time
import pyautogui


def _planar_distance(a, b):
    """
    Euclidean distance between two landmarks using only X and Y axes. Z is intentionally ignored — depth estimation is
    unreliable on a standard monocular webcam.
    """
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

class ClickDetector:
    """
    Gesture-based left click detector using a thumb–index pinch.

    Thresholds are normalized by hand size, making detection robust to changes in distance between the hand and the
    camera.
    """

    # ── Pinch thresholds (normalized ratio) ───────────────────────────────────
    # Ratio = dist(thumb, index) / dist(wrist, middle_mcp)
    # Tune these values using the debug output after calibration.
    CLICK_THRESHOLD   = 0.20   # Ratio below this → fingers are pinching
    RELEASE_THRESHOLD = 0.30   # Ratio above this → fingers are open

    # ── Timing ────────────────────────────────────────────────────────────────
    COOLDOWN = 0.35            # Seconds to wait between consecutive clicks

    # ── Frame confirmation ─────────────────────────────────────────────────────
    FRAMES_TO_CLICK = 3
    FRAMES_TO_RELEASE = 3

    # ── MediaPipe landmark indices ─────────────────────────────────────────────
    THUMB_TIP = 3  # Thumb distal joint (where the nail is)
    INDEX_TIP = 8  # Index fingertip
    WRIST = 0  # Wrist — start of the reference segment
    MIDDLE_MCP = 9  # Middle finger MCP joint — end of the reference segment

    def __init__(self, cursor_controller):
        """
        :param cursor_controller: CursorController instance used to move the cursor when no click is active.
        """
        self.cursor          = cursor_controller
        self.pressing        = False    # True while mouseDown is active
        self.last_click      = 0.0      # Timestamp of the last confirmed click
        self.frames_closed   = 0        # Consecutive frames below CLICK_THRESHOLD
        self.frames_open     = 0        # Consecutive frames above RELEASE_THRESHOLD
        self._frame_count    = 0        # Internal counter used for debug throttling

    # ── Private helpers ───────────────────────────────────────────────────────

    def _normalized_dist(self, hand):
        """
        Compute the pinch ratio: distance between thumb and index tips divided by the hand's reference length
        (wrist → middle MCP).

        Normalizing by hand size makes the ratio stable across different distances from the camera — a smaller hand in
        frame produces a proportionally smaller pinch distance, so the ratio stays constant.

        :param hand: List of 21 MediaPipe NormalizedLandmark objects.
        :return: Normalized pinch ratio. Returns 1.0 if hand size is too small to compute reliably (avoids division
                 by zero).
        """
        pinch_dist = _planar_distance(hand[self.THUMB_TIP], hand[self.INDEX_TIP])
        hand_size  = _planar_distance(hand[self.WRIST], hand[self.MIDDLE_MCP])

        if hand_size < 0.001:
            return 1.0

        return pinch_dist / hand_size

    def update(self, hand_landmarks, sx, sy, cam_w, cam_h):
        """
        Process one frame. Must be called once per frame inside the main loop.

        Moves the cursor when idle, or fires mouseDown/mouseUp when the pinch gesture is confirmed over enough
        consecutive frames.

        The cursor is intentionally frozen during an active press so the click lands exactly where the user was pointing
        before pinching.

        :param hand_landmarks: List of 21 MediaPipe NormalizedLandmark objects.
        :param sx: Smoothed X position of the anchor landmark (camera pixels).
        :param sy: Smoothed Y position of the anchor landmark (camera pixels).
        :param cam_w: Camera frame width in pixels.
        :param cam_h: Camera frame height in pixels.
        """
        ratio = self._normalized_dist(hand_landmarks)
        now   = time.time()

        # ── Debug output (throttled) ───────────────────────────────────────
        self._frame_count += 1
        if self._frame_count % 3 == 0:
            print(
                f"ratio: {ratio:.3f} | "
                f"frames_closed: {self.frames_closed} | "
                f"pressing: {self.pressing}"
            )

        # ── Frame counters ─────────────────────────────────────────────────
        # Count consecutive frames in each state.
        # Frames in the dead band (between thresholds) leave counters unchanged,
        # which prevents a single noisy frame from resetting accumulated progress.
        if ratio < self.CLICK_THRESHOLD:
            self.frames_closed += 1
            self.frames_open = 0
        elif ratio > self.RELEASE_THRESHOLD:
            self.frames_open += 1
            self.frames_closed = 0

        # ── State machine ──────────────────────────────────────────────────
        if not self.pressing:
            if self.frames_closed >= self.FRAMES_TO_CLICK:
                if (now - self.last_click) > self.COOLDOWN:
                    # Pinch confirmed — fire click and freeze cursor
                    self.pressing = True
                    self.last_click = now
                    self.frames_closed = 0
                    pyautogui.mouseDown(button='left')
            else:
                # No active click — move cursor normally
                self.cursor.move_cursor(sx, sy, cam_w, cam_h)

        else:
            if self.frames_open >= self.FRAMES_TO_RELEASE:
                # Fingers separated — release click
                pyautogui.mouseUp(button='left')
                self.pressing = False
                self.frames_open = 0

    def release(self) -> None:
        """
        Force-release the mouse button and reset all state. Call this when the application exits to ensure the mouse
        is never left in a pressed state.
        """
        pyautogui.mouseUp(button='left')
        self.pressing      = False
        self.frames_closed = 0
        self.frames_open   = 0