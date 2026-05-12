"""
cursor_controller.py — Screen Cursor Controller
------------------------------------------------
Maps hand landmark positions from camera space to screen space.
Only a centered fraction of the frame (ACTIVE_ZONE) is used as input,
so the hand never needs to reach the edges of the frame to hit a screen corner.
"""

import pyautogui

# Disable fail-safe and default action delay
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0


class CursorController:
    """
    Translates smoothed landmark coordinates from camera space to screen coordinates using a configurable active
    zone.
    """
    # Fraction of the camera frame used as input (0.6 = small zone, 0.8 = large zone)
    ACTIVE_ZONE = 0.65

    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()

    def get_active_rect(self, cam_w, cam_h):
        """
        Returns the active zone as (x1, y1, x2, y2) in camera pixels.
        Useful for drawing the zone rectangle on the frame.
        """
        margin_x = int(cam_w * (1 - self.ACTIVE_ZONE) / 2)
        margin_y = int(cam_h * (1 - self.ACTIVE_ZONE) / 2)

        return (
            margin_x,
            margin_y,
            cam_w - margin_x,
            cam_h - margin_y
        )

    def move_cursor(self, x, y, cam_w, cam_h):
        """
        Moves the system cursor to the position corresponding to (x, y) in camera space, clamped and scaled through the
        active zone.

        :param x: Smoothed X landmark position in camera pixels.
        :param y: Smoothed Y landmark position in camera pixels.
        :param cam_w: Camera frame width in pixels.
        :param cam_h: Camera frame height in pixels.
        """
        margin_x = cam_w * (1 - self.ACTIVE_ZONE) / 2
        margin_y = cam_h * (1 - self.ACTIVE_ZONE) / 2

        # Clamp to active zone boundaries
        x = max(margin_x, min(cam_w - margin_x, x))
        y = max(margin_y, min(cam_h - margin_y, y))

        # Scale active zone → full screen
        screen_x = int((x - margin_x) / (cam_w * self.ACTIVE_ZONE) * self.screen_w)
        screen_y = int((y - margin_y) / (cam_h * self.ACTIVE_ZONE) * self.screen_h)

        pyautogui.moveTo(screen_x, screen_y, duration=0)