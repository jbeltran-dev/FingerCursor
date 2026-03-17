import pyautogui


class CursorController:
    """
    Class responsible for controlling the mouse cursor movement
    based on coordinates coming from an external source (e.g., a camera).
    """

    def __init__(self):
        """
        Initializes the cursor controller.
        Retrieves the screen size in pixels in order to properly map external coordinates to the screen coordinate
        system.
        """

        # Get screen width and height
        self.screen_w, self.screen_h = pyautogui.size()

    def move_cursor(self, x, y, cam_w, cam_h):
        """
        Moves the mouse cursor to a specific position on the screen, scaling coordinates from an external system
        (such as a camera).

        :param x: X coordinate detected (e.g., from the camera).
        :param y: Y coordinate detected (e.g., from the camera).
        :param cam_w: Width of the camera resolution.
        :param cam_h: Height of the camera resolution.
        :return: None
        """

        # Scale X coordinate from camera space to screen width
        screen_x = int(x / cam_w * self.screen_w)

        # Scale Y coordinate from camera space to screen height
        screen_y = int(y / cam_h * self.screen_h)

        # Move the cursor to the calculated position
        pyautogui.moveTo(screen_x, screen_y)