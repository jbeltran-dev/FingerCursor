import cv2
from hand_tracker import HandTracker
from cursor_controller import CursorController
from click_detector import ClickDetector

# ── Configuration ─────────────────────────────────────────────────────────────

ALPHA = 0.30    # EMA smoothing factor (0 = frozen, 1 = no smoothing)
DEAD_ZONE = 3   # Minimum pixel delta required to update cursor position
INDEX_TIP = 6   # Hand landmark used as cursor anchor (6 = index PIP joint)

# ── Visual constants ───────────────────────────────────────────────────────────

COLOR_IDLE    = (255,   0,   0)   # Blue  — cursor dot when not clicking
COLOR_CLICK   = (  0, 255,   0)   # Green — cursor dot when click is active
COLOR_SMOOTH  = (255, 255, 255)   # White — smoothed position dot
COLOR_ZONE    = (  0, 255, 150)   # Teal  — active zone rectangle

def main():
    """
    Main loop: captures webcam frames, detects hand landmarks, applies smoothing, and delegates cursor movement and
    click detection to their respective controllers.
    """

    cap     = cv2.VideoCapture(0)
    tracker = HandTracker()
    cursor  = CursorController()
    click   = ClickDetector(cursor)

    prev_x: int = 0
    prev_y: int = 0

    print("--- Gesture Mouse active | Press ESC to exit ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame so movement feels natural (like a mirror)
        frame   = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        result  = tracker.get_full_result(frame)

        if result and result.hand_landmarks:
            hand = result.hand_landmarks[0]

            # ── Cursor anchor position (raw pixel coordinates) ──────────────────
            tip = hand[INDEX_TIP]
            ix  = int(tip.x * w)
            iy  = int(tip.y * h)

            # ── EMA smoothing + dead zone ───────────────────────────────────
            if abs(ix - prev_x) < DEAD_ZONE and abs(iy - prev_y) < DEAD_ZONE:
                sx, sy = prev_x, prev_y
            else:
                sx = int(ALPHA * ix + (1 - ALPHA) * prev_x)
                sy = int(ALPHA * iy + (1 - ALPHA) * prev_y)

            prev_x, prev_y = sx, sy

            # ── Click detection & cursor movement ──────────────────────────
            click.update(hand, sx, sy, w, h)

            # ── Overlay: landmark dot (raw) ────────────────────────────────
            color_tip = COLOR_CLICK if click.pressing else COLOR_IDLE
            cv2.circle(frame, (ix, iy), 10, color_tip, -1)
            cv2.circle(frame, (sx, sy),  6, color_tip, -1)

            # ── Overlay: smoothed position dot ─────────────────────────────
            cv2.circle(frame, (sx, sy), 6, COLOR_SMOOTH, -1)

            # ── Overlay: status text ───────────────────────────────────────
            status      = "CLICK" if click.pressing else f"cursor: ({sx}, {sy})"
            text_color  = COLOR_CLICK if click.pressing else COLOR_SMOOTH
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)

        # ── Overlay: active zone rectangle (always drawn) ──────────────────
        x1, y1, x2, y2 = cursor.get_active_rect(w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ZONE, 1)

        cv2.imshow("Gesture Mouse", frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
            break

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    click.release()


if __name__ == "__main__":
    main()