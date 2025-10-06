import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Brush & canvas settings
brush_color = (0, 0, 255)
prev_brush_color = brush_color
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
color_index = 0
brush_thickness = 5
prev_brush_thickness = brush_thickness
default_brush_thickness = 5
eraser_thickness = 50
background_color = (0, 0, 0)
canvas = np.zeros((720, 1280, 3), np.uint8)
canvas[:] = background_color

undo_stack = deque(maxlen=20)
redo_stack = deque(maxlen=20)

# Shape and tool modes
shape_mode = None
shape_start = None
bucket_mode = False
eraser_mode = False
drawing_mode = True  # True = painting allowed

prev_x, prev_y = None, None  # track last brush point

def save_canvas(img):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"virtual_paint_{timestamp}.png"
    cv2.imwrite(filename, img)
    print(f"Canvas saved as {filename}")

def add_to_undo(img):
    undo_stack.append(img.copy())

def undo():
    if undo_stack:
        redo_stack.append(undo_stack.pop())
        return undo_stack[-1] if undo_stack else np.zeros_like(canvas)
    return canvas

def redo():
    if redo_stack:
        img = redo_stack.pop()
        undo_stack.append(img.copy())
        return img
    return canvas

def draw_brush_preview(img, position, color, thickness):
    cv2.circle(img, position, thickness, color, -1)

def fill_canvas(img, color):
    img[:] = color

def flood_fill(img, seed_point, new_color):
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img, mask, seed_point, new_color)

def get_index_finger_tip(hand_landmarks, shape):
    h, w = shape
    x = int(hand_landmarks.landmark[8].x * w)
    y = int(hand_landmarks.landmark[8].y * h)
    return x, y

def main():
    global brush_color, brush_thickness, canvas, background_color, color_index
    global shape_mode, shape_start, bucket_mode, eraser_mode, drawing_mode
    global prev_brush_color, prev_brush_thickness, prev_x, prev_y

    cap = cv2.VideoCapture(0)

    instructions = [
        "'C': Cycle color | 'R/G/B/Y': Select color",
        "'+/-': Brush thickness | 'Z': Reset thickness",
        "'E': Eraser toggle | 'P': Toggle drawing",
        "'X': Clear canvas | 'S': Save",
        "'U/D': Undo/Redo | 'L/T/O': Line/Rect/Circle",
        "'F': Fill/Bucket | 'K/W/M': Background",
        "Space: Finalise shape | ESC: Quit"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_frame)
        finger_pos = None

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            finger_pos = get_index_finger_tip(hand_landmarks, frame.shape[:2])
            x, y = finger_pos

            # Draw preview / shapes only if drawing is enabled
            if drawing_mode:
                # Free-hand brush or eraser
                if shape_mode is None and not bucket_mode:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y),
                                 brush_color, brush_thickness)
                    prev_x, prev_y = x, y

                # Shape preview
                elif shape_mode:
                    temp_canvas = canvas.copy()
                    if shape_start is None:
                        shape_start = (x, y)
                    else:
                        x0, y0 = shape_start
                        if shape_mode == 'line':
                            cv2.line(temp_canvas, (x0, y0), (x, y), brush_color, brush_thickness)
                        elif shape_mode == 'rectangle':
                            cv2.rectangle(temp_canvas, (x0, y0), (x, y), brush_color, brush_thickness)
                        elif shape_mode == 'circle':
                            radius = int(np.hypot(x - x0, y - y0))
                            cv2.circle(temp_canvas, (x0, y0), radius, brush_color, brush_thickness)
                    frame = cv2.addWeighted(frame, 0.5, temp_canvas, 0.5, 0)

                # Brush/Eraser preview
                if not bucket_mode and shape_mode is None:
                    draw_brush_preview(frame, (x, y), brush_color, brush_thickness)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            prev_x, prev_y = None, None

        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        for i, text in enumerate(instructions):
            cv2.putText(combined, text, (10, 30 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Virtual Painter", combined)

        key = cv2.waitKey(1) & 0xFF

        # Hotkeys
        if key == ord('c'):
            color_index = (color_index + 1) % len(colors)
            brush_color = colors[color_index]
            eraser_mode = False
        elif key == ord('r'):
            brush_color = (0, 0, 255)
            eraser_mode = False
        elif key == ord('g'):
            brush_color = (0, 255, 0)
            eraser_mode = False
        elif key == ord('b'):
            brush_color = (255, 0, 0)
            eraser_mode = False
        elif key == ord('y'):
            brush_color = (0, 255, 255)
            eraser_mode = False
        elif key == ord('+'):
            brush_thickness += 1
        elif key == ord('-'):
            brush_thickness = max(1, brush_thickness - 1)
        elif key == ord('z'):
            brush_thickness = default_brush_thickness
        elif key == ord('e'):
            if not eraser_mode:
                prev_brush_color = brush_color
                prev_brush_thickness = brush_thickness
                brush_color = background_color
                brush_thickness = eraser_thickness
                eraser_mode = True
                shape_mode = None
                bucket_mode = False
                prev_x, prev_y = None, None
            else:
                brush_color = prev_brush_color
                brush_thickness = prev_brush_thickness
                eraser_mode = False
        elif key == ord('p'):
            drawing_mode = not drawing_mode
            prev_x, prev_y = None, None
        elif key == ord('x'):
            add_to_undo(canvas)
            fill_canvas(canvas, background_color)
        elif key == ord('s'):
            save_canvas(canvas)
        elif key == ord('u'):
            canvas[:] = undo()
        elif key == ord('d'):
            canvas[:] = redo()
        elif key == ord('f'):
            bucket_mode = not bucket_mode
            shape_mode = None
            eraser_mode = False
            prev_x, prev_y = None, None
        elif key == ord('l'):
            shape_mode = 'line'
            shape_start = None
            bucket_mode = False
            eraser_mode = False
        elif key == ord('t'):
            shape_mode = 'rectangle'
            shape_start = None
            bucket_mode = False
            eraser_mode = False
        elif key == ord('o'):
            shape_mode = 'circle'
            shape_start = None
            bucket_mode = False
            eraser_mode = False
        elif key == ord('k'):
            background_color = (0, 0, 0)
            canvas[:] = background_color
        elif key == ord('w'):
            background_color = (255, 255, 255)
            canvas[:] = background_color
        elif key == ord('m'):
            background_color = (50, 50, 50)
            canvas[:] = background_color
        elif key == 27:
            break
        elif key == ord(' '):
            if shape_mode and shape_start and drawing_mode and finger_pos:
                add_to_undo(canvas)
                x0, y0 = shape_start
                x1, y1 = finger_pos
                if shape_mode == 'line':
                    cv2.line(canvas, (x0, y0), (x1, y1), brush_color, brush_thickness)
                elif shape_mode == 'rectangle':
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), brush_color, brush_thickness)
                elif shape_mode == 'circle':
                    radius = int(np.hypot(x1 - x0, y1 - y0))
                    cv2.circle(canvas, (x0, y0), radius, brush_color, brush_thickness)
                shape_start = None
                shape_mode = None

        # Fill
        if bucket_mode and finger_pos and drawing_mode:
            add_to_undo(canvas)
            flood_fill(canvas, finger_pos, brush_color)
            bucket_mode = False
            prev_x, prev_y = None, None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
