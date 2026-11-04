"""Shared RGB prompt rendering for direct decoding and predecoded caches."""
import cv2


def draw_prompt_boxes(frame, object_box=None, target_box=None):
    image = frame.copy()
    for box, label, color in ((object_box, 'obj', (0, 255, 0)),
                              (target_box, 'tgt', (255, 64, 64))):
        if box is None:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, color, 2, cv2.LINE_AA)
    return image
