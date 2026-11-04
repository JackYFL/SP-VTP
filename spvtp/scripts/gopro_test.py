#!/usr/bin/env python3
"""Simple GoPro streaming tester.

Try a USB device or a video file and display frames. Exits non-zero on failure.
"""
import argparse
import time
import sys

import cv2


def main():
    parser = argparse.ArgumentParser(description="Minimal GoPro stream tester")
    parser.add_argument("--device", type=int, default=1, help="Video device index (e.g. 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to GoPro video file")
    parser.add_argument("--width", type=int, default=None, help="Requested width")
    parser.add_argument("--height", type=int, default=None, help="Requested height")
    parser.add_argument("--fps", type=int, default=None, help="Requested fps")
    args = parser.parse_args()

    if args.video:
        src_desc = args.video
        cap = cv2.VideoCapture(args.video)
    else:
        src_desc = f"camera {args.device}"
        cap = cv2.VideoCapture(args.device)

    if not cap or not cap.isOpened():
        print(f"Failed to open {src_desc}")
        return 2

    # Try to set requested properties
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opened {src_desc}: {w}x{h} @ {fps:.2f} fps")

    frame_count = 0
    start_t = time.time()
    no_frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                no_frame_count += 1
                print(f"No frame data from {src_desc} (retry {no_frame_count})", end="\r", flush=True)
                if no_frame_count > 60:
                    print(f"\nNo frame data from {src_desc} (giving up)")
                    return 3
                time.sleep(0.05)
                continue

            no_frame_count = 0
            frame_count += 1
            elapsed = time.time() - start_t if start_t else 0.0
            cur_fps = frame_count / elapsed if elapsed > 0 else 0.0

            # Overlay info and show
            text = f"{w}x{h} {cur_fps:.1f}FPS - {src_desc}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("GoPro Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
