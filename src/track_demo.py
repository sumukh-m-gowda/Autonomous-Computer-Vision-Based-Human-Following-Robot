"""Live person detection with a follow command (steering + speed).

Part 2 of the project: the robot must detect a person and follow them.

Pipeline, per camera frame:
  1. A pretrained SSDLite-MobileNetV3 detector (torchvision, COCO classes)
     returns boxes. We keep only class 1 = "person".
  2. The target is the largest person box (no memory between frames, so it
     can switch people if several are visible - see README limitations).
  3. STEERING from the box's horizontal offset from the image centre.
  4. SPEED from the box height as a fraction of the frame height, which is a
     crude distance proxy: a small box means far away, so go faster.

Usage (from the project root, with .venv active):
    python src/track_demo.py                 # webcam 0
    python src/track_demo.py --camera 1      # another camera
    python src/track_demo.py --video clip.mp4

Keys:  q = quit,  s = save a snapshot to Output/
First run downloads the model weights (~13 MB), so it needs internet once.
"""
import argparse
import os
import time

import cv2

PERSON_CLASS = 1        # COCO label id for "person"
DEAD_ZONE = 0.12        # |offset| below this -> go straight
TARGET_HEIGHT = 0.60    # desired box height / frame height (follow distance)
HEIGHT_TOLERANCE = 0.08 # within +/- this of the target -> hold position


def follow_command(box, frame_w, frame_h):
    """Turn one person box into a follow command.

    Args:
        box: (x1, y1, x2, y2) in pixels.
    Returns:
        (steer, speed, label)
        steer: -1.0 (full left) .. +1.0 (full right), 0 = straight
        speed: 0.0 (stop) .. 1.0 (full speed forward)
        label: short text describing the command
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    # Horizontal offset of the box centre, normalised to [-1, 1]
    offset = (cx - frame_w / 2.0) / (frame_w / 2.0)
    steer = 0.0 if abs(offset) < DEAD_ZONE else max(-1.0, min(1.0, offset))

    # Box height as a fraction of the frame: bigger = person is closer
    height_ratio = (y2 - y1) / float(frame_h)
    if height_ratio > TARGET_HEIGHT + HEIGHT_TOLERANCE:
        speed, move = 0.0, "TOO CLOSE - STOP"
    elif height_ratio >= TARGET_HEIGHT - HEIGHT_TOLERANCE:
        speed, move = 0.0, "HOLD"
    else:
        # The further away (smaller box), the faster, capped at 1.0
        speed = min(1.0, (TARGET_HEIGHT - height_ratio) / TARGET_HEIGHT)
        move = "FORWARD"

    if steer == 0.0:
        turn = "STRAIGHT"
    elif steer < 0:
        turn = "LEFT"
    else:
        turn = "RIGHT"
    return steer, speed, f"{move} | {turn}"


def load_detector(device):
    """Load the pretrained SSDLite-MobileNetV3 detector (COCO)."""
    import torch
    from torchvision.models.detection import (
        ssdlite320_mobilenet_v3_large,
        SSDLite320_MobileNet_V3_Large_Weights,
    )
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights, score_thresh=0.05)
    model.to(device).eval()
    return model, torch


def detect_people(model, torch, frame_bgr, device, conf):
    """Return a list of (x1, y1, x2, y2, score) for people above `conf`."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
    with torch.inference_mode():
        out = model([tensor])[0]
    people = []
    for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
        if int(label) == PERSON_CLASS and float(score) >= conf:
            x1, y1, x2, y2 = [float(v) for v in box]
            people.append((x1, y1, x2, y2, float(score)))
    return people


def draw_overlay(frame, people, target, steer, speed, label, fps):
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)  # image centre
    for p in people:
        x1, y1, x2, y2, s = [int(v) if i < 4 else v for i, v in enumerate(p)]
        is_target = target is not None and p is target
        colour = (0, 255, 0) if is_target else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"person {s:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    if target is not None:
        cx = int((target[0] + target[2]) / 2)
        cv2.circle(frame, (cx, int((target[1] + target[3]) / 2)), 5, (0, 255, 0), -1)
        text = f"{label}  steer={steer:+.2f}  speed={speed:.2f}"
    else:
        text = "NO PERSON - STOP"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"{fps:.1f} FPS", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def open_source(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():  # on Windows the DirectShow backend sometimes works
            cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=str, default=None, help="video file instead of webcam")
    ap.add_argument("--conf", type=float, default=0.5, help="min detection confidence")
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto)")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading detector on {device} ...", flush=True)
    model, torch = load_detector(device)

    cap = open_source(args)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera/video. Try --camera 1 or --video file.mp4")

    os.makedirs("Output", exist_ok=True)
    snap_n, fps, last = 0, 0.0, time.time()
    print("Running. Press q to quit, s to save a snapshot.", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        people = detect_people(model, torch, frame, device, args.conf)

        target, steer, speed, label = None, 0.0, 0.0, ""
        if people:
            # Target = largest box (closest person); no tracking between frames
            target = max(people, key=lambda p: (p[2] - p[0]) * (p[3] - p[1]))
            steer, speed, label = follow_command(target[:4], w, h)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        draw_overlay(frame, people, target, steer, speed, label, fps)
        cv2.imshow("Human-following demo (q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = f"Output/track_snapshot_{snap_n}.jpg"
            cv2.imwrite(path, frame)
            print("saved", path, flush=True)
            snap_n += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()