import cv2
import numpy as np
import os

VIDEO_PATH = "Video/2025/Alle_Spiele/"
OUTPUT_PATH = "presnap_frames_optical/"

ROI_TOP = 0.20
ROI_BOTTOM = 0.90
ROI_LEFT = 0.10
ROI_RIGHT = 0.90

MOTION_THRESHOLD = 1.5
MIN_STATIC_FRAMES = 5

os.makedirs(OUTPUT_PATH, exist_ok=True)

def apply_roi(frame):
    h, w = frame.shape[:2]
    return frame[
        int(h * ROI_TOP):int(h * ROI_BOTTOM),
        int(w * ROI_LEFT):int(w * ROI_RIGHT)
    ]

def stabilize_frame(prev_gray, gray):
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    if prev_pts is None:
        return gray

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

    valid_prev = prev_pts[status == 1]
    valid_curr = curr_pts[status == 1]

    if len(valid_prev) < 10:
        return gray

    matrix, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
    if matrix is None:
        return gray

    h, w = gray.shape
    stabilized = cv2.warpAffine(gray, matrix, (w, h))

    return stabilized

def extract_presnap_frame(video_file):
    cap = cv2.VideoCapture(video_file)

    ret, frame = cap.read()
    if not ret:
        return None

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    prev_gray = apply_roi(prev_gray)

    static_counter = 0
    presnap_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        roi_gray = apply_roi(gray)

        roi_gray = stabilize_frame(prev_gray, roi_gray)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, roi_gray, None,0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion = np.mean(mag)

        if motion < MOTION_THRESHOLD:
            static_counter += 1
            presnap_frame = frame.copy()
        else:
            if static_counter >= MIN_STATIC_FRAMES:
                break
            static_counter = 0

        prev_gray = roi_gray

    cap.release()
    return presnap_frame

for file in os.listdir(VIDEO_PATH):
    if file.endswith(".mp4"):
        video_file = os.path.join(VIDEO_PATH, file)
        print(f"Verarbeite: {video_file}")

        frame = extract_presnap_frame(video_file)

        if frame is not None:
            output_file = os.path.join(OUTPUT_PATH, file.replace(".mp4", "_presnap.jpg"))
            cv2.imwrite(output_file, frame)
            print(f"→ Pre-Snap gespeichert: {output_file}")
        else:
            print(f"⚠ Kein Pre-Snap erkannt in {file}")
