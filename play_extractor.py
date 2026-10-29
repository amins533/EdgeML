import os
import sys
import gc
import json
import time
import sqlite3
import torch
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

# --- CONFIGURATION LINKED TO YOUR ACTUAL DIRECTORY ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "plays.db"
ANALYTICS_DB_PATH = BASE_DIR / "scouting_analytics.db"

# Setting the tracking target directly to your group's v6 model
PLAYER_MODEL_PATH = BASE_DIR / "models" / "spielererkennung" / "v6" / "weights" / "best.pt"

POST_SNAP_MAX_FRAMES = 90

class VideoPipeLine:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.stopped = False
    def start(self): return self
    def read(self):
        ret, frame = self.cap.read()
        if not ret: self.stopped = True; return None
        return frame
    def more(self): return self.cap.isOpened() and not self.stopped
    def stop(self): self.cap.release(); self.stopped = True

def resolve_video_directory():
    for folder_name in ["videos-ocr", "clips", "videos"]:
        target = BASE_DIR / folder_name
        if target.exists() and any(target.glob("*.mp4")):
            return target
    return BASE_DIR

VIDEOS_DIR = resolve_video_directory()

def setup_clean_databases():
    # Setup baseline tracking DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS play_trajectories (
            clip_id TEXT PRIMARY KEY,
            pre_snap_coords TEXT,
            post_snap_matrix TEXT
        )
    """)
    conn.commit()
    conn.close()

    # Setup New Analytics & Training Target DB
    conn_an = sqlite3.connect(ANALYTICS_DB_PATH)
    cursor_an = conn_an.cursor()
    cursor_an.execute("""
        CREATE TABLE IF NOT EXISTS standardized_trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_name TEXT,
            player_id INTEGER,
            frame_idx INTEGER,
            rel_x REAL,
            rel_y REAL
        )
    """)
    conn_an.commit()
    conn_an.close()

def get_completed_clips():
    if not DB_PATH.exists(): return set()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT clip_id FROM play_trajectories")
    clips = {row[0] for row in cursor.fetchall()}
    conn.close()
    return clips

def force_gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(0.1)

def main():
    print("==================================================")
    print("      GRUPPE 1 DIRECTORY EXTRACTION PIPELINE     ")
    print("==================================================")
    
    if not PLAYER_MODEL_PATH.exists():
        print(f"[ERROR] Could not find your group's v6 weights at: {PLAYER_MODEL_PATH}")
        return

    setup_clean_databases()
    video_files = list(VIDEOS_DIR.glob("*.mp4"))
    video_files.sort()
    
    print(f"[SYSTEM] Targeted video storage folder: {VIDEOS_DIR}")
    completed_clips = get_completed_clips()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[SYSTEM] Initializing player detection weights from: {PLAYER_MODEL_PATH.name}")
    shared_player_model = YOLO(str(PLAYER_MODEL_PATH))

    for video_index, video_path in enumerate(video_files):
        clip_id = video_path.stem
        if clip_id in completed_clips: continue

        print(f"\n[PROCESSING] Clip {video_index + 1}/{len(video_files)} -> {clip_id}")
        if video_index % 10 == 0: force_gpu_cleanup()

        pipeline = None
        try:
            pipeline = VideoPipeLine(video_path=str(video_path)).start()
            processed_frames = 0
            is_play_active = False
            snap_frame_idx = 0
            
            pre_snap_state = []
            raw_tracking_history = {} 
            pre_snap_anchor = None
            last_progress_time = time.time()

            while True:
                if time.time() - last_progress_time > 10.0:
                    print(f"  -> Timeout recovery triggered on {clip_id}")
                    break

                if not pipeline.more():
                    if pipeline.stopped: break
                    continue

                frame = pipeline.read()
                if frame is None: break
                processed_frames += 1
                last_progress_time = time.time()

                results = shared_player_model.track(
                    frame, conf=0.30, persist=True, verbose=False, tracker="bytetrack.yaml", device=device
                )
                
                if not results or results[0].boxes is None: continue

                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []

                current_frame_pts = []
                for box, t_id in zip(boxes, track_ids):
                    t_id_key = int(t_id)
                    center_x = float((box[0] + box[2]) / 2.0)
                    center_y = float((box[1] + box[3]) / 2.0)
                    current_frame_pts.append((center_x, center_y))

                    if is_play_active:
                        frame_offset = processed_frames - snap_frame_idx
                        if frame_offset < POST_SNAP_MAX_FRAMES:
                            if t_id_key not in raw_tracking_history:
                                raw_tracking_history[t_id_key] = {}
                            raw_tracking_history[t_id_key][frame_offset] = (center_x, center_y)

                # Automated Snap Identification via displacement variance
                if not is_play_active and len(current_frame_pts) >= 5 and processed_frames > 10:
                    current_anchor = np.mean(current_frame_pts, axis=0)
                    if pre_snap_anchor is None:
                        pre_snap_anchor = current_anchor
                        pre_snap_state = current_frame_pts.copy()
                    else:
                        if np.linalg.norm(current_anchor - pre_snap_anchor) > 4.5:
                            is_play_active = True
                            snap_frame_idx = processed_frames
                            print(f"  -> Snap detected at frame {processed_frames}!")

                if is_play_active and (processed_frames - snap_frame_idx) >= POST_SNAP_MAX_FRAMES:
                    break

            if is_play_active and len(raw_tracking_history) > 0:
                # 1. Base Saving Logic
                flat_matrix_dataset = {}
                for t_id, frame_data in raw_tracking_history.items():
                    player_timeline = [frame_data[f] if f in frame_data else (None, None) for f in range(POST_SNAP_MAX_FRAMES)]
                    flat_matrix_dataset[str(t_id)] = {"role": 2, "timeline": player_timeline}

                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("INSERT OR REPLACE INTO play_trajectories (clip_id, pre_snap_coords, post_snap_matrix) VALUES (?, ?, ?)", 
                               (clip_id, json.dumps(pre_snap_state), json.dumps(flat_matrix_dataset)))
                conn.commit()
                conn.close()

                # 2. Advanced Standardization Saving Logic (SFU-inspired coordinates)
                snap_pts = [raw_tracking_history[tid][0] for tid in raw_tracking_history if 0 in raw_tracking_history[tid]]
                if snap_pts:
                    center_x_0 = np.median([p[0] for p in snap_pts])
                    center_y_0 = np.median([p[1] for p in snap_pts])

                    conn_an = sqlite3.connect(ANALYTICS_DB_PATH)
                    cursor_an = conn_an.cursor()
                    
                    for t_id, frame_data in raw_tracking_history.items():
                        for f_idx, (cx, cy) in frame_data.items():
                            rel_x = cx - center_x_0
                            rel_y = cy - center_y_0
                            cursor_an.execute(
                                "INSERT INTO standardized_trajectories (clip_name, player_id, frame_idx, rel_x, rel_y) VALUES (?, ?, ?, ?, ?)",
                                (clip_id, int(t_id), int(f_idx), float(rel_x), float(rel_y))
                            )
                    conn_an.commit()
                    conn_an.close()

                print(f"  -> Stored normalized coordinate tracking metrics.")
            else:
                print(f"  -> Skipped: Incomplete play movement metrics.")

        except Exception as e:
            print(f"  -> Pipeline error: {e}")
        finally:
            if pipeline is not None: pipeline.stop()

    print("\n[SUCCESS] Extraction complete. True moving coordinate fields populated.")

if __name__ == "__main__":
    main()