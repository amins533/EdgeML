import os
import gc
import time
import sqlite3
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

from homography_engine import HomographyEngine
from data_base import DatabasePipeline
import heatmap # Import to override title and text behaviors dynamically

# --- PATH CONFIGURATIONS ---
BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos-ocr"
NEW_CARDS_DIR = BASE_DIR / "training_playcards"

PLAYER_MODEL_PATH = BASE_DIR / "models" / "spielererkennung" / "v6" / "weights" / "best.pt"
PITCH_MODEL_PATH = BASE_DIR / "models" / "felderkennung" / "tests" / "v3" / "weights" / "best.pt"

# --- SPEED OPTIMIZATION CONFIGS ---
FRAME_STRIDE = 2         # Process every 2nd frame (Skips half the compute overhead!)
HOMOGRAPHY_STEP = 5      # Re-infer pitch mapping coordinates every 5 active frames

def main():
    print("==================================================")
    print("🏈   FAST-TRACK CLEAN PLAYCARD GENERATION        🏈")
    print("==================================================")

    NEW_CARDS_DIR.mkdir(parents=True, exist_ok=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🔥 [GPU DEVICE] ACTIVE: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ [WARNING] Running on CPU!")

    video_files = sorted(list(VIDEOS_DIR.glob("*.mp4")))
    if not video_files:
        print(f"[ERROR] No video clips found in {VIDEOS_DIR}")
        return
    print(f"[INFO] Found {len(video_files)} video clips to process.")

    device_target = "cuda" if torch.cuda.is_available() else "cpu"
    player_model = YOLO(PLAYER_MODEL_PATH).to(device_target)
    
    for index, video_path in enumerate(video_files):
        clip_id = video_path.stem
        print(f"\n⚡ [{index + 1}/{len(video_files)}] Processing: {video_path.name}")
        
        tracked_players_cache = defaultdict(list)
        raw_frame_counter = 0
        saved_frame_counter = 0
        start_time = time.time()
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Could not open video file: {video_path.name}")
            continue

        try:
            homography_engine = HomographyEngine(str(PITCH_MODEL_PATH), device_target)
            H_matrix = None

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                raw_frame_counter += 1
                
                # OPTIMIZATION: STRIDE SKIP
                if raw_frame_counter % FRAME_STRIDE != 0:
                    continue

                saved_frame_counter += 1
                
                # OPTIMIZATION: CONDITIONAL HOMOGRAPHY INFERENCE
                should_infer = (saved_frame_counter % HOMOGRAPHY_STEP == 1) or (H_matrix is None)
                if should_infer:
                    H_matrix = homography_engine.process_frame(frame, None, run_inference=True)

                # Track players via low-memory stream mode on the GPU
                yolo_results = player_model.track(
                    frame, conf=0.25, persist=True, device=device_target, verbose=False, stream=True
                )
                
                for res in yolo_results:
                    if res.boxes is not None and res.boxes.id is not None:
                        boxes_data = res.boxes.xyxy.cpu().numpy()
                        ids_data = res.boxes.id.cpu().numpy().astype(int)
                        cls_data = res.boxes.cls.cpu().numpy().astype(int)

                        for box, t_id, cls_idx in zip(boxes_data, ids_data, cls_data):
                            if cls_idx == 1: # Ignore Defense
                                continue 

                            x_min, y_min, x_max, y_max = map(int, box)
                            feet_x = (x_min + x_max) / 2.0
                            feet_y = float(y_max)
                            
                            field_x, field_y = feet_x, feet_y

                            if H_matrix is not None:
                                feet_coord = np.array([[[feet_x, feet_y]]], dtype=np.float32)
                                mapped_point = cv2.perspectiveTransform(feet_coord, H_matrix)
                                field_x = float(mapped_point[0][0][0])
                                field_y = float(mapped_point[0][0][1])

                            player_role = "Unknown"
                            match cls_idx:
                                case 0: player_role = "Center"
                                case 2: player_role = "Attacker"
                                case 3: player_role = "Quarterback"
                                case 4: player_role = "Referee"

                            tracked_players_cache[t_id].append({
                                'frame': saved_frame_counter,
                                'coords': f"({field_x:.1f}, {field_y:.1f})",
                                'class': cls_idx,
                                'role': player_role
                            })

            cap.release()
            
            target_db_path = BASE_DIR / "tactical_system.db"
            memory_matrix_buffer = {}
            for t_id, frames_list in tracked_players_cache.items():
                if not frames_list: continue
                role = frames_list[0]['role']
                
                memory_matrix_buffer[t_id] = {"role": role}
                for f_data in frames_list:
                    memory_matrix_buffer[t_id][f_data['frame']] = f_data['coords']
            
            rows = []
            for player_id, frames_data in memory_matrix_buffer.items():
                row_entry = {"YOLO_Tracking_ID": player_id, "Role": frames_data.get("role", "Offense"), "Jersey_Number": "-"}
                for f in range(1, saved_frame_counter + 1):
                    row_entry[f"Frame_{f}"] = frames_data.get(f, "-")
                rows.append(row_entry)
            
            if rows:
                df = pd.DataFrame(rows)
                conn = sqlite3.connect(str(target_db_path))
                df.to_sql("offense_trajectory_matrix", conn, if_exists="replace", index=False)
                conn.close()

            # --- DYNAMIC MONKEY-PATCH: Disable Text Boxes AND Title ---
            original_ax_text = heatmap.plt.Axes.text
            original_ax_set_title = heatmap.plt.Axes.set_title
            
            def clean_text_wrapper(self, x, y, s, *args, **kwargs):
                # Completely bypass formation labels and metadata descriptors
                if any(k in str(s) for k in ["Formation:", "Recognized Play:", "Tactical Playcard"]):
                    return None
                return original_ax_text(self, x, y, s, *args, **kwargs)

            def clean_title_wrapper(self, label, *args, **kwargs):
                # Completely bypass the header text title at the top of the canvas layout
                return None

            heatmap.plt.Axes.text = clean_text_wrapper
            heatmap.plt.Axes.set_title = clean_title_wrapper

            # Run image creation
            heatmap.generate_numbered_yard_playcard(
                clip_name=clip_id, db_path=str(target_db_path), formation_text=""
            )

            # Restore original function behavior immediately
            heatmap.plt.Axes.text = original_ax_text
            heatmap.plt.Axes.set_title = original_ax_set_title

            # Relocate from cards/ to training_playcards/
            old_file_path = BASE_DIR / "cards" / f"{clip_id}_playcard.png"
            new_file_path = NEW_CARDS_DIR / f"{clip_id}_playcard.png"
            
            if old_file_path.exists():
                shutil.move(str(old_file_path), str(new_file_path))

        except Exception as e:
            print(f"❌ [ERROR] Issues processing clip {clip_id}: {e}")
        
        finally:
            if cap.isOpened():
                cap.release()
            
            tracked_players_cache.clear()
            elapsed_time = time.time() - start_time
            if saved_frame_counter > 0:
                print(f"🏁 Done with {clip_id} ({raw_frame_counter} raw frames -> parsed {saved_frame_counter} in {elapsed_time:.1f}s at {raw_frame_counter/elapsed_time:.1f} Eff-FPS)")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1.0)

    print(f"\n🎉 [SUCCESS] Finished! All clean playcard images are safely inside: '{NEW_CARDS_DIR}'")

if __name__ == "__main__":
    main()