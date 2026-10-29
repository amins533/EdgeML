import argparse
import platform
import random
import time
import json
from collections import deque, defaultdict
from pathlib import Path
from heatmap import generate_numbered_yard_playcard
from cardbrowser import open_viewer

import cv2
import torch
import heatmap
import numpy as np
import pandas as pd
import xgboost as xgb

from ultralytics import YOLO
from data_base import DatabasePipeline
from data_class import GameData, PlayData, PlayerData, RouteData, TrackingPoint
from ocr_pipeline import OCRAnalysisPipeline
from video_pipeline import VideoPipeLine
from homography_engine import HomographyEngine

# --- CONFIGURATION & PATHS ---

CLASS_COLORS = {
    0: (36, 204, 255), # Center
    1: (94, 3, 252),   # Defense
    2: (219, 252, 3),  # Offense
    3: (255, 179, 0),  # Quarterback
    4: (46, 164, 255), # Referee
}

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
ANALYSIS_DIR = BASE_DIR / "homographie_analisis"
SNAPSHOTS_DIR = ANALYSIS_DIR / "video_snapshots"
TACTICAL_DIR = ANALYSIS_DIR / "tactical_maps"

PLAYER_MODEL_PATH = BASE_DIR / "models" / "spielererkennung" / "v6" / "weights" / "best.pt"
PITCH_MODEL_PATH = BASE_DIR / "models" / "felderkennung" / "tests" / "v3" / "weights" / "best.pt"
LEGIBILITY_MODEL_PATH = BASE_DIR / "models" / "lesbarkeit" / "v1" / "weights" / "best.pt"

HOMOGRAPHY_INTERVAL = 5

FORMATION_MODEL_PATH = BASE_DIR / "models" / "formationerkennung" / "v2" / "formation_xgboost_v1.json"
FORMATION_LABELS_PATH = BASE_DIR / "models" / "formationerkennung" / "v2" / "formation_labels_map.json"

def parse_args():
    """Parses command line arguments for execution modes."""
    parser = argparse.ArgumentParser(description="Tactical Edge Analysis Pipeline")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV GUI for headless Edge execution")
    return parser.parse_args()

def is_edge_device():
    """Detects if running on an ARM-based Linux device (like NVIDIA Jetson)."""
    return platform.machine() == 'aarch64' and platform.system() == 'Linux'

def get_optimized_model_path(base_pt_path, run_optimized):
    """Returns the .pt path, or auto-generates/loads a TensorRT .engine on Edge devices."""
    pt_path = Path(base_pt_path)
    if not run_optimized:
        return pt_path
    
    engine_path = pt_path.with_suffix('.engine')
    if engine_path.exists():
        print(f"\033[38;5;82m[TENSOR-RT] Loaded optimized engine for: {pt_path.name}\033[0m")
        return engine_path
    
    print(f"\n\033[38;5;208m[OPTIMIZATION] TensorRT engine not found for {pt_path.name}.\033[0m")
    print(f"[OPTIMIZATION] Exporting to .engine now (FP16). This will take ~5-10 mins but only happens once!")
    
    try:
        model = YOLO(str(pt_path))
        model.export(format='engine', half=True, device=0, workspace=4)
        print(f"\033[38;5;82m[OPTIMIZATION] Export successful: {engine_path.name}\033[0m\n")
        return engine_path
    except Exception as e:
        print(f"\033[38;5;196m[ERROR] Failed to export TensorRT engine: {e}\033[0m")
        print("[ERROR] Falling back to standard .pt model.")
        return pt_path

def prepare_directories():
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    TACTICAL_DIR.mkdir(parents=True, exist_ok=True)

def resolve_inference_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return 0
    return "cpu"

class PreSnapTemporalPooler:
    """Aggregates player detections across a rolling window to survive single-frame occlusions."""
    def __init__(self, max_frames=30):
        self.buffer = deque(maxlen=max_frames)

    def update(self, players_list):
        self.buffer.append(players_list)

    def get_pooled_offense(self):
        id_history = {}
        untracked_offense = []

        for frame_players in self.buffer:
            for p in frame_players:
                if p["team_side"] == "Offense" or p["player_role"] in ["Center", "Quarterback", "Attacker"]:
                    if p["field_x"] is None or p["field_y"] is None: continue
                    
                    tid = p["tracking_id"]
                    if tid is not None:
                        if tid not in id_history:
                            id_history[tid] = {"x": [], "y": [], "roles": []}
                        id_history[tid]["x"].append(p["field_x"])
                        id_history[tid]["y"].append(p["field_y"])
                        id_history[tid]["roles"].append(p["player_role"])
                    else:
                        untracked_offense.append(([p["field_x"], p["field_y"]], p["player_role"]))

        pooled_points = []
        for tid, data in id_history.items():
            if len(data["x"]) >= 3:
                med_x = float(np.median(data["x"]))
                med_y = float(np.median(data["y"]))
                role = max(set(data["roles"]), key=data["roles"].count)
                pooled_points.append({"pt": [med_x, med_y], "role": role, "tid": tid})

        if len(pooled_points) < 5 and untracked_offense:
            for pt, role in untracked_offense[-11:]:
                pooled_points.append({"pt": pt, "role": role, "tid": None})

        return pooled_points

class FormationVotingPool:
    """Evaluates and scores formations based on confidence AND actual real detected player count."""
    def __init__(self):
        self.candidates = defaultdict(float)
        self.best_label = "Scanning..."
        self.best_score = 0.0
        self.consistency_counter = defaultdict(int)
        self.locked_formation = None

    def add_prediction(self, label, confidence, real_player_count):
        if self.locked_formation:
            return

        if label == "Unknown" or confidence < 0.40: return

        completeness_ratio = min(real_player_count, 11) / 11.0
        composite_score = confidence * (0.5 + 0.5 * completeness_ratio)

        self.candidates[label] = (0.7 * self.candidates[label]) + (0.3 * composite_score)
        self.consistency_counter[label] += 1
        
        if self.consistency_counter[label] >= 15 and self.candidates[label] > 0.65:
            self.locked_formation = label
            self.best_label = label
            self.best_score = self.candidates[label]
            print(f"\033[38;5;82m[FORMATION] Formation state safely locked to prevent late pre-snap drift: {label}\033[0m")
            return

        if self.candidates[label] > self.best_score:
            self.best_score = self.candidates[label]
            self.best_label = label

    def get_current_leader(self):
        if self.locked_formation:
            return self.locked_formation, min(self.best_score, 0.99)
        return self.best_label, min(self.best_score, 0.99)

def predict_formation_xgboost(pooled_offense, def_pts, xgb_model, label_map):
    """Predicts formation resiliently and returns label, confidence, and real player count."""
    real_player_count = len(pooled_offense)
    if real_player_count < 6:
        return "Gathering positions...", 0.0, real_player_count

    center_pts = [p["pt"] for p in pooled_offense if p["role"] == "Center"]
    qb_pts = [p["pt"] for p in pooled_offense if p["role"] == "Quarterback"]
    all_off_coords = [p["pt"] for p in pooled_offense]

    x_all = np.array(all_off_coords, dtype=float)

    mean_def_x = np.mean([p[0] for p in def_pts]) if def_pts else 0
    mean_off_x = np.mean(x_all[:, 0]) if len(x_all) > 0 else 0

    if mean_def_x < mean_off_x and mean_def_x != 0:
        x_all[:, 0] = 1200.0 - x_all[:, 0]
        center_pts = [[1200.0 - p[0], p[1]] for p in center_pts]
        qb_pts = [[1200.0 - p[0], p[1]] for p in qb_pts]

    if center_pts:
        center_pt = np.array(center_pts[0])
    else:
        center_idx = np.argmax(x_all[:, 0])
        center_pt = x_all[center_idx]

    if qb_pts:
        qb_pt = np.array(qb_pts[0])
    else:
        fallback_qb = np.array([center_pt[0] - 15.0, center_pt[1]])
        dists = np.linalg.norm(x_all - fallback_qb, axis=1)
        qb_idx = np.argmin(dists)
        qb_pt = x_all[qb_idx]

    norm_pts = x_all - center_pt
    c_idx = np.argmin(np.linalg.norm(norm_pts, axis=1))
    norm_pts_no_c = np.delete(norm_pts, c_idx, axis=0)

    qb_norm = qb_pt - center_pt
    if len(norm_pts_no_c) > 0:
        q_idx = np.argmin(np.linalg.norm(norm_pts_no_c - qb_norm, axis=1))
        remaining_pts = np.delete(norm_pts_no_c, q_idx, axis=0)
    else:
        remaining_pts = norm_pts_no_c

    remaining_pts = remaining_pts[remaining_pts[:, 1].argsort()]

    ordered_p = np.full((9, 2), np.nan)
    num_avail = min(len(remaining_pts), 9)
    if num_avail > 0:
        ordered_p[:num_avail] = remaining_pts[:num_avail]

    vector = np.vstack(([norm_pts[c_idx]], [qb_norm], ordered_p)).flatten().reshape(1, -1)

    cols = ['C_x', 'C_y', 'QB_x', 'QB_y']
    for i in range(1, 10): cols.extend([f'P{i}_x', f'P{i}_y'])

    df_pred = pd.DataFrame(vector, columns=cols)
    probs = xgb_model.predict_proba(df_pred)[0]
    best_idx = np.argmax(probs)
    confidence = float(probs[best_idx])
    label = label_map.get(str(best_idx), "Unknown")

    return label, confidence, real_player_count

def draw_clean_hud(frame, formation_text, confidence, is_locked):
    """Draws a broadcast-style modern dark banner overlay at the top left."""
    h, w, _ = frame.shape
    text_display = f"{'LOCKED' if is_locked else 'PRE-SNAP'}: {formation_text}"
    if confidence > 0:
        text_display += f" ({confidence*100:.1f}%)"

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.75
    thickness = 2
    (txt_w, txt_h), baseline = cv2.getTextSize(text_display, font, font_scale, thickness)

    box_w = txt_w + 40
    box_h = txt_h + 26

    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + box_w, 15 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    accent_color = (0, 165, 255) if is_locked else (0, 255, 100)
    cv2.rectangle(frame, (15, 15), (21, 15 + box_h), accent_color, -1)

    text_x = 32
    text_y = 15 + txt_h + 10
    cv2.putText(frame, text_display, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(frame, text_display, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

def mock_route_recognition(play_id, player_list):
    routes = ["Fly", "slant", "out"]
    result = []
    for pid in player_list:
        result.append(RouteData(play_id=play_id, tracking_id=pid, route_type=random.choice(routes)))
    return result

def player_tracking(frame, model, tracker_config, device, shared_cache):
    results = model.track(frame, conf=0.377, persist=True, verbose=False, imgsz=1024, device=device, tracker=tracker_config)
    player_data = []

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return player_data

    boxes = results[0].boxes.xyxy.cpu().numpy()
    clss = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else [0] * len(boxes)
    track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [None] * len(boxes)

    for index, (box, cls, track_id) in enumerate(zip(boxes, clss, track_ids)):
        class_id = int(cls)
        track_id = int(track_id) if track_id is not None else None
        x_min, y_min, x_max, y_max = map(int, box)

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        jersey = None
        if track_id is not None and track_id in shared_cache:
            if shared_cache[track_id]["confidence"] >= 0.50:
                jersey = shared_cache[track_id]["jersey"]
            elif shared_cache[track_id].get("jersey") is not None:
                jersey = shared_cache[track_id].get("jersey")

        player_role, team_side = "Unknown", "Unknown"
        match class_id:
            case 0: player_role, team_side = "Center", "Offense"
            case 1: player_role, team_side = "Defenser", "Defense"
            case 2: player_role, team_side = "Attacker", "Offense"
            case 3: player_role, team_side = "Quarterback", "Offense"
            case 4: player_role, team_side = "Referee", "Referee"

        player_data.append({
            "tracking_id": track_id, "jersey_number": jersey, "player_role": player_role,
            "team_side": team_side, "index": index, "box": (x_min, y_min, x_max, y_max),
            "x": center_x, "y": center_y, "class_id": class_id,
            "field_x": None, "field_y": None
        })

    return player_data

def build_and_save_matrix(matrix_dict, db_path, total_frames, tracked_players_cache):
    if not matrix_dict: return
    rows = []
    for player_id, frames_data in matrix_dict.items():
        player_info = tracked_players_cache.get(player_id, {})
        jersey = player_info.get("jersey_number", None)
        
        row_entry = {
            "YOLO_Tracking_ID": player_id, 
            "Role": frames_data.get("role", "Offense"),
            "Jersey_Number": jersey if jersey is not None else "-" 
        }
        for f in range(1, total_frames + 1):
            coord = frames_data.get(f, "-")
            row_entry[f"Frame_{f}"] = coord
        rows.append(row_entry)
        
    import sqlite3
    df = pd.DataFrame(rows)
    conn = sqlite3.connect(str(db_path))
    df.to_sql("offense_trajectory_matrix", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[DB] Saved trajectory matrix (with Jersey Numbers) to DB.")

def setup_windows(video_filename):
    video_win_name = "RT Video Analyse - Pipeline Dashboard"

    # Combine WINDOW_AUTOSIZE with WINDOW_GUI_NORMAL to strip the Qt toolbar
    cv2.namedWindow(video_win_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    # Let the OS create the native window framework
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow(video_win_name, blank)
    cv2.waitKey(100)

    # Position it nicely on your screen
    cv2.moveWindow(video_win_name, 0, 0)
    cv2.waitKey(50)

    cv2.setWindowTitle(video_win_name, f"RT Video Analyse - {video_filename}")
    return video_win_name

def main():
    args = parse_args()
    is_headless = args.headless
    device_is_edge = is_edge_device()
    device = resolve_inference_device()

    prepare_directories()
    db = DatabasePipeline().start()

    if not VIDEOS_DIR.exists():
        print(f"[ERROR] Videos directory not found at {VIDEOS_DIR}")
        return

    video_files = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
        video_files.extend(list(VIDEOS_DIR.glob(ext)))

    if not video_files:
        print(f"[ERROR] No compatible video files found in {VIDEOS_DIR}")
        return

    print(f"Initializing Unified Models on target engine: {device}")
    if is_edge_device():
        print("[HARDWARE] ARM Edge Device detected. Enabling TensorRT optimization path.")
    
    active_player_model = get_optimized_model_path(PLAYER_MODEL_PATH, device_is_edge)
    active_pitch_model = get_optimized_model_path(PITCH_MODEL_PATH, device_is_edge)
    active_ocr_model = get_optimized_model_path(LEGIBILITY_MODEL_PATH, device_is_edge)

    if not active_player_model.exists(): raise FileNotFoundError(f"Player model not found: {active_player_model}")

    player_model = YOLO(str(active_player_model), task="detect")
    tracker_config_path = str(BASE_DIR / "custom_bytetrack.yaml")
    homography_engine = HomographyEngine(str(active_pitch_model), device)

    print("[ENGINE] Loading XGBoost Formation Model...")
    formation_model = xgb.XGBClassifier()
    formation_model.load_model(str(FORMATION_MODEL_PATH))

    with open(FORMATION_LABELS_PATH, "r") as f:
        formation_labels = json.load(f)

    video_index = 0
    play_id_counter = 1
    CARDS_DIR = Path("cards") if 'Path' in globals() else None
    
    if CARDS_DIR:
        if CARDS_DIR.exists():
            print(f"🧹 Clearing stale assets from global storage: '{CARDS_DIR}'...")
            import shutil
            shutil.rmtree(CARDS_DIR)
        CARDS_DIR.mkdir(parents=True, exist_ok=True)
    else:
        import os, shutil
        if os.path.exists("cards"):
            print("🧹 Clearing stale assets from global storage: 'cards/'...")
            shutil.rmtree("cards")
        os.makedirs("cards", exist_ok=True)
        
    while 0 <= video_index < len(video_files):
        video_path = video_files[video_index]
        video_filename = video_path.name
        clip_id = video_path.stem

        print(f"\n" + "="*60)
        print(f"[NAVIGATOR] Analyzing Clip: {video_filename} ({video_index + 1}/{len(video_files)})")
        if is_headless:
            print("[MODE] HEADLESS ACTIVE: GUI rendering is disabled. Processing at max speed.")
        else:
            print("Controls: [n] Next | [p] Prev | [Space] Screenshot | [q] Quit")
        print("="*60)

        shared_cache = {}
        ocr_pipeline = OCRAnalysisPipeline(
            base_dir=BASE_DIR, legibility_model_path=str(active_ocr_model),
            shared_cache=shared_cache
        ).start()

        pipeline = VideoPipeLine(video_path=str(video_path)).start()
        time.sleep(0.5)

        video_win_name = None
        if not is_headless:
            video_win_name = setup_windows(video_filename)
            if pipeline.more():
                warmup_frame = pipeline.read()
                if warmup_frame is not None:
                    cv2.imshow(video_win_name, cv2.resize(warmup_frame, (1280, 720)))
                    cv2.waitKey(30)
        game_id = 1
        current_play_id = None

        db.save_game(GameData(game_id=game_id, date=time.strftime("%Y-%m-%d"), video=video_filename))

        tracked_players_cache = {}
        processed_frames = 0
        memory_matrix_buffer = {}
        homography_engine.last_valid_H = None

        temporal_pooler = PreSnapTemporalPooler(max_frames=30)
        voting_pool = FormationVotingPool()

        current_formation = "Scanning..."
        current_conf = 0.0
        is_play_active = False
        pre_snap_core_anchor = None

        time_stats = {
            "homography": {"total": 0.0, "count": 0},
            "yolo_inference": {"total": 0.0, "count": 0},
            "formation_inference": {"total": 0.0, "count": 0},
            "db_tracking": {"total": 0.0, "count": 0},
            "ocr_submit": {"total": 0.0, "count": 0}
        }

        start_run_time = time.perf_counter()
        action = 'eof'

        while True:
            if not pipeline.more():
                time.sleep(0.01)
                if pipeline.stopped: break
                continue

            frame = pipeline.read()
            if frame is None: break
            processed_frames += 1
            frame_disp = frame.copy() if not is_headless else None

            # 1. Map Generation & Homography
            t_start_h = time.perf_counter()
            minimap = homography_engine.draw_2d_pitch()
            should_infer_homography = (processed_frames % HOMOGRAPHY_INTERVAL == 1) or (homography_engine.last_valid_H is None)
            
            current_H = homography_engine.process_frame(frame, frame_disp, run_inference=should_infer_homography)
            
            time_stats["homography"]["total"] += (time.perf_counter() - t_start_h)
            time_stats["homography"]["count"] += 1

            # 2. Player Tracking
            t_start = time.perf_counter()
            players = player_tracking(frame, player_model, tracker_config_path, device, shared_cache)
            time_stats["yolo_inference"]["total"] += (time.perf_counter() - t_start)
            time_stats["yolo_inference"]["count"] += 1

            tracking_batch = []
            players_needing_ocr = []
            def_pts = []

            for s in players:
                player_id = s["tracking_id"]
                if player_id is None: continue

                if player_id not in shared_cache or shared_cache[player_id]["confidence"] < 0.50:
                    players_needing_ocr.append(s)

                if player_id not in tracked_players_cache:
                    tracked_players_cache[player_id] = {
                        "jersey_number": s["jersey_number"], 
                        "player_role": s["player_role"], 
                        "team_side": s["team_side"]
                    }
                else:
                    if s["jersey_number"] is not None: 
                        tracked_players_cache[player_id]["jersey_number"] = s["jersey_number"]
                    if s["player_role"] in ["Center", "Quarterback"] or tracked_players_cache[player_id]["player_role"] == "Unknown":
                        tracked_players_cache[player_id]["player_role"] = s["player_role"]
                        tracked_players_cache[player_id]["team_side"] = s["team_side"]

                x_min, y_min, x_max, y_max = s["box"]
                feet_x = (x_min + x_max) / 2.0
                feet_y = float(y_max)
                field_x, field_y = None, None

                if current_H is not None:
                    feet_coord = np.array([[[feet_x, feet_y]]], dtype=np.float32)
                    mapped_point = cv2.perspectiveTransform(feet_coord, current_H)
                    field_x = float(mapped_point[0][0][0])
                    field_y = float(mapped_point[0][0][1])

                    if 0 <= int(field_x) < homography_engine.PITCH_WIDTH and 0 <= int(field_y) < homography_engine.PITCH_HEIGHT:
                        cv2.circle(minimap, (int(field_x), int(field_y)), 8, CLASS_COLORS.get(s["class_id"], (0,255,255)), -1)
                        s["field_x"] = field_x
                        s["field_y"] = field_y
                        if s["team_side"] == "Defense": def_pts.append([field_x, field_y])

                final_x = field_x if field_x is not None else feet_x
                final_y = field_y if field_y is not None else feet_y

                if s["team_side"] == "Offense" or s["player_role"] in ["Center", "Attacker", "Quarterback"]:
                    if player_id not in memory_matrix_buffer:
                        memory_matrix_buffer[player_id] = {"role": s["player_role"]}
                    memory_matrix_buffer[player_id][processed_frames] = f"({final_x:.1f}, {final_y:.1f})"

                if current_play_id is not None:
                    tracking_batch.append(TrackingPoint(
                        tracking_id=player_id, play_id=current_play_id, frame_id=processed_frames,
                        x_coord=final_x, y_coord=final_y
                    ))

                if not is_headless and frame_disp is not None:
                    label = f"#{s['jersey_number']}" if s['jersey_number'] else f"ID: {player_id}"
                    cv2.rectangle(frame_disp, (x_min, y_min), (x_max, y_max), CLASS_COLORS.get(s["class_id"], (0,255,255)), 3)
                    cv2.putText(frame_disp, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLASS_COLORS.get(s["class_id"], (0,255,255)), 3)

            # 4. XGBoost ML Formation Inference & Early Freezing
            t_start_form = time.perf_counter()
            temporal_pooler.update(players)
            pooled_offense = temporal_pooler.get_pooled_offense()

            if not is_play_active and processed_frames > 20 and len(pooled_offense) >= 6:
                core_players = [p["pt"] for p in pooled_offense if p["role"] in ["Center", "Quarterback"]]
                anchor_point = np.mean(core_players, axis=0) if core_players else np.median([p["pt"] for p in pooled_offense], axis=0)

                if pre_snap_core_anchor is None:
                    pre_snap_core_anchor = anchor_point
                else:
                    if np.linalg.norm(anchor_point - pre_snap_core_anchor) > 15.0:
                        is_play_active = True
                        print("\033[38;5;208m[TACTICAL] Snap Detected! Freezing Formation state.\033[0m")

            if not is_play_active and current_H is not None:
                raw_label, raw_conf, real_count = predict_formation_xgboost(
                    pooled_offense, def_pts, formation_model, formation_labels
                )
                voting_pool.add_prediction(raw_label, raw_conf, real_count)
                current_formation, current_conf = voting_pool.get_current_leader()

            time_stats["formation_inference"]["total"] += (time.perf_counter() - t_start_form)
            time_stats["formation_inference"]["count"] += 1

            current_play_id = play_id_counter

            if tracking_batch:
                t_start = time.perf_counter()
                db.save_tracking_batch(tracking_batch)
                time_stats["db_tracking"]["total"] += (time.perf_counter() - t_start)
                time_stats["db_tracking"]["count"] += 1

            if processed_frames % 5 == 0 and len(players_needing_ocr) > 0:
                t_start_ocr = time.perf_counter()
                ocr_pipeline.submit(processed_frames, frame, players_needing_ocr)
                time_stats["ocr_submit"]["total"] += (time.perf_counter() - t_start_ocr)
                time_stats["ocr_submit"]["count"] += 1

            if not is_headless:
                frame_disp = cv2.resize(frame_disp, (1280, 720))
                map_w, map_h = 270, 120
                mini_map_resized = cv2.resize(minimap, (map_w, map_h))
                cv2.rectangle(mini_map_resized, (0,0), (map_w-1, map_h-1), (100, 100, 100), 2)

                pad = 20
                y1, y2 = pad, pad + map_h
                x1, x2 = 1280 - map_w - pad, 1280 - pad
                frame_disp[y1:y2, x1:x2] = mini_map_resized

                draw_clean_hud(frame_disp, current_formation, current_conf, is_play_active or (voting_pool.locked_formation is not None))
                cv2.imshow(video_win_name, frame_disp)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    action = 'q'
                    break
                elif key == ord("n"):
                    action = 'n'
                    break
                elif key == ord("p"):
                    action = 'p'
                    break
                elif key == ord(" "):
                    snap_path = SNAPSHOTS_DIR / f"ss_{clip_id}_f{processed_frames:06d}.jpg"
                    map_path = TACTICAL_DIR / f"map_{clip_id}_f{processed_frames:06d}.jpg"
                    cv2.imwrite(str(snap_path), frame_disp)
                    cv2.imwrite(str(map_path), minimap)
                    print(f"\033[38;5;82m[SNAPSHOT] Saved successfully into analysis folders.\033[0m")
            else:
                action = 'continue'

        end_run_time = time.perf_counter()
        total_time = end_run_time - start_run_time

        print(f"\n[INFO] Analyzing routes run in {video_filename}...")
        end_routes = mock_route_recognition(current_play_id or 1, list(tracked_players_cache.keys()))
        for r in end_routes: 
            db.save_route(r)

        for track_id, data in tracked_players_cache.items():
            db.save_player(PlayerData(tracking_id=track_id, jersey_number=data["jersey_number"], player_role=data["player_role"], team_side=data["team_side"]))

        target_db_path = BASE_DIR / "tactical_system.db"
        build_and_save_matrix(memory_matrix_buffer, target_db_path, processed_frames, tracked_players_cache)
        
        pipeline.stop()
        ocr_stats = ocr_pipeline.stop()
        
        # --- PRESERVE WINDOW LOCATION BETWEEN CLIPS ---
        # We purposely do not issue cv2.destroyAllWindows() here anymore.
        # This prevents the window from bouncing randomly on your monitors.
        time.sleep(0.5)

        try:
            form_string = f"{current_formation}"
            if current_conf > 0:
                form_string += f" ({current_conf*100:.1f}%)"
            generate_numbered_yard_playcard(clip_name=clip_id, db_path=str(target_db_path), formation_text=form_string)
        except Exception as e:
            print(f"⚠️ Error generating playcard: {e}")

        def avg_ms(stats_dict, key):
            if key not in stats_dict or stats_dict[key]["count"] == 0: return 0.0
            return (stats_dict[key]["total"] / stats_dict[key]["count"]) * 1000

        print("\n" + "─"*60)
        print(f" 📊 CLIP REPORT: {video_filename} ".center(60))
        print("─"*60)
        print(f"▶️ Processed Frames:             {processed_frames}")
        print(f"▶️ Tracked Players:              {len(tracked_players_cache)}")
        print(f"▶️ Total OCR Scans Executed:     {ocr_stats['easyocr']['count'] if ocr_stats and 'easyocr' in ocr_stats else 0}")
        print(f"▶️ Overall FPS:                  {processed_frames / total_time if total_time > 0 else 0.0:.2f}")
        
        print("\n[PERFORMANCE] Synch. Main Thread:")
        print(f"  - Homography Engine:           {avg_ms(time_stats, 'homography'):.2f} ms/frame")
        print(f"  - YOLO Tracking:               {avg_ms(time_stats, 'yolo_inference'):.2f} ms/frame")
        print(f"  - XGBoost Formation ML:        {avg_ms(time_stats, 'formation_inference'):.2f} ms/frame")
        print(f"  - SQLite Tracking Inserts:     {avg_ms(time_stats, 'db_tracking'):.2f} ms/batch")
        print("═"*60 + "\n")

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        time.sleep(1.0)

        if action == 'q':
            print("[EXIT] User requested shutdown. Exiting cleanly...")
            break
        elif action == 'p':
            video_index = max(0, video_index - 1)
        else:
            video_index += 1
            play_id_counter += 1

    # Final Teardown of window references after entire script processes fully
    if not is_headless:
        cv2.destroyAllWindows()

    db.stop()
    print("[SYSTEM] All queued clips processed. Pipeline shutdown complete.")
    open_viewer(target_folder="cards")

if __name__ == "__main__":
    main()