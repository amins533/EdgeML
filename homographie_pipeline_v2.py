import cv2
import torch
import numpy as np
import sqlite3
import json
from pathlib import Path
from ultralytics import YOLO

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = Path("/home/amin/infprojekt/gruppe1/videos-ocr")
PLAYER_MODEL_PATH = BASE_DIR / "models" / "spielererkennung" / "v4" / "weights" / "best.pt"
PITCH_MODEL_PATH = BASE_DIR / "models" / "felderkennung" / "v10" / "weights" / "best.pt"

# --- STORAGE CONFIGURATION ---
ANALYSIS_DIR = BASE_DIR / "homographie_analisis"
SNAPSHOTS_DIR = ANALYSIS_DIR / "video_snapshots"
TACTICAL_DIR = ANALYSIS_DIR / "tactical_maps"
HARD_FRAMES_DIR = ANALYSIS_DIR / "hard_frames"
DB_PATH = BASE_DIR / "tactical_system.db"



# --- 2D TACTICAL MAP DIMENSIONS (10px = 1 Yard) ---
PITCH_WIDTH = 1200  
PITCH_HEIGHT = 533  

# --- GLOBAL COLORS (BGR Format) ---
CLASS_COLORS = {
    0: (36, 204, 255), # Center
    1: (94, 3, 252),   # Defense
    2: (219, 252, 3),  # Offense
    3: (255, 179, 0),  # Quarterback
    4: (46, 164, 255), # Referee
}

# --- SKELETON ABSOLUTE PITCH MAP ---
ABSOLUTE_PITCH_MAP = {
    0: [600, 0], 1: [600, PITCH_HEIGHT],
    2: [500, 0], 3: [500, PITCH_HEIGHT],
    4: [700, 0], 5: [700, PITCH_HEIGHT],
    6: [400, 0], 7: [400, PITCH_HEIGHT],
    8: [800, 0], 9: [800, PITCH_HEIGHT],
    10: [300, 0], 11: [300, PITCH_HEIGHT],
    12: [900, 0], 13: [900, PITCH_HEIGHT],
    14: [200, 0], 15: [200, PITCH_HEIGHT],
    16: [1000, 0], 17: [1000, PITCH_HEIGHT]
}

KEYPOINT_NAMES = {
    0: "00_Y50_T", 1: "01_Y50_B", 2: "02_Y40L_T", 3: "03_Y40L_B",
    4: "04_Y40R_T", 5: "05_Y40R_B", 6: "06_Y30L_T", 7: "07_Y30L_B",
    8: "08_Y30R_T", 9: "09_Y30R_B", 10: "10_Y20L_T", 11: "11_Y20L_B",
    12: "12_Y20R_T", 13: "13_Y20R_B", 14: "14_Y10L_T", 15: "15_Y10L_B",
    16: "16_Y10R_T", 17: "17_Y10R_B"
}

def resolve_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return 0
    return "cpu"

def prepare_directories():
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    TACTICAL_DIR.mkdir(parents=True, exist_ok=True)
    HARD_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

def init_database():
    """Initializes the SQLite Database for logging raw tracking coordinates."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracking_frames (
            clip_id TEXT,
            frame_idx INTEGER,
            offense_coords TEXT,
            defense_coords TEXT,
            center_coords TEXT,
            qb_coords TEXT,
            PRIMARY KEY (clip_id, frame_idx)
        )
    """)
    conn.commit()
    return conn

def draw_2d_pitch():
    pitch = np.zeros((PITCH_HEIGHT, PITCH_WIDTH, 3), dtype=np.uint8)
    pitch[:] = (40, 150, 40) 
    cv2.line(pitch, (100, 0), (100, PITCH_HEIGHT), (255, 255, 255), 2)
    cv2.line(pitch, (1100, 0), (1100, PITCH_HEIGHT), (255, 255, 255), 2)
    for x_coord in range(200, 1100, 100):
        cv2.line(pitch, (x_coord, 0), (x_coord, PITCH_HEIGHT), (220, 220, 220), 1)
    cv2.line(pitch, (600, 0), (600, PITCH_HEIGHT), (255, 255, 255), 3)
    return pitch

def calculate_homography_dynamic(frame, pose_model, device, frame_disp=None):
    results = pose_model.predict(frame, verbose=False, device=device, conf=0.402, save=False)
    if not results or len(results[0].keypoints.xy) == 0: return None
        
    kp = results[0].keypoints.xy[0].cpu().numpy()
    confidences = results[0].keypoints.conf[0].cpu().numpy()
    
    top_candidates, bottom_candidates = [], []
    for i, point in enumerate(kp):
        if i in ABSOLUTE_PITCH_MAP and i < len(confidences):
            if confidences[i] > 0.402 and point[0] > 0 and point[1] > 0:
                if i % 2 == 0: top_candidates.append({'id': i, 'pt': point})
                else: bottom_candidates.append({'id': i, 'pt': point})
                    
    if len(top_candidates) == 0 or len(bottom_candidates) == 0: return None
        
    median_top_y = np.median([c['pt'][1] for c in top_candidates])
    median_bottom_y = np.median([c['pt'][1] for c in bottom_candidates])
    y_ecuador = (median_top_y + median_bottom_y) / 2.0
    
    valid_points = []
    rejected_points = 0

    for c in top_candidates:
        if c['pt'][1] < y_ecuador: valid_points.append(c)
        else:
            rejected_points += 1
            if frame_disp is not None: cv2.circle(frame_disp, (int(c['pt'][0]), int(c['pt'][1])), 8, (0, 0, 255), -1) 
            
    for c in bottom_candidates:
        if c['pt'][1] > y_ecuador: valid_points.append(c)
        else:
            rejected_points += 1
            if frame_disp is not None: cv2.circle(frame_disp, (int(c['pt'][0]), int(c['pt'][1])), 8, (0, 0, 255), -1)
            
    top_count = sum(1 for c in valid_points if c['id'] % 2 == 0)
    bottom_count = sum(1 for c in valid_points if c['id'] % 2 != 0)
    if top_count < 2 or bottom_count < 2: return None
        
    dst_points, src_points = [], []
    for c in valid_points:
        dst_points.append([c['pt'][0], c['pt'][1]])
        src_points.append(ABSOLUTE_PITCH_MAP[c['id']])
        if frame_disp is not None:
            cv2.circle(frame_disp, (int(c['pt'][0]), int(c['pt'][1])), 6, (0, 255, 0), -1)
            cv2.putText(frame_disp, KEYPOINT_NAMES.get(c['id'], ""), (int(c['pt'][0])+8, int(c['pt'][1])-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if len(valid_points) < 4:
        return None

    M, mask = cv2.findHomography(np.array(dst_points, dtype=np.float32), np.array(src_points, dtype=np.float32), cv2.RANSAC, 10.0)
    if M is None or M.shape !=(3,3):
        return None
    if mask is None:
        inlier_ratio = 0
    else:
        inlier_ratio = float(np.sum(mask)) /  len(mask)
    return{
        "H":M,
        "valid_points":len(valid_points),
        "rejected_points": rejected_points,
        "inlier_ratio": inlier_ratio
    }
def get_player_feet(box):
    x_min, y_min, x_max, y_max = map(int, box)
    return np.array([[[ (x_min + x_max) / 2.0, float(y_max) ]]], dtype=np.float32)

def process_video(video_path, player_model, pitch_model, device, db_conn):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return 'eof'

    total_frames = int (cap.get(cv2.CAP_PROP_FRAME_COUNT))

    one_third = total_frames // 3
    two_thirds = (total_frames // 3) * 2

    saved_beginning = False
    saved_middle = False
    saved_end = False

    print(f"\n[PIPELINE] Processing: {video_path.name} | Controls: [n] Next | [p] Prev | [q] Quit")
    ret, first_frame = cap.read()
    if not ret: return 'eof'
        
    first_info = calculate_homography_dynamic(first_frame, pitch_model, device)
    if first_info is None:
        H_matrix = None
    else:
        H_matrix = first_info["H"]

    EMA_ALPHA = 0.85 
    frame_index = 0
    clip_id = video_path.stem # e.g., "wide-clip058"
    hard_frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame_index += 1
        frame_disp = frame.copy()
        minimap = draw_2d_pitch()

        homography_info = calculate_homography_dynamic(frame, pitch_model, device, frame_disp)

        if homography_info is not None:
            new_H_matrix = homography_info["H"]
            if H_matrix is None:
                H_matrix = new_H_matrix
            else:
                H_matrix = (EMA_ALPHA * H_matrix + (1.0 - EMA_ALPHA) * new_H_matrix)
        

        is_bad_frame = (homography_info is None or 
                        homography_info["valid_points"] < 6 or 
                        homography_info["rejected_points"] > 0 or 
                        homography_info["inlier_ratio"] < 0.75)
                        
        if is_bad_frame:
            # ZONE 1: Beginning of the clip (First 33%)
            if frame_index <= one_third and not saved_beginning:
                cv2.imwrite(str(HARD_FRAMES_DIR / f"{clip_id}_{frame_index:06d}_beg.jpg"), frame)
                saved_beginning = True
                print(f"   -> [SAVED] Bad frame caught in BEGINNING zone (Frame {frame_index})")

            elif one_third < frame_index <= two_thirds and not saved_middle:
                cv2.imwrite(str(HARD_FRAMES_DIR / f"{clip_id}_{frame_index:06d}_mid.jpg"), frame)
                saved_middle = True
                print(f"   -> [SAVED] Bad frame caught in MIDDLE zone (Frame {frame_index})")

            elif frame_index > two_thirds and not saved_end:
                cv2.imwrite(str(HARD_FRAMES_DIR / f"{clip_id}_{frame_index:06d}_end.jpg"), frame)
                saved_end = True
                print(f"   -> [SAVED] Bad frame caught in END zone (Frame {frame_index})")
       
            
        #     if(homography_info["valid_points"] <6 or homography_info["rejected_points"]>0 or homography_info["inlier_ratio"]<0.75):
        #         if frame_index % 100 == 0:
        #             cv2.imwrite(str(HARD_FRAMES_DIR/f"{clip_id}_{frame_index:06d}.jpg"),frame,)

        player_results = player_model.predict(frame, conf=0.479, verbose=False, device=device)
        
        # Lists to store 2D tactical coordinates for the database
        off_pts, def_pts, center_pts, qb_pts = [], [], [], []

        if player_results and player_results[0].boxes is not None:
            boxes = player_results[0].boxes.xyxy.cpu().numpy()
            clases = player_results[0].boxes.cls.cpu().numpy()
            
            for box, cls in zip(boxes, clases):
                cls_id = int(cls)
                color = CLASS_COLORS.get(cls_id, (0, 255, 255))
                cv2.rectangle(frame_disp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                
                if H_matrix is not None:

                    mapped_point = cv2.perspectiveTransform(
                        get_player_feet(box),
                        H_matrix
                    )

                    if not np.isfinite(mapped_point).all():
                        continue

                    map_x = int(mapped_point[0, 0, 0])
                    map_y = int(mapped_point[0, 0, 1])

                    if 0 <= map_x < PITCH_WIDTH and 0 <= map_y < PITCH_HEIGHT:

                        cv2.circle(minimap, (map_x, map_y), 8, color, -1)
                        cv2.circle(minimap, (map_x, map_y), 8, (255, 255, 255), 1)

                        pt = [float(map_x), float(map_y)]

                        if cls_id == 2:
                            off_pts.append(pt)
                        elif cls_id == 1:
                            def_pts.append(pt)
                        elif cls_id == 0:
                            center_pts.append(pt)
                        elif cls_id == 3:
                            qb_pts.append(pt)

        # Log frame data to SQLite if homography was valid
        if H_matrix is not None:
            cursor = db_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tracking_frames 
                (clip_id, frame_idx, offense_coords, defense_coords, center_coords, qb_coords)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (clip_id, frame_index, json.dumps(off_pts), json.dumps(def_pts), json.dumps(center_pts), json.dumps(qb_pts)))
            db_conn.commit()

        cv2.imshow("Main Video Stream", cv2.resize(frame_disp, (1280, 720)))
        cv2.imshow("Tactical 2D Map", cv2.resize(minimap, (600, 266)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.imwrite(str(SNAPSHOTS_DIR / f"ss_{clip_id}_f{frame_index:06d}.jpg"), frame_disp)
            cv2.imwrite(str(TACTICAL_DIR / f"map_{clip_id}_f{frame_index:06d}.jpg"), minimap)
            print(f"\033[38;5;82m[SNAPSHOT] Saved frame {frame_index}\033[0m")
        elif key == ord('q'): 
            cap.release() 
            return 'q'
        elif key == ord('n'): 
            cap.release() 
            return 'n'
        elif key == ord('p'): 
            cap.release() 
            return 'p'
        
    cap.release()
    return 'eof'

def main():
    prepare_directories()
    db_conn = init_database()
    device = resolve_device()
    print(f"Initializing Homography Pipeline on engine target: {device}")
    
    player_model = YOLO(str(PLAYER_MODEL_PATH))
    pitch_model = YOLO(str(PITCH_MODEL_PATH))
    
    if not VIDEOS_DIR.exists():
        print(f"Video directory not found") 
        return

    all_videos = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
        all_videos.extend(list(VIDEOS_DIR.glob(ext)))

    if len (all_videos)== 0:
        print("No videos found.")
        return

    print(f"Found {len(all_videos)} videos.")
        
    video_index = 0
    while 0 <= video_index < len(all_videos):
        print(f"\n{video_index +1 }/{len(all_videos)}] {all_videos[video_index].name}")

        action = process_video(all_videos[video_index], player_model, pitch_model, device, db_conn)
        if action == 'q': break
        elif action == 'p': video_index = max(0, video_index - 1)
        elif action in ('n', 'eof'): video_index += 1
            
    cv2.destroyAllWindows()
    db_conn.close()
    print("\n[FINISHED] Process complete.")

if __name__ == "__main__":
    main()