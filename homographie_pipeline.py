import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
PLAYER_MODEL_PATH = BASE_DIR / "models" / "spielererkennung" / "v4" / "weights" / "best.pt"
PITCH_MODEL_PATH = BASE_DIR / "models" / "felderkennung" / "v9" / "weights" / "best.pt"

# --- CONFIGURACIÓN DE ALMACENAMIENTO DE ANÁLISIS ---
ANALYSIS_DIR = BASE_DIR / "homographie_analisis"
SNAPSHOTS_DIR = ANALYSIS_DIR / "video_snapshots"
TACTICAL_DIR = ANALYSIS_DIR / "tactical_maps"

# --- DIMENSIONES DEL MAPA TÁCTICO 2D (10px = 1 Yarda) ---
PITCH_WIDTH = 1200  
PITCH_HEIGHT = 533  

# --- COLORES GLOBALES DESDE MAIN.PY (Formato BGR para OpenCV) ---
CLASS_COLORS = {
    0: (36, 204, 255), # Center
    1: (94, 3, 252),   # Defense
    2: (219, 252, 3),  # Offense
    3: (255, 179, 0),  # Quarterback
    4: (46, 164, 255), # Referee
}

# --- MAPA GEOMÉTRICO ABSOLUTO DE LOS 18 KEYPOINTS DEL ESQUELETO ---
ABSOLUTE_PITCH_MAP = {
    0: [600, 0],             # 00_Yarda50_Top
    1: [600, PITCH_HEIGHT],  # 01_Yarda50_Bottom
    2: [500, 0],             # 02_Yarda40L_Top
    3: [500, PITCH_HEIGHT],  # 03_Yarda40L_Bottom
    4: [700, 0],             # 04_Yarda40R_Top
    5: [700, PITCH_HEIGHT],  # 05_Yarda40R_Bottom
    6: [400, 0],             # 06_Yarda30L_Top
    7: [400, PITCH_HEIGHT],  # 07_Yarda30L_Bottom
    8: [800, 0],             # 08_Yarda30R_Top
    9: [800, PITCH_HEIGHT],  # 09_Yarda30R_Bottom
    10: [300, 0],            # 10_Yarda20L_Top
    11: [300, PITCH_HEIGHT], # 11_Yarda20L_Bottom
    12: [900, 0],            # 12_Yarda20R_Top
    13: [900, PITCH_HEIGHT], # 13_Yarda20R_Bottom
    14: [200, 0],            # 14_Yarda10L_Top
    15: [200, PITCH_HEIGHT], # 15_Yarda10L_Bottom
    16: [1000, 0],           # 16_Yarda10R_Top
    17: [1000, PITCH_HEIGHT] # 17_Yarda10R_Bottom
}

# --- NOMBRES LEGIBLES PARA EL DEBUG VISUAL ---
KEYPOINT_NAMES = {
    0: "00_Y50_T", 1: "01_Y50_B",
    2: "02_Y40L_T", 3: "03_Y40L_B",
    4: "04_Y40R_T", 5: "05_Y40R_B",
    6: "06_Y30L_T", 7: "07_Y30L_B",
    8: "08_Y30R_T", 9: "09_Y30R_B",
    10: "10_Y20L_T", 11: "11_Y20L_B",
    12: "12_Y20R_T", 13: "13_Y20R_B",
    14: "14_Y10L_T", 15: "15_Y10L_B",
    16: "16_Y10R_T", 17: "17_Y10R_B"
}

def resolve_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return 0
    return "cpu"

def prepare_directories():
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    TACTICAL_DIR.mkdir(parents=True, exist_ok=True)

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
    results = pose_model.predict(frame, verbose=False, device=device, conf=0.307, save=False)
    
    if not results or len(results[0].keypoints.xy) == 0:
        return None
        
    kp = results[0].keypoints.xy[0].cpu().numpy()
    confidences = results[0].keypoints.conf[0].cpu().numpy()
    
    dst_points = [] 
    src_points = [] 
    
    for i, point in enumerate(kp):
        if i in ABSOLUTE_PITCH_MAP and i < len(confidences):
            # Umbral en 0.40 para forzar lecturas
            if confidences[i] > 0.307 and point[0] > 0 and point[1] > 0:
                dst_points.append([point[0], point[1]])
                src_points.append(ABSOLUTE_PITCH_MAP[i])
                
                if frame_disp is not None:
                    kp_name = KEYPOINT_NAMES.get(i, f"ID:{i}")
                    cv2.circle(frame_disp, (int(point[0]), int(point[1])), 6, (255, 255, 0), -1)
                    cv2.putText(frame_disp, kp_name, (int(point[0])+8, int(point[1])-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
    if len(dst_points) < 4:
        return None
        
    dst_points = np.array(dst_points, dtype=np.float32)
    src_points = np.array(src_points, dtype=np.float32)
    
    M, _ = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 4.0)
    return M

def get_player_feet(box):
    x_min, y_min, x_max, y_max = map(int, box)
    return np.array([[[ (x_min + x_max) / 2.0, float(y_max) ]]], dtype=np.float32)

def process_video(video_path, player_model, pitch_model, device):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el archivo de video: {video_path.name}")
        return 'eof'

    print(f"\n[PIPELINE] Procesando: {video_path.name} | Controles: [n] Siguiente | [p] Anterior | [q] Salir")
    
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return 'eof'
        
    H_matrix = calculate_homography_dynamic(first_frame, pitch_model, device)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Fin natural del video, avisa que debe saltar al siguiente
            cap.release()
            return 'eof'
            
        frame_index += 1
        frame_disp = frame.copy()
        minimap = draw_2d_pitch()

        # Recálculo de homografía frame a frame
        new_H_matrix = calculate_homography_dynamic(frame, pitch_model, device, frame_disp)
        if new_H_matrix is not None:
            H_matrix = new_H_matrix

        player_results = player_model.predict(frame, conf=0.479, verbose=False, device=device)

        if player_results and player_results[0].boxes is not None:
            boxes = player_results[0].boxes.xyxy.cpu().numpy()
            clases = player_results[0].boxes.cls.cpu().numpy()
            
            for box, cls in zip(boxes, clases):
                cls_id = int(cls)
                color = CLASS_COLORS.get(cls_id, (0, 255, 255))
                x1, y1, x2, y2 = map(int, box)
                
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), color, 2)
                
                if H_matrix is not None:
                    feet_coord = get_player_feet(box)
                    mapped_point = cv2.perspectiveTransform(feet_coord, H_matrix)
                    map_x, map_y = int(mapped_point[0][0][0]), int(mapped_point[0][0][1])
                    
                    if 0 <= map_x < PITCH_WIDTH and 0 <= map_y < PITCH_HEIGHT:
                        cv2.circle(minimap, (map_x, map_y), 8, color, -1)
                        cv2.circle(minimap, (map_x, map_y), 8, (255, 255, 255), 1)

        minimap_resized = cv2.resize(minimap, (600, 266))
        frame_resized = cv2.resize(frame_disp, (1280, 720))
        
        cv2.imshow("Main Video Stream - Production View", frame_resized)
        cv2.imshow("Tactical 2D Ground Truth Map", minimap_resized)

        key = cv2.waitKey(1) & 0xFF
        
        # --- SISTEMA DE NAVEGACIÓN POR TECLADO ---
        if key == ord(' '):
            video_stem = video_path.stem
            snapshot_filename = f"ss_{video_stem}_frame_{frame_index:06d}.jpg"
            map_filename = f"map_{video_stem}_frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(SNAPSHOTS_DIR / snapshot_filename), frame_disp)
            cv2.imwrite(str(TACTICAL_DIR / map_filename), minimap)
            print(f"\033[38;5;82m[SNAPSHOT SAVED] Guardado pre-snap en {snapshot_filename} y {map_filename}\033[0m")
            
        elif key == ord('q'):
            print("[PIPELINE] Video abortado por el usuario.")
            cap.release()
            return 'q' # Quit
            
        elif key == ord('n'):
            print("[NAVEGACIÓN] Saltando al siguiente video...")
            cap.release()
            return 'n' # Next
            
        elif key == ord('p'):
            print("[NAVEGACIÓN] Retrocediendo al video anterior...")
            cap.release()
            return 'p' # Previous

def main():
    prepare_directories()
    device = resolve_device()
    print(f"Initializing Homography Pipeline on engine target: {device}")
    
    player_model = YOLO(str(PLAYER_MODEL_PATH))
    pitch_model = YOLO(str(PITCH_MODEL_PATH))
    
    if not VIDEOS_DIR.exists():
        print(f"[ERROR] La carpeta de origen de videos no existe en: {VIDEOS_DIR}")
        return
        
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(list(VIDEOS_DIR.glob(ext)))
        
    if not all_videos:
        print(f"No se encontraron archivos de video compatibles en {VIDEOS_DIR}")
        return
        
    print(f"Se encontraron {len(all_videos)} videos listos para procesamiento en lote.")
    
    # --- CICLO DE NAVEGACIÓN BASADO EN ÍNDICE ---
    video_index = 0
    while 0 <= video_index < len(all_videos):
        video_file = all_videos[video_index]
        action = process_video(video_file, player_model, pitch_model, device)
        
        if action == 'q':
            break # Aborta todo
        elif action == 'p':
            video_index = max(0, video_index - 1) # Retrocede sin salir del rango
        elif action == 'n' or action == 'eof':
            video_index += 1 # Avanza de video
            
    cv2.destroyAllWindows()
    print("\n[FINISHED] Procesamiento de la lista de videos completado.")

if __name__ == "__main__":
    main()