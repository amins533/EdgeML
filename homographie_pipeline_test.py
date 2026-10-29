import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "videos" / "wide-clip-002.mp4"
PLAYER_MODEL_PATH = BASE_DIR / "models" / "spielererkennung" / "v4" / "weights" / "best.pt"
# Make sure this points to your newly fine-tuned pitch model
PITCH_MODEL_PATH = BASE_DIR / "models" / "felderkennung" / "v1" / "weights" / "best.pt" 

# --- 2D AMERICAN FOOTBALL PITCH DEFINITION (MOCKUP) ---
# We define a tactical top-down view of the pitch.
# A standard field is 120 yards x 53.3 yards. We multiply by 10 for pixel resolution.
PITCH_WIDTH = 1200  
PITCH_HEIGHT = 533  

# --- 2D REFERENCE POINTS (SOURCE MAP) ---
# These are the exact (x, y) coordinates on our 2D tactical minimap.
# You MUST map these to the exact 34 keypoints your YOLO-Pose model detects.
# SIMPLIFIED EXAMPLE: We assume the first 4 keypoints detected by your model 
# correspond to the 4 main corners of the playing field (Endzone to Endzone).
SRC_POINTS_2D = np.array([
    [100, 0],              # Index 0: Top-Left (Left goal line, top sideline)
    [1100, 0],             # Index 1: Top-Right (Right goal line, top sideline)
    [1100, PITCH_HEIGHT],  # Index 2: Bottom-Right (Right goal line, bottom sideline)
    [100, PITCH_HEIGHT]    # Index 3: Bottom-Left (Left goal line, bottom sideline)
], dtype=np.float32)

def resolve_device():
    """Determines the best available hardware accelerator."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
        return "mps" # Apple Silicon
    if torch.cuda.is_available(): 
        return 0     # NVIDIA GPU
    return "cpu"

def draw_2d_pitch():
    """Creates a blank green image representing the 2D tactical pitch."""
    pitch = np.zeros((PITCH_HEIGHT, PITCH_WIDTH, 3), dtype=np.uint8)
    pitch[:] = (40, 150, 40) # Dark green background
    
    # Draw basic field lines (White)
    cv2.line(pitch, (100, 0), (100, PITCH_HEIGHT), (255,255,255), 2)   # Left Goal Line
    cv2.line(pitch, (1100, 0), (1100, PITCH_HEIGHT), (255,255,255), 2) # Right Goal Line
    cv2.line(pitch, (600, 0), (600, PITCH_HEIGHT), (255,255,255), 2)   # 50-Yard Line
    
    return pitch

def calculate_homography(frame, pose_model, device):
    """
    Predicts pitch keypoints using YOLO-Pose and calculates the Homography Matrix
    to transform camera coordinates into 2D tactical coordinates.
    """
    # 1. Run inference to find field keypoints
    results = pose_model.predict(frame, verbose=False, device=device)
    
    # Safety Check: Did the model return anything at all?
    if not results or results[0].keypoints is None:
        return None
        
    # Safety Check: Did the model detect at least one object (the pitch)?
    if len(results[0].keypoints.xy) == 0:
        return None
        
    # Extract the (x, y) coordinates of the detected keypoints
    kp = results[0].keypoints.xy[0].cpu().numpy()
    
    # We need at least 4 points to calculate a homography matrix
    if len(kp) < 4:
        return None
        
    # --- KEYPOINT EXTRACTION LOGIC ---
    dst_points = [] # Points on the camera frame
    src_points = [] # Corresponding points on our 2D map
    
    # Loop through the first 4 keypoints (as defined in our simplified example)
    for i in range(4): 
        # Ensure the index exists and the point was actually detected (x > 0, y > 0)
        if i < len(kp) and kp[i][0] > 0 and kp[i][1] > 0:
            dst_points.append([kp[i][0], kp[i][1]])
            src_points.append(SRC_POINTS_2D[i])
            
    # Final check: Did we find 4 valid, visible points?
    if len(dst_points) < 4:
        return None 
        
    # Convert lists to NumPy arrays required by OpenCV
    dst_points = np.array(dst_points, dtype=np.float32)
    src_points = np.array(src_points, dtype=np.float32)
    
    # 2. Calculate the Transformation Matrix (M)
    # This matrix "warps" the perspective from the camera angle to a flat top-down view.
    M, mask = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 5.0)
    return M

def get_player_feet(box):
    """Returns the bottom-center coordinate of a bounding box (where the feet touch the ground)."""
    x_min, y_min, x_max, y_max = map(int, box)
    # x = middle of the box, y = bottom of the box
    return np.array([[(x_min + x_max) / 2.0, y_max]], dtype=np.float32)

def main():
    device = resolve_device()
    
    # Load Models
    player_model = YOLO(str(PLAYER_MODEL_PATH))
    pitch_model = YOLO(str(PITCH_MODEL_PATH))
    
    # Open Video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened(): 
        raise FileNotFoundError("Could not open video file.")

    print(f"Running Homography Pipeline on {device}...")

    # Read the very first frame to establish an initial matrix
    ret, first_frame = cap.read()
    if not ret: return
    
    print("Calculating initial homography matrix...")
    H_matrix = calculate_homography(first_frame, pitch_model, device)
    
    if H_matrix is None:
        print("WARNING: Could not calculate initial homography. Check keypoint detections.")

    # --- MAIN VIDEO LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_disp = frame.copy()
        minimap = draw_2d_pitch()

        # 1. Recalculate Homography (Camera Motion Compensation)
        # We attempt to find the field lines every frame. 
        # If successful, we update the matrix. If players block the lines, we keep the old matrix.
        new_H_matrix = calculate_homography(frame, pitch_model, device)
        if new_H_matrix is not None:
            H_matrix = new_H_matrix

        # 2. Detect Players
        player_results = player_model.predict(frame, conf=0.479, verbose=False, device=device)
        
        if player_results and player_results[0].boxes is not None:
            boxes = player_results[0].boxes.xyxy.cpu().numpy()
            clases = player_results[0].boxes.cls.cpu().numpy()
            
            # Loop through every detected player
            for box, cls in zip(boxes, clases):
                cls_id = int(cls)
                # Color Coding: Red for Defense (1), Green for Offense/Others
                color = (0, 0, 255) if cls_id == 1 else (0, 255, 0) 
                
                # Draw bounding box on the camera view
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), color, 2)
                
                # 3. Project Player to 2D Minimap
                if H_matrix is not None:
                    # Get feet coordinates
                    feet_coord = get_player_feet(box)
                    feet_coord = np.array([feet_coord]) 
                    
                    # Apply Homography: Transform camera (x,y) to tactical (x',y')
                    mapped_point = cv2.perspectiveTransform(feet_coord, H_matrix)
                    map_x, map_y = int(mapped_point[0][0][0]), int(mapped_point[0][0][1])
                    
                    # If the mapped point lands inside the pitch boundaries, draw it
                    if 0 <= map_x < PITCH_WIDTH and 0 <= map_y < PITCH_HEIGHT:
                        cv2.circle(minimap, (map_x, map_y), 10, color, -1)             # Solid dot
                        cv2.circle(minimap, (map_x, map_y), 10, (255,255,255), 2)      # White border

        # 4. Display Results
        # Resize minimap to fit on a normal screen
        minimap_resized = cv2.resize(minimap, (int(PITCH_WIDTH/2), int(PITCH_HEIGHT/2)))
        frame_resized = cv2.resize(frame_disp, (1280, 720))
        
        cv2.imshow("Camera View", frame_resized)
        cv2.imshow("Tactical 2D View", minimap_resized)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()