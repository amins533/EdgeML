import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def get_zentrum_koordinaten(box):
    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
    return int((x_min + x_max) / 2), int((y_min + y_max) / 2)

def generate_synthetic_dataset(dataset_dir, detector_path, output_dir):
    print("Lade YOLO Detector (Raw Perspective)... 🚀")
    model = YOLO(detector_path)
    
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    formation_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    total_bilder = 0

    for folder in formation_folders:
        label = folder.name
        out_folder = output_path / label
        out_folder.mkdir(parents=True, exist_ok=True)
        
        for image_file in folder.glob("*.jpg"):
            results = model.predict(str(image_file), conf=0.50, verbose=False)
            boxes = results[0].boxes
            
            offense_spieler = []
            center_spieler = None
            qb_spieler = None
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in [0, 2, 3]:
                    cx, cy = get_zentrum_koordinaten(box)
                    point = {'x': cx, 'y': cy}
                    if cls_id == 0: center_spieler = point
                    elif cls_id == 3: qb_spieler = point
                    else: offense_spieler.append(point)
            
            valid_spieler = []
            if center_spieler: valid_spieler.append(center_spieler)
            if qb_spieler: valid_spieler.append(qb_spieler)
            valid_spieler.extend(offense_spieler)

            if len(valid_spieler) < 5:
                continue

            # 1. Encontrar los bordes reales de la formación en la foto original
            x_coords = [p['x'] for p in valid_spieler]
            y_coords = [p['y'] for p in valid_spieler]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Dar un margen de seguridad (15 pixeles)
            min_x -= 15; max_x += 15
            min_y -= 15; max_y += 15
            
            width = max_x - min_x
            height = max_y - min_y
            
            if width <= 0 or height <= 0: continue

            # 2. Crear un lienzo del tamaño exacto de la formación
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # 3. Dibujar los puntos EXACTAMENTE como se ven en la cámara
            for p in valid_spieler:
                px = p['x'] - min_x
                py = p['y'] - min_y
                
                color = (255, 255, 255) # Blanco
                if p == qb_spieler: color = (0, 0, 255) # Rojo
                elif p == center_spieler: color = (0, 255, 0) # Verde
                
                # Círculos más grandes para que la CNN los vea claro
                cv2.circle(canvas, (px, py), radius=6, color=color, thickness=-1)

            # 4. Magia de Aspect Ratio: Convertir a cuadrado rellenando con negro
            max_dim = max(width, height)
            square_canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            
            # Centrar el rectángulo original dentro del cuadrado
            x_offset = (max_dim - width) // 2
            y_offset = (max_dim - height) // 2
            
            square_canvas[y_offset:y_offset+height, x_offset:x_offset+width] = canvas

            # 5. Redimensionar a 224x224 para YOLO
            final_img = cv2.resize(square_canvas, (224, 224), interpolation=cv2.INTER_AREA)

            out_filename = out_folder / f"{image_file.stem}_synth.jpg"
            cv2.imwrite(str(out_filename), final_img)
            total_bilder += 1
            
    print(f"✅ Synthetic Dataset generiert! Total Bilder: {total_bilder}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_DIR = BASE_DIR / "datasets" / "dataset-formation-cropped" / "train" 
    DETECTOR_PATH = BASE_DIR / "models" / "spielererkennung" / "weights" / "best.pt" 
    OUTPUT_DIR = BASE_DIR / "datasets" / "synthetic_dataset" / "train" 
    generate_synthetic_dataset(DATASET_DIR, DETECTOR_PATH, OUTPUT_DIR)