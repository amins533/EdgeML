import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def get_center_coordinates(box):
    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
    return (x_min + x_max) / 2, (y_min + y_max) / 2

def create_formation_dataset(dataset_dir, model_path, output_csv):
    model = YOLO(model_path)
    dataset_rows = []
    formation_folders = [d for d in Path(dataset_dir).iterdir() if d.is_dir()]

    for folder in formation_folders:
        label = folder.name
        for image_file in folder.glob("*.jpg"):
            results = model.predict(str(image_file), conf=0.50, verbose=False)
            boxes = results[0].boxes

            offense_spieler = []
            center_spieler = None
            qb_spieler = None

            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in [0, 2, 3]:
                    cx, cy = get_center_coordinates(box)
                    spieler = {'cls': cls_id, 'x': cx, 'y': cy}
                    if cls_id == 0:
                        center_spieler = spieler
                    elif cls_id == 3:
                        qb_spieler = spieler
                    else:
                        offense_spieler.append(spieler)

            valid_spieler = []
            if center_spieler: valid_spieler.append(center_spieler)
            if qb_spieler: valid_spieler.append(qb_spieler)
            valid_spieler.extend(offense_spieler)

            if len(valid_spieler) < 5:
                continue

            # 1. Asegurar el Center (Ancla 0,0)
            if not center_spieler:
                avg_x = np.mean([s['x'] for s in offense_spieler])
                avg_y = np.mean([s['y'] for s in offense_spieler])
                center_spieler = {'x': avg_x, 'y': avg_y}

            # 2. Descubrir Dirección de Ataque (Brújula)
            if qb_spieler:
                is_facing_left = qb_spieler['x'] > center_spieler['x']
            else:
                avg_x = np.mean([s['x'] for s in offense_spieler])
                is_facing_left = avg_x > center_spieler['x']

            # Si miran a la izquierda, la matemática se invierte para estandarizar
            dir_multiplier = 1 if is_facing_left else -1

            # 3. Escala global estricta (Solo usa el Ancho para no deformar Aspect Ratio)
            x_coords = [s['x'] for s in valid_spieler]
            escala = max(x_coords) - min(x_coords)
            if escala == 0: escala = 1

            # 4. Normalización Relativa 
            def norm_rel(s):
                # X positivo SIEMPRE es el Backfield (profundidad). X negativo es la linea.
                rel_x = ((s['x'] - center_spieler['x']) * dir_multiplier) / escala
                # Y es la distribución a lo ancho del campo
                rel_y = (s['y'] - center_spieler['y']) / escala
                return rel_x, rel_y

            if qb_spieler:
                q_nx, q_ny = norm_rel(qb_spieler)
            else:
                q_nx, q_ny = np.nan, np.nan

            for s in offense_spieler:
                s['nx'], s['ny'] = norm_rel(s)

            # Ordenar por Y (De la banda de arriba a la banda de abajo de la cámara)
            offense_spieler = sorted(offense_spieler, key=lambda s: s['ny'])

            row_data = {
                'formation': label,
                'qb_x': q_nx, 'qb_y': q_ny
            }

            # 5. Smart Features para el Random Forest
            row_data['backfield_count'] = sum(1 for s in offense_spieler if s['nx'] > 0.05)
            row_data['left_count'] = sum(1 for s in offense_spieler if s['ny'] < -0.05)
            row_data['right_count'] = sum(1 for s in offense_spieler if s['ny'] > 0.05)

            for i in range(9):
                prefix = f'o{i+1}'
                if i < len(offense_spieler):
                    row_data[f'{prefix}_x'] = offense_spieler[i]['nx']
                    row_data[f'{prefix}_y'] = offense_spieler[i]['ny']
                else:
                    row_data[f'{prefix}_x'] = np.nan
                    row_data[f'{prefix}_y'] = np.nan

            dataset_rows.append(row_data)

    df = pd.DataFrame(dataset_rows)
    df.to_csv(output_csv, index=False)
    print(f"Dataset V2 exportiert! Total rows: {len(df)}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_DIR = BASE_DIR / "datasets" / "dataset-formation-cropped" / "train"
    MODEL_PATH = BASE_DIR / "models" / "spielererkennung" / "v2" / "weights" / "best.pt"
    OUTPUT_FILE = BASE_DIR / "dataset_formationen.csv"

    create_formation_dataset(DATASET_DIR, MODEL_PATH, OUTPUT_FILE)