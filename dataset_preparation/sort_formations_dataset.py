from pathlib import Path
import re
import shutil
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "dataset-formation"
TRAIN_DIR = DATASET_DIR / "train"

IMAGE_NAME_RE = re.compile(r"^Wide - Clip (\d+)(?:_presnap)?\.(png|jpe?g)$", re.IGNORECASE)

def formation_label(formation_val) -> str:
    if pd.isna(formation_val):
        return "Uncategorized"
    label = str(formation_val).strip()
    return label if label else "Uncategorized"

def process_folder(folder_path: Path):
    csv_path = folder_path / "breakdown.csv"
    if not csv_path.exists():
        print(f"Skipping {folder_path.name}: breakdown.csv not found")
        return 0, 0

    # Leer CSV
    df = pd.read_csv(csv_path, sep=";")
    
    # Mapear Clip ID (index + 1) -> OFF FORM
    # Así, "Wide - Clip 003_presnap.jpg" (ID 3) corresponde al índice 2 del CSV (3 - 1)
    play_to_form = {}
    for index, row in df.iterrows():
        clip_id = index + 1
        play_to_form[clip_id] = formation_label(row.get('OFF FORM'))

    moved = 0
    ignored = 0
    
    # Procesar imagenes
    for image_path in folder_path.iterdir():
        if not image_path.is_file():
            continue
            
        match = IMAGE_NAME_RE.match(image_path.name)
        if not match:
            continue
            
        clip_id_str = match.group(1)
        clip_id = int(clip_id_str)
        
        # Obtener formacion. Si el clip de la imagen no esta en el CSV, va a Uncategorized
        formation_name = play_to_form.get(clip_id, "Uncategorized")
        
        # Para evitar sobreescribir Clips con el mismo ID de diferentes partidos/años, agregamos el nombre del folder
        safe_folder_name = folder_path.name.replace(" ", "_").replace(".", "")
        new_filename = f"presnap_{safe_folder_name}_clip_{clip_id_str}.jpg"
        
        target_dir = TRAIN_DIR / formation_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = target_dir / new_filename
        
        # Solo movemos si el destino no es el mismo archivo
        if target_path != image_path:
            shutil.move(str(image_path), str(target_path))
            moved += 1

    return moved, len(list(folder_path.glob("*.jpg")))  # remaining unparsed jpgs if any

def main():
    print("Iniciando mapeo de imagenes a traves del PLAY # de los breakdown.csv...")
    if not TRAIN_DIR.exists():
        TRAIN_DIR.mkdir(parents=True)

    total_moved = 0
    
    # Procesar 2024 (Estructura plana)
    folder_2024 = DATASET_DIR / "presnap_frames_2024"
    if folder_2024.exists():
        moved, _ = process_folder(folder_2024)
        total_moved += moved
        print(f"[presnap_frames_2024] Movidas: {moved}")

    # Procesar 2025 (Subcarpetas por partido)
    folder_2025 = DATASET_DIR / "presnap_frames_2025"
    if folder_2025.exists():
        for subfolder in sorted(folder_2025.iterdir()):
            if subfolder.is_dir():
                moved, _ = process_folder(subfolder)
                total_moved += moved
                print(f"[{subfolder.name}] Movidas: {moved}")

    print("-" * 40)
    print(f"¡MAGIA COMPLETADA! 🪄")
    print(f"✅ Se mapearon y movieron un total de {total_moved} imágenes a {TRAIN_DIR.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
