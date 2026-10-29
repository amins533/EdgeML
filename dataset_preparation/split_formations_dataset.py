from pathlib import Path
import random
import shutil


BASE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "dataset-formation"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
SPLIT_RATIO = 0.2
RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def reset_val_to_train() -> None:
    if not VAL_DIR.exists():
        return

    for image_path in VAL_DIR.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            # Extract formation name (parent folder)
            formation_name = image_path.parent.name
            target_dir = TRAIN_DIR / formation_name
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(target_dir / image_path.name))

def split_train_val() -> None:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")

    reset_val_to_train()
    rng = random.Random(RANDOM_SEED)
    total_moved = 0

    for formation_dir in sorted([path for path in TRAIN_DIR.iterdir() if path.is_dir()]):
        images = sorted(
            [path for path in formation_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS]
        )

        if not images:
            continue

        move_count = int(len(images) * SPLIT_RATIO)
        selected_images = rng.sample(images, move_count) if move_count else []
        val_formation_dir = VAL_DIR / formation_dir.name
        val_formation_dir.mkdir(parents=True, exist_ok=True)

        for image_path in selected_images:
            destination_path = val_formation_dir / image_path.name
            if destination_path.exists() and destination_path != image_path:
                destination_path.unlink()
            shutil.move(str(image_path), str(destination_path))
            total_moved += 1

        train_count = len(images) - move_count
        print(f"[{formation_dir.name}] -> Train: {train_count} images | Val: {move_count} images")

    print("-" * 40)
    print(f"Validation split refreshed: {total_moved} images moved to val.")


def main():
    print("Creating an 80% train / 20% val split...")
    split_train_val()


if __name__ == "__main__":
    main()