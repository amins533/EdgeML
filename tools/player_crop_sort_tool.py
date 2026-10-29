import os
import shutil
import cv2
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parent.parent
    source_dir = base_dir / "datasets" / "cropped_dataset_ocr" / "unfiltered_dataset"
    legible_dir = base_dir / "datasets" / "cropped_dataset_ocr" / "train" / "legible"
    ilegible_dir = base_dir / "datasets" / "cropped_dataset_ocr" / "train" / "ilegible"

    os.makedirs(legible_dir, exist_ok=True)
    os.makedirs(ilegible_dir, exist_ok=True)

    if not source_dir.exists():
        print(f"Source folder not found: {source_dir}")
        return

    images = [
        image_path
        for image_path in source_dir.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    print(f"Classifying {len(images)} images.")
    print("CONTROLS: '1' for Legible | '2' for Ilegible | '3' for Delete | 'q' for Quit")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_show = cv2.resize(img, (520, 520), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Sorter", img_show)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            print("Quiting sorter...")
            break

        if key == ord('1'):
            shutil.move(str(img_path), str(legible_dir / img_path.name))
        elif key == ord('2'):
            shutil.move(str(img_path), str(ilegible_dir / img_path.name))
        elif key == ord('3'):
            os.remove(img_path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()