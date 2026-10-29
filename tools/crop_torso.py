import cv2
import os
from pathlib import Path
from ultralytics import YOLO

def extract_torso(img_path, model, output_path):
    img = cv2.imread(str(img_path))
    if img is None: return

    results = model(img, verbose=False)
    h, w = img.shape[:2]
    
    if len(results[0]) > 0 and results[0].keypoints is not None:
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        
        if kpts.shape[0] >= 13:
            torso_points = [kpts[5], kpts[6], kpts[11], kpts[12]]
            valid_points = [pt for pt in torso_points if pt[0] > 0 and pt[1] > 0]
            
            if len(valid_points) >= 3:
                xs = [pt[0] for pt in valid_points]
                ys = [pt[1] for pt in valid_points]
                
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                
                box_w = x_max - x_min
                box_h = y_max - y_min
                
                # --- PADDING LOGIC ---
                pad_x = int(box_w * 0.15)
                pad_y = int(box_h * 0.10)
                
                # --- SMART SQUEEZE LOGIC ---
                # squeeze_x = int(box_w * 0.10)
                # squeeze_y_bottom = int(box_h * 0.15)

                squeeze_x = 0
                squeeze_y_bottom = 0

                x1 = (x_min + squeeze_x) - pad_x
                x2 = (x_max - squeeze_x) + pad_x
                y1 = y_min - pad_y
                y2 = (y_max - squeeze_y_bottom) + pad_y
                
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                
                if x1 < x2 and y1 < y2:
                    torso_crop = img[y1:y2, x1:x2]
                    if torso_crop.size > 0:
                        cv2.imwrite(str(output_path), torso_crop)
                        return

    crop_x = int(w * 0.15)
    crop_top = int(h * 0.15)
    crop_bottom = int(h * 0.55)
    
    x1_fb = crop_x
    x2_fb = w - crop_x
    y1_fb = crop_top
    y2_fb = h - crop_bottom
    
    if x1_fb < x2_fb and y1_fb < y2_fb:
        fallback_crop = img[y1_fb:y2_fb, x1_fb:x2_fb]
        if fallback_crop.size > 0:
            cv2.imwrite(str(output_path), fallback_crop)
            return
            
    cv2.imwrite(str(output_path), img)

def main():
    input_dataset = Path("/home/iai-labor/info-projekt/raw_dataset_ocr/player_crops_v2")
    output_dataset = Path("/home/iai-labor/info-projekt/cropped_dataset_ocr")
    
    print("Loading heavy pose model (yolo11x-pose.pt)...")
    model = YOLO("yolo11x-pose.pt")
    
    has_splits = (input_dataset / "train").exists() or (input_dataset / "val").exists()
    total_processed = 0
    
    if has_splits:
        splits = ['train', 'val']
        classes = ['legible', 'ilegible']
        
        for split in splits:
            for c in classes:
                input_folder = input_dataset / split / c
                output_folder = output_dataset / split / c
                
                if not input_folder.exists():
                    print(f"Skipping: Source folder does not exist -> {input_folder}")
                    continue
                    
                images = list(input_folder.glob("*.jpg"))
                if not images:
                    print(f"Notice: No .jpg images found inside -> {input_folder}")
                    continue
                    
                output_folder.mkdir(parents=True, exist_ok=True)
                print(f"Processing {len(images)} images in {split}/{c}...")
                
                for img_path in images:
                    out_path = output_folder / img_path.name
                    extract_torso(img_path, model, out_path)
                    total_processed += 1
                    if total_processed % 100 == 0:
                        print(f"Cropped {total_processed} torsos...")
    else:
        images = list(input_dataset.glob("*.jpg"))
        if not images:
            print(f"Error: Absolutely no .jpg images found directly in {input_dataset}")
            return
            
        output_dataset.mkdir(parents=True, exist_ok=True)
        print(f"Processing {len(images)} images directly from flat folder...")
        
        for img_path in images:
            out_path = output_dataset / img_path.name
            extract_torso(img_path, model, out_path)
            total_processed += 1
            if total_processed % 100 == 0:
                print(f"Cropped {total_processed} images...")
                    
    print(f"\nExecution complete! Total processed images: {total_processed}")
    print(f"Saved dataset destination: {output_dataset}")

if __name__ == "__main__":
    main()