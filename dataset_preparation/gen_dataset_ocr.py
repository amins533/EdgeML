import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from video_pipeline import VideoPipeLine

def save_crops(frame, model, output_dir, frame_count, video_name):
    results = model.track(frame, conf=0.479, persist=True, verbose=False, imgsz=320, device="mps", tracker="bytetrack.yaml")
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy() # Extract classes to identify referees

        # Now we zip boxes, track_ids AND clss
        for box, track_id, cls in zip(boxes, track_ids, clss):
            
            # Filter out the referees (Class 4)
            if int(cls) == 4:
                continue

            track_id = int(track_id)
            x_min, y_min, x_max, y_max = map(int, box)
            
            frame_height, frame_width = frame.shape[:2]
            y1 = max(0, y_min)
            y2 = min(frame_height, y_max)
            x1 = max(0, x_min)
            x2 = min(frame_width, x_max)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                filename = f"{video_name}_f{frame_count}_id{track_id}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, crop)

def main():
    base_dir = Path(__file__).resolve().parent.parent
    video_dir = base_dir / "videos-ocr"
    model_path = base_dir / "models" / "spielererkennung" / "v3" / "weights" / "best.pt"
    
    output_dir = base_dir / "datasets" / "dataset" / "player_crops_v2"
    os.makedirs(output_dir, exist_ok=True)

    if not video_dir.exists():
        raise FileNotFoundError(f"Directory not found: {video_dir}")
        
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLO(str(model_path))
    videos = list(video_dir.glob("*.mp4"))

    print(f"Found {len(videos)} videos. Starting extraction to: {output_dir}")

    frame_skip = 10

    for video_path in videos:
        print(f"Processing video: {video_path.name}")
        pipeline = VideoPipeLine(video_path=str(video_path)).start()
        frame_count = 0

        while True:
            if not pipeline.more():
                if pipeline.stopped:
                    break
                continue

            frame = pipeline.read()
            if frame is None: 
                break

            if frame_count % frame_skip == 0:
                save_crops(frame, model, str(output_dir), frame_count, video_path.stem)
            
            frame_count += 1
            
            if frame_count % 300 == 0:
                print(f"Processed {frame_count} frames for {video_path.name}...")

        pipeline.stop()

    print("Extraction complete! Check the player_crops folder.")

if __name__ == "__main__":
    main()