import queue
import shutil
import threading
import time
from pathlib import Path

import cv2
import easyocr
import torch
from ultralytics import YOLO

def resolve_inference_device():
    """Selects the best available hardware accelerator for PyTorch."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return 0
    return "cpu"

def prepare_directories(base_dir):
    """Creates and clears directories for storing OCR crops and analysis images."""
    legible_dir = base_dir / "ocr_real_runs" / "legible_player_crops"
    analysis_dir = base_dir / "ocr_real_runs" / "ocr_analisis"
    
    for d in (legible_dir, analysis_dir):
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    return legible_dir, analysis_dir

def get_model_class_names(model):
    """Safely retrieves class names from a YOLO model."""
    names = getattr(model, "names", {})
    if isinstance(names, dict): return {int(k): v for k, v in names.items()}
    return {i: v for i, v in enumerate(names)}

def crop_torso_from_box(frame, box):
    """Extracts a rough upper-body crop based on the bounding box coordinates."""
    x_min, y_min, x_max, y_max = map(int, box)
    h, w = frame.shape[:2]
    bw, bh = x_max - x_min, y_max - y_min

    if bw <= 0 or bh <= 0: return None

    # Focus on the inner upper portion of the box (where numbers usually are)
    l = max(0, int(x_min + bw * 0.10))
    r = min(w, int(x_max - bw * 0.10))
    
    t = max(0, int(y_min + bh * 0.15))
    b = min(h, int(y_min + bh * 0.625))

    if l >= r or t >= b: return None

    crop = frame[t:b, l:r]
    return crop if crop.size > 0 else None

def classify_legibility(torso_crop, legibility_model, device):
    """Checks if the torso crop has a readable jersey number."""
    results = legibility_model.predict(torso_crop, verbose=False, imgsz=224, device=device)
    if not results or results[0].probs is None: return False, None, 0.0

    probs = results[0].probs
    class_idx = int(probs.top1)
    conf = float(probs.top1conf.item())
    names = get_model_class_names(legibility_model)
    label = str(names.get(class_idx, class_idx))

    return label.lower() == "legible", label, conf

def perform_ocr(reader, torso_crop):
    """Uses EasyOCR to extract numeric text from the valid torso crop."""
    results = reader.readtext(
        torso_crop, 
        allowlist="0123456789", 
        decoder="beamsearch", 
        beamWidth=8, 
        mag_ratio=2.5, 
        contrast_ths=0.25, 
        adjust_contrast=0.7, 
        width_ths=1.2, 
        min_size=5,
        text_threshold=0.75,
        low_text=0.5,
        add_margin=0.2
    )

    if not results: return None, None
    best = max(results, key=lambda i: i[2])
    return results, best

def save_raw_crop(output_dir, torso_crop, frame_index, detection_index, track_id, crop_method, ocr_text):
    """Saves the final processed crop image to the disk."""
    safe_track_id = track_id if track_id is not None else -1
    display_text = ocr_text if ocr_text else "None"
    
    name = f"det_{detection_index:02d}_{crop_method}-{display_text}.jpg"
    path = output_dir / name
    
    cv2.imwrite(str(path), torso_crop)
    return path

def flush_ocr_visual_logs(analysis_dir, logs):
    """Draws bounding boxes around recognized text and saves the analyzed images."""
    for entry in logs:
        img = entry["image"].copy()
        for bbox, text, score in entry["all_results"]:
            tl = tuple(map(int, bbox[0]))
            br = tuple(map(int, bbox[2]))
            cv2.rectangle(img, tl, br, (0, 0, 255), 2)
            cv2.putText(img, f"{text} ({score:.2f})", tl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if entry["best_result"]:
            btl = tuple(map(int, entry["best_result"][0][0]))
            bbr = tuple(map(int, entry["best_result"][0][2]))
            cv2.rectangle(img, btl, bbr, (0, 255, 0), 2)
            
        path = analysis_dir / entry["filename"]
        cv2.imwrite(str(path), img)
    
    print(f"[OCR] Flushed {len(logs)} OCR analysis images to disk.")

class OCRAnalysisPipeline:
    """A background pipeline to handle OCR processing asynchronously to prevent blocking the main video thread."""
    
    def __init__(self, base_dir, legibility_model_path, shared_cache, max_queue_size=8):
        self.base_dir = Path(base_dir)
        self.legibility_model_path = Path(legibility_model_path)
        self.legible_dir, self.analysis_dir = prepare_directories(self.base_dir)
        self.device = resolve_inference_device()
        self.stop_requested = False
        
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.worker = None
        self.shared_cache = shared_cache
        self.visual_logs = [] 

        # --- INTERNAL OCR TIME METRICS ---
        self.time_stats = {
            "legibility": {"total": 0.0, "count": 0},
            "easyocr": {"total": 0.0, "count": 0}
        }

        # Load models
        self.legibility_model = YOLO(str(self.legibility_model_path), task="classify")
        self.ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    def start(self):
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        return self

    def submit(self, frame_index, frame, players):
        if self.stop_requested: return
        try:
            self.queue.put_nowait((frame_index, frame.copy(), players))
        except queue.Full:
            pass

    def stop(self):
        self.stop_requested = True
        if self.worker is not None: self.worker.join(timeout=2.0)
        flush_ocr_visual_logs(self.analysis_dir, self.visual_logs)
        return self.time_stats

    def _run(self):
        while not self.stop_requested or not self.queue.empty():
            try:
                frame_index, frame, players = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            for detection_index, player in enumerate(players):
                # We strictly rely on the Bounding Box Fallback now.
                torso_crop = crop_torso_from_box(frame, player["box"])
                if torso_crop is None: continue

                t2 = time.perf_counter()
                is_legible, _, _ = classify_legibility(torso_crop, self.legibility_model, self.device)
                t3 = time.perf_counter()
                self.time_stats["legibility"]["total"] += (t3 - t2)
                self.time_stats["legibility"]["count"] += 1

                if not is_legible: continue

                track_id = player.get("tracking_id", -1)
                
                t4 = time.perf_counter()
                all_res, best_res = perform_ocr(self.ocr_reader, torso_crop)
                t5 = time.perf_counter()
                self.time_stats["easyocr"]["total"] += (t5 - t4)
                self.time_stats["easyocr"]["count"] += 1

                ocr_text = None
                if best_res is not None: ocr_text = best_res[1]

                raw_path = save_raw_crop(self.legible_dir, torso_crop, frame_index, detection_index, track_id, "bbox", ocr_text)

                if all_res is not None:
                    self.visual_logs.append({
                        "filename": raw_path.name,
                        "image": torso_crop,
                        "all_results": all_res,
                        "best_result": best_res
                    })

                if best_res is None: continue

                _, text, conf = best_res
                if len(text) > 2: continue

                print(f"\033[38;5;82m[OCR] {raw_path.name} -> {text} (Confidence: {conf:.2%})\033[0m")

                if track_id != -1 and track_id is not None:
                    cache = self.shared_cache.get(track_id)
                    if not cache or conf > cache["confidence"]:
                        self.shared_cache[track_id] = {"jersey": text, "confidence": conf}
                        print(f"\033[38;5;202m[CACHE UPDATE] ID {track_id} has now jersey #{text} with conf: {conf:.2f}\033[0m")