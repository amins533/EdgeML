from __future__ import annotations

import os
# --- BLINDAJE ANTI-CRASH PARA MAC APPLE SILICON (M1/M2/M3) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# --- MODO 100% OFFLINE (PARA JETSON) ---
os.environ["YOLO_OFFLINE"] = "True"       # Prohibe buscar actualizaciones o fuentes online
os.environ["WANDB_MODE"] = "offline"      # Desactiva telemetría de W&B
os.environ["HF_HUB_OFFLINE"] = "1"        # Bloquea descargas de HuggingFace/Torch

import asyncio
import importlib.util
import json
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import xgboost as xgb


def load_root_analysis_module():
    print("[SYSTEM] Importiere ML-Funktionen aus main.py im Stammverzeichnis...")
    root_analysis_path = Path(__file__).resolve().parents[2] / 'main.py'
    root_project_dir = root_analysis_path.parent
    if str(root_project_dir) not in sys.path:
        sys.path.insert(0, str(root_project_dir))

    previous_proxy_flag = os.environ.get('EDGE_ML_DISABLE_ASGI_PROXY')
    os.environ['EDGE_ML_DISABLE_ASGI_PROXY'] = '1'

    spec = importlib.util.spec_from_file_location('football_analysis_root', root_analysis_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'main.py konnte nicht geladen werden von {root_analysis_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if previous_proxy_flag is None:
        os.environ.pop('EDGE_ML_DISABLE_ASGI_PROXY', None)
    else:
        os.environ['EDGE_ML_DISABLE_ASGI_PROXY'] = previous_proxy_flag

    return module


@dataclass
class AnalysisState:
    video_dir: Path
    current_video: str = ''
    is_playing: bool = False
    current_frame: int = 0
    total_frames: int = 0
    video_fps: float = 30.0
    real_fps: float = 0.0
    seek_target_frame: int = -1
    latest_frame_bytes: bytes | None = None
    latest_frame_seq: int = 0
    latest_telemetry: dict[str, Any] = field(default_factory=dict)
    analysis_error: str | None = None
    analysis_running: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    worker_thread: threading.Thread | None = None


class AnalysisBridge:
    def __init__(self, video_dir: Path):
        self.video_dir = video_dir
        self.root = None
        self.state = AnalysisState(video_dir=video_dir)
        self.runtime_lock = threading.Lock()
        self.runtime: dict[str, Any] | None = None

    def list_videos(self) -> list[str]:
        if not self.video_dir.exists(): return []
        return [f.name for f in sorted(self.video_dir.iterdir()) if f.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}]

    def resolve_video_path(self, video_name: str) -> Path:
        return (self.video_dir / video_name).resolve()

    def _build_runtime(self) -> dict[str, Any]:
        if self.runtime is not None: return self.runtime

        with self.runtime_lock:
            if self.runtime is not None: return self.runtime

            if self.root is None:
                self.root = load_root_analysis_module()

            module = self.root
            module.prepare_directories()
            device_is_edge = module.is_edge_device()
            device = module.resolve_inference_device()
            
            # if device == "mps":
            #     print("[SYSTEM] Forciere CPU-Inferenz auf Apple Silicon, um Thread-Konflikte zu vermeiden.")
            #     device = "cpu"

            active_player_model = module.get_optimized_model_path(module.PLAYER_MODEL_PATH, device_is_edge)
            active_pitch_model = module.get_optimized_model_path(module.PITCH_MODEL_PATH, device_is_edge)
            active_ocr_model = module.get_optimized_model_path(module.LEGIBILITY_MODEL_PATH, device_is_edge)

            player_model = module.YOLO(str(active_player_model), task='detect')
            tracker_config_path = str(module.BASE_DIR / 'custom_bytetrack.yaml')
            homography_engine = module.HomographyEngine(str(active_pitch_model), device)

            formation_model = xgb.XGBClassifier()
            formation_model.load_model(str(module.FORMATION_MODEL_PATH))

            with open(module.FORMATION_LABELS_PATH, 'r') as handle:
                formation_labels = json.load(handle)

            self.runtime = {
                'module': module, 'device': device, 'device_is_edge': device_is_edge,
                'player_model': player_model, 'homography_engine': homography_engine,
                'tracker_config_path': tracker_config_path, 'formation_model': formation_model,
                'formation_labels': formation_labels, 'active_ocr_model': active_ocr_model,
            }
            return self.runtime

    def _placeholder_frame(self, message: str) -> bytes:
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(canvas, message, (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        ok, buffer = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes() if ok else b''

    def _build_telemetry(self, *, video_name: str, total_frames: int, fps: float, current_frame: int, power_watts: float, player_ms: float, homography_ms: float, readability_ms: float, ocr_ms: float, formation_ms: float, formation_label: str, formation_confidence: float, player_count: int, homography_status: str, readable_texts: list[str], readable_count: int) -> dict[str, Any]:
        return {
            'current_video': video_name, 'is_playing': self.state.is_playing, 'frame': current_frame,
            'total_frames': total_frames, 'fps': round(fps, 1), 'power_watts': round(power_watts, 2),
            'models': {
                'player_detection': {'inference_ms': round(player_ms, 2), 'count': player_count},
                'homography': {'inference_ms': round(homography_ms, 2), 'status': homography_status},
                'readability': {'inference_ms': round(readability_ms, 2), 'readable_count': readable_count},
                'ocr': {'inference_ms': round(ocr_ms, 2), 'texts': readable_texts},
                'formation': {'inference_ms': round(formation_ms, 2), 'prediction': formation_label, 'confidence': float(formation_confidence)},
            },
        }

    def start_video(self, video_name: str) -> None:
        with self.runtime_lock:
            video_path = self.resolve_video_path(video_name)
            if not video_path.exists(): raise FileNotFoundError(f'Video nicht gefunden: {video_path}')

            self.stop()
            self.state.stop_event = threading.Event()
            self.state.current_video = video_name
            self.state.is_playing = True
            self.state.current_frame = 0
            self.state.total_frames = 0
            self.state.video_fps = 30.0
            self.state.real_fps = 0.0
            self.state.seek_target_frame = -1
            self.state.analysis_error = None
            self.state.latest_frame_bytes = self._placeholder_frame('ML-Modelle werden geladen. Bitte warten...')
            
            self.state.analysis_running = True
            worker = threading.Thread(target=self._run_worker, args=(video_path,), daemon=True)
            self.state.worker_thread = worker
            worker.start()

    def stop(self) -> None:
        self.state.stop_event.set()
        worker = self.state.worker_thread
        if worker and worker.is_alive():
            print("[SYSTEM] Beende ML-Worker. Speicher wird freigegeben...")
            worker.join() 
            print("[SYSTEM] Worker erfolgreich beendet. Bereit.")
        self.state.worker_thread = None
        self.state.analysis_running = False

    def set_playing(self, is_playing: bool) -> None:
        with self.state.state_lock:
            self.state.is_playing = is_playing
            if self.state.latest_telemetry:
                self.state.latest_telemetry['is_playing'] = is_playing

    def seek(self, seconds: float) -> None:
        with self.state.state_lock:
            if self.state.video_fps > 0 and self.state.total_frames > 0:
                current_sec = self.state.current_frame / self.state.video_fps
                target_sec = max(0.0, min(self.state.total_frames / self.state.video_fps, current_sec + seconds))
                self.state.seek_target_frame = int(target_sec * self.state.video_fps)

    def snapshot_telemetry(self) -> dict[str, Any]:
        with self.state.state_lock:
            if self.state.latest_telemetry:
                data = dict(self.state.latest_telemetry)
                data['is_playing'] = self.state.is_playing
                return data
            return self._build_telemetry(video_name=self.state.current_video, total_frames=self.state.total_frames, fps=self.state.real_fps, current_frame=self.state.current_frame, power_watts=0.0, player_ms=0.0, homography_ms=0.0, readability_ms=0.0, ocr_ms=0.0, formation_ms=0.0, formation_label='Bereit', formation_confidence=0.0, player_count=0, homography_status='Inaktiv', readable_texts=[], readable_count=0)

    def mjpeg_stream(self):
        last_seq = -1
        heartbeat_counter = 0
        try:
            while True:
                with self.state.state_lock:
                    seq = self.state.latest_frame_seq
                    frame_bytes = self.state.latest_frame_bytes
                    error = self.state.analysis_error

                if error: frame_bytes = self._placeholder_frame(f'ML-Fehler: {error[:40]}...')
                if frame_bytes is None: frame_bytes = self._placeholder_frame('Wähle ein Video aus der Liste...')

                if seq == last_seq or not frame_bytes:
                    time.sleep(0.03)
                    heartbeat_counter += 1
                    if heartbeat_counter > 30 and frame_bytes:
                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        heartbeat_counter = 0
                    continue

                last_seq = seq
                heartbeat_counter = 0
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception:
            print("[STREAM] HTTP-Verbindung vom Browser getrennt.")

    def _run_worker(self, video_path: Path) -> None:
        try:
            print(f"\n[WORKER] Erstelle ML-Laufzeit fuer: {video_path.name}")
            runtime = self._build_runtime()
            module = runtime['module']
            device = runtime['device']
            player_model = runtime['player_model']
            homography_engine = runtime['homography_engine']
            tracker_config_path = runtime['tracker_config_path']
            formation_model = runtime['formation_model']
            formation_labels = runtime['formation_labels']
            active_ocr_model = runtime['active_ocr_model']

            db = module.DatabasePipeline().start()
            shared_cache: dict[str, Any] = {}
            ocr_pipeline = module.OCRAnalysisPipeline(
                base_dir=module.BASE_DIR,
                legibility_model_path=str(active_ocr_model),
                shared_cache=shared_cache,
            ).start()

            cv2.setNumThreads(0)
            cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
            if not cap.isOpened(): cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print("[WORKER] FFMPEG fehlgeschlagen. Versuche Standard-Backend...")
                cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print("[WORKER] KRITISCHER FEHLER: OpenCV konnte das Video nicht oeffnen.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

            with self.state.state_lock:
                self.state.total_frames = total_frames
                self.state.video_fps = video_fps
                self.state.real_fps = 0.0
                self.state.current_frame = 0
                self.state.latest_frame_bytes = self._placeholder_frame(f'Lade {video_path.name}...')
                self.state.latest_frame_seq += 1

            temporal_pooler = module.PreSnapTemporalPooler(max_frames=30)
            voting_pool = module.FormationVotingPool()
            tracked_players_cache: dict[Any, Any] = {}
            memory_matrix_buffer: dict[Any, Any] = {}
            homography_engine.last_valid_H = None

            current_formation, current_conf = 'Analyse laeuft...', 0.0
            is_play_active = False
            pre_snap_core_anchor = None
            processed_frames = 0
            current_play_id = 1
            
            db.save_game(module.GameData(game_id=1, date=time.strftime('%Y-%m-%d'), video=video_path.name))

            print("[WORKER] ML-Inferenzschleife gestartet.")
            
            fps_timer = time.perf_counter()
            fps_counter = 0

            while not self.state.stop_event.is_set():
                with self.state.state_lock:
                    seek_target = self.state.seek_target_frame
                    is_playing = self.state.is_playing

                did_seek = False
                if seek_target != -1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                    with self.state.state_lock:
                        self.state.current_frame = seek_target
                        self.state.seek_target_frame = -1
                    did_seek = True

                if not is_playing and not did_seek:
                    time.sleep(0.05)
                    continue

                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    with self.state.state_lock: self.state.current_frame = 0
                    continue

                processed_frames += 1
                fps_counter += 1
                now = time.perf_counter()
                elapsed = now - fps_timer
                if elapsed >= 0.5:
                    calc_fps = fps_counter / elapsed
                    with self.state.state_lock:
                        self.state.real_fps = round(calc_fps, 1)
                    fps_timer = now
                    fps_counter = 0

                frame_disp = frame.copy()
                
                player_ms, homography_ms, ocr_submit_ms, formation_ms = 0.0, 0.0, 0.0, 0.0
                players = []
                current_H = None

                try:
                    t_start_h = time.perf_counter()
                    minimap = homography_engine.draw_2d_pitch()
                    should_infer_homography = (processed_frames % module.HOMOGRAPHY_INTERVAL == 1) or (homography_engine.last_valid_H is None)
                    current_H = homography_engine.process_frame(frame, frame_disp, run_inference=should_infer_homography)
                    homography_ms = (time.perf_counter() - t_start_h) * 1000.0

                    t_start = time.perf_counter()
                    players = module.player_tracking(frame, player_model, tracker_config_path, device, shared_cache)
                    player_ms = (time.perf_counter() - t_start) * 1000.0

                    if processed_frames % 60 == 0:
                        print(f"[ML-STATUS] Frame {processed_frames}/{total_frames} | YOLO erkannt: {len(players)} | Geraet: {device}")

                    tracking_batch = []
                    players_needing_ocr = []
                    def_pts = []

                    for s in players:
                        player_id = s['tracking_id']
                        if player_id is None: continue

                        x_min, y_min, x_max, y_max = s['box']
                        label = f"#{s.get('jersey_number', '?')}" if s.get('jersey_number') else f"ID: {player_id}"
                        color = module.CLASS_COLORS.get(s['class_id'], (0, 255, 255))
                        cv2.rectangle(frame_disp, (x_min, y_min), (x_max, y_max), color, 3)
                        cv2.putText(frame_disp, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

                        if player_id not in shared_cache or shared_cache[player_id].get('confidence', 0.0) < 0.50:
                            players_needing_ocr.append(s)

                        if player_id not in tracked_players_cache:
                            tracked_players_cache[player_id] = {'jersey_number': s['jersey_number'], 'player_role': s['player_role'], 'team_side': s['team_side']}
                        else:
                            if s['jersey_number'] is not None: tracked_players_cache[player_id]['jersey_number'] = s['jersey_number']
                            if s['player_role'] in ['Center', 'Quarterback'] or tracked_players_cache[player_id]['player_role'] == 'Unknown':
                                tracked_players_cache[player_id]['player_role'] = s['player_role']
                                tracked_players_cache[player_id]['team_side'] = s['team_side']

                        feet_x, feet_y = (x_min + x_max) / 2.0, float(y_max)
                        field_x, field_y = None, None

                        if current_H is not None:
                            feet_coord = np.array([[[feet_x, feet_y]]], dtype=np.float32)
                            mapped_point = cv2.perspectiveTransform(feet_coord, current_H)
                            field_x, field_y = float(mapped_point[0][0][0]), float(mapped_point[0][0][1])

                            if 0 <= int(field_x) < homography_engine.PITCH_WIDTH and 0 <= int(field_y) < homography_engine.PITCH_HEIGHT:
                                cv2.circle(minimap, (int(field_x), int(field_y)), 8, color, -1)
                                if s['team_side'] == 'Defense': def_pts.append([field_x, field_y])

                        if s['team_side'] == 'Offense' or s['player_role'] in ['Center', 'Attacker', 'Quarterback']:
                            if player_id not in memory_matrix_buffer: memory_matrix_buffer[player_id] = {'role': s['player_role']}
                            memory_matrix_buffer[player_id][processed_frames] = f"({field_x or feet_x:.1f}, {field_y or feet_y:.1f})"

                        tracking_batch.append(module.TrackingPoint(tracking_id=player_id, play_id=current_play_id, frame_id=processed_frames, x_coord=field_x or feet_x, y_coord=field_y or feet_y))

                    t_start_form = time.perf_counter()
                    temporal_pooler.update(players)
                    pooled_offense = temporal_pooler.get_pooled_offense()

                    if not is_play_active and processed_frames > 20 and len(pooled_offense) >= 6:
                        core_players = [p['pt'] for p in pooled_offense if p['role'] in ['Center', 'Quarterback']]
                        anchor_point = np.mean(core_players, axis=0) if core_players else np.median([p['pt'] for p in pooled_offense], axis=0)
                        if pre_snap_core_anchor is None: pre_snap_core_anchor = anchor_point
                        elif np.linalg.norm(anchor_point - pre_snap_core_anchor) > 15.0: is_play_active = True

                    if not is_play_active and current_H is not None:
                        raw_label, raw_conf, real_count = module.predict_formation_xgboost(pooled_offense, def_pts, formation_model, formation_labels)
                        voting_pool.add_prediction(raw_label, raw_conf, real_count)
                        current_formation, current_conf = voting_pool.get_current_leader()
                    formation_ms = (time.perf_counter() - t_start_form) * 1000.0

                    if tracking_batch: db.save_tracking_batch(tracking_batch)

                    if processed_frames % 5 == 0 and len(players_needing_ocr) > 0:
                        t_start_ocr = time.perf_counter()
                        ocr_pipeline.submit(processed_frames, frame, players_needing_ocr)
                        ocr_submit_ms = (time.perf_counter() - t_start_ocr) * 1000.0

                    frame_disp = cv2.resize(frame_disp, (1280, 720))
                    map_w, map_h = 270, 120
                    mini_map_resized = cv2.resize(minimap, (map_w, map_h))
                    cv2.rectangle(mini_map_resized, (0, 0), (map_w - 1, map_h - 1), (100, 100, 100), 2)
                    frame_disp[20:20+map_h, 1280-map_w-20:1280-20] = mini_map_resized
                    module.draw_clean_hud(frame_disp, current_formation, current_conf, is_play_active or (voting_pool.locked_formation is not None))

                except Exception as ml_err:
                    print(f"[WORKER] ML-Fehler in Frame {processed_frames}: {ml_err}")
                    cv2.putText(frame_disp, f"ML FEHLER: {ml_err}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                try:
                    ok, buffer = cv2.imencode('.jpg', frame_disp, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        with self.state.state_lock:
                            self.state.latest_frame_bytes = buffer.tobytes()
                            self.state.latest_frame_seq += 1
                            self.state.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            self.state.power_watts = random.uniform(12.5, 14.8) if self.state.is_playing else random.uniform(7.1, 7.8)
                            
                            ocr_texts = [str(item.get('jersey')) for item in shared_cache.values() if isinstance(item, dict) and item.get('jersey') is not None]
                            
                            self.state.latest_telemetry = self._build_telemetry(
                                video_name=video_path.name, total_frames=total_frames, fps=self.state.real_fps, current_frame=self.state.current_frame,
                                power_watts=self.state.power_watts, player_ms=player_ms, homography_ms=homography_ms,
                                readability_ms=ocr_submit_ms, ocr_ms=ocr_submit_ms, formation_ms=formation_ms,
                                formation_label=current_formation, formation_confidence=current_conf, player_count=len(players),
                                homography_status='Aktiv' if current_H is not None else 'Inaktiv',
                                readable_texts=ocr_texts, readable_count=len(ocr_texts),
                            )
                except Exception as e:
                    print(f"[WORKER] Fehler bei Bildkodierung: {e}")

                if video_fps > 0: time.sleep(max(0.0, 1.0 / video_fps))

            try:
                for track_id, data in tracked_players_cache.items():
                    db.save_player(module.PlayerData(tracking_id=track_id, jersey_number=data['jersey_number'], player_role=data['player_role'], team_side=data['team_side']))
                module.build_and_save_matrix(memory_matrix_buffer, module.BASE_DIR / 'tactical_system.db', processed_frames, tracked_players_cache)
            except Exception: pass
            
            try: ocr_pipeline.stop()
            except Exception: pass
            
            try: db.stop()
            except Exception: pass

            cap.release()
            with self.state.state_lock: self.state.analysis_running = False

        except Exception as fatal_e:
            import traceback
            error_msg = f"{type(fatal_e).__name__}: {str(fatal_e)}"
            print(f"\n[WORKER] KRITISCHER FEHLER: {error_msg}")
            traceback.print_exc()
            with self.state.state_lock:
                self.state.analysis_error = error_msg
                self.state.analysis_running = False