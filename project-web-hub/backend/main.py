import os
# --- BLINDAJE ANTI-CRASH PARA MAC ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# --- MODO 100% OFFLINE (PARA JETSON) ---
os.environ["YOLO_OFFLINE"] = "True"       # Prohibe buscar actualizaciones o fuentes online
os.environ["WANDB_MODE"] = "offline"      # Desactiva telemetría de W&B
os.environ["HF_HUB_OFFLINE"] = "1"        # Bloquea descargas de HuggingFace/Torch

import asyncio
import traceback
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

try:
    from .analysis_backend import AnalysisBridge
except ImportError:
    from analysis_backend import AnalysisBridge

app = FastAPI(title='EdgeML Football Backend - Auf Abruf')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], 
    allow_credentials=False, 
    allow_methods=['*'],
    allow_headers=['*'],
)

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / 'videos'

bridge = AnalysisBridge(VIDEO_DIR)

@app.get('/video_feed')
def video_feed():
    print("\n[API - HTTP] /video_feed angefordert. Starte Byte-Uebertragung...")
    return StreamingResponse(bridge.mjpeg_stream(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/api/videos')
def list_videos():
    print("\n[API - HTTP] React fordert Videoliste an. Sende Dateinamen...")
    return bridge.list_videos()

@app.post('/api/control')
def control_playback(action: str, video_name: str = None, seconds: float = 0):
    print(f"\n[API - STEUERUNG] Frontend-Befehl: {action} | Video: {video_name}")
    try:
        if action == 'play': bridge.set_playing(True)
        elif action == 'pause': bridge.set_playing(False)
        elif action == 'select' and video_name: bridge.start_video(video_name)
        elif action == 'seek': bridge.seek(seconds)
        
        telemetry = bridge.snapshot_telemetry()
        return {'status': 'success', 'current_action': action}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "details": str(e)})

@app.websocket('/ws/telemetry')
async def telemetry_websocket(websocket: WebSocket):
    await websocket.accept()
    print("\n[API - WEBSOCKET] Frontend erfolgreich mit Telemetrie verbunden.")
    try:
        while True:
            payload = bridge.snapshot_telemetry()
            await websocket.send_json(payload)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("\n[API - WEBSOCKET] Client getrennt.")