// const host = window.location.hostname || 'localhost';
const host = 'localhost';

export const API_URL = `http://${host}:8000/api`;
export const STREAM_URL = `http://${host}:8000/video_feed`;
export const WS_URL = `ws://${host}:8000/ws/telemetry`;

export const DEFAULT_TELEMETRY = {
  current_video: '',
  is_playing: false,
  frame: 0,
  total_frames: 0,
  fps: 0,
  power_watts: 0,
  models: {
    player_detection: { inference_ms: 0, count: 0 },
    homography: { inference_ms: 0, status: 'N/A' },
    readability: { inference_ms: 0, readable_count: 0 },
    ocr: { inference_ms: 0, texts: [] },
    formation: { inference_ms: 0, prediction: 'N/A', confidence: 0 },
  },
};

export async function fetchVideos() {
  const response = await fetch(`${API_URL}/videos`);

  if (!response.ok) {
    throw new Error(`No se pudo cargar la lista de videos (${response.status})`);
  }

  return response.json();
}

export async function sendControlRequest(action, params = {}) {
  const queryString = new URLSearchParams({ action, ...params }).toString();
  const response = await fetch(`${API_URL}/control?${queryString}`, { method: 'POST' });

  if (!response.ok) {
    throw new Error(`No se pudo enviar la acción ${action} (${response.status})`);
  }

  return response;
}