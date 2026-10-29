import { useEffect, useState } from 'react';
import { AppHeader } from './components/AppHeader.jsx';
import { MetricsSection } from './components/MetricsSection.jsx';
import { PlaybackControls } from './components/PlaybackControls.jsx';
import { VideoPlayerPanel } from './components/VideoPlayerPanel.jsx';
import { VideoSidebar } from './components/VideoSidebar.jsx';
import {
  DEFAULT_TELEMETRY,
  STREAM_URL,
  WS_URL,
  fetchVideos,
  sendControlRequest,
} from './services/api.js';
import './styles/main.scss';

export default function App() {
  const [videos, setVideos] = useState([]);
  const [telemetry, setTelemetry] = useState(DEFAULT_TELEMETRY);

  useEffect(() => {
    fetchVideos()
      .then((data) => {
        console.log("[App] Videoliste vom Server empfangen:", data);
        setVideos(data);
      })
      .catch((err) => console.error("[App] Fehler beim Laden der Videoliste:", err));
  }, []);

  useEffect(() => {
    let isMounted = true;
    const host = window.location.hostname || 'localhost';
    const ws = new WebSocket(`ws://${host}:8000/ws/telemetry`);

    ws.onopen = () => {
      console.log("[App] WebSocket erfolgreich mit ML-Bridge verbunden.");
    };

    ws.onmessage = (event) => {
      if (!isMounted) return;
      try {
        const data = JSON.parse(event.data);
        setTelemetry(data);
      } catch (err) {
        console.error("[App] Fehler beim Lesen der WebSocket-Telemetrie:", err);
      }
    };

    ws.onerror = (error) => {
      // Konsolenrauschen im Entwicklungsmodus verhindern
    };

    return () => {
      isMounted = false;
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      } else if (ws.readyState === WebSocket.CONNECTING) {
        ws.onopen = () => ws.close();
      }
    };
  }, []);

  const requestControl = async (action, params = {}) => {
    try {
      await sendControlRequest(action, params);
    } catch (error) {
      console.error(`[App] Fehler bei Steuerungsbefehl ${action}:`, error);
    }
  };

  const handleSelectVideo = (videoName) => {
    void requestControl('select', { video_name: videoName });
  };

  const handleNextVideo = () => {
    if (videos.length === 0) return;
    const currentIndex = videos.indexOf(telemetry.current_video);
    const nextIndex = currentIndex < 0 ? 0 : (currentIndex + 1) % videos.length;
    handleSelectVideo(videos[nextIndex]);
  };

  const handlePrevVideo = () => {
    if (videos.length === 0) return;
    const currentIndex = videos.indexOf(telemetry.current_video);
    const prevIndex = currentIndex <= 0 ? videos.length - 1 : currentIndex - 1;
    handleSelectVideo(videos[prevIndex]);
  };

  const progressPercent = telemetry.total_frames > 0
    ? (telemetry.frame / telemetry.total_frames) * 100
    : 0;

  return (
    <div className="app-shell">
      <AppHeader />

      <main className="app-layout">
        <section className="app-layout__player-column">
          <VideoPlayerPanel streamUrl={STREAM_URL} progressPercent={progressPercent} />

          <PlaybackControls
            isPlaying={telemetry.is_playing}
            onSeek={(seconds) => requestControl('seek', { seconds })}
            onPrev={handlePrevVideo}
            onNext={handleNextVideo}
            onPlay={() => requestControl('play')}
            onPause={() => requestControl('pause')}
          />
        </section>

        <VideoSidebar
          videos={videos}
          currentVideo={telemetry.current_video}
          onSelectVideo={handleSelectVideo}
        />
      </main>

      <MetricsSection telemetry={telemetry} />
    </div>
  );
}