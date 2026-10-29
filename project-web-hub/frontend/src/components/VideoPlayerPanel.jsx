import { useState } from 'react';

export function VideoPlayerPanel({ streamUrl, progressPercent }) {
  const [streamKey, setStreamKey] = useState(Date.now());
  const [isStreamLoading, setIsStreamLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleStreamError = () => {
    console.warn("[VideoPlayer] Stream vom Browser unterbrochen. Verbinde in 1.5s neu...");
    setErrorMessage("Verbindung zum Video-Stream getrennt. Automatische Wiederherstellung...");
    setIsStreamLoading(true);
    setTimeout(() => {
      setErrorMessage(null);
      setStreamKey(Date.now());
    }, 1500);
  };

  const handleStreamLoad = () => {
    setIsStreamLoading(false);
    setErrorMessage(null);
  };

  return (
    <div className="video-wrapper" style={{ position: 'relative', backgroundColor: '#0a0f1d', minHeight: '360px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
      {(isStreamLoading || errorMessage) && (
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'rgba(10, 15, 29, 0.85)',
          color: '#ffffff',
          fontFamily: 'monospace',
          fontSize: '0.95rem',
          zIndex: 10,
          padding: '20px',
          textAlign: 'center'
        }}>
          {errorMessage || "ML-Analyse läuft. Stream wird geladen..."}
        </div>
      )}
      
      <img 
        key={streamKey}
        src={`${streamUrl}?k=${streamKey}`} 
        onLoad={handleStreamLoad}
        onError={handleStreamError}
        alt="ML Edge Live-Stream" 
        className="video-stream"
        style={{ width: '100%', height: 'auto', display: 'block' }}
      />
      
      <div className="progress-bar" style={{ width: '100%', backgroundColor: '#1e293b', height: '6px' }}>
        <div className="progress-fill" style={{ width: `${progressPercent}%`, backgroundColor: '#10b981', height: '100%', transition: 'width 0.1s linear' }}></div>
      </div>
    </div>
  );
}