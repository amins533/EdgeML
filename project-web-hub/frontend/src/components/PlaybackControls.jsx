export function PlaybackControls({ isPlaying, onSeek, onPrev, onNext, onPlay, onPause }) {
  const btnStyle = { touchAction: 'manipulation', cursor: 'pointer', userSelect: 'none' };

  return (
    <div className="playback-controls">
      <button type="button" className="playback-controls__button" style={btnStyle} onClick={() => onSeek(-10)}>
        -10 s
      </button>
      <button type="button" className="playback-controls__button" style={btnStyle} onClick={onPrev}>
        Zurück
      </button>

      {isPlaying ? (
        <button
          type="button"
          className="playback-controls__button playback-controls__button--danger"
          style={btnStyle}
          onClick={onPause}
        >
          Pause
        </button>
      ) : (
        <button
          type="button"
          className="playback-controls__button playback-controls__button--success"
          style={btnStyle}
          onClick={onPlay}
        >
          Abspielen
        </button>
      )}

      <button type="button" className="playback-controls__button" style={btnStyle} onClick={onNext}>
        Weiter
      </button>
      <button type="button" className="playback-controls__button" style={btnStyle} onClick={() => onSeek(10)}>
        +10 s
      </button>
    </div>
  );
}