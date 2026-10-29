import { useEffect, useMemo, useState } from 'react';

const PAGE_SIZE = 8;

export function VideoSidebar({ videos, currentVideo, onSelectVideo }) {
  const [page, setPage] = useState(1);

  const totalPages = Math.max(1, Math.ceil(videos.length / PAGE_SIZE));

  const currentVideoPage = useMemo(() => {
    const currentIndex = videos.indexOf(currentVideo);
    if (currentIndex < 0) return 1;
    return Math.floor(currentIndex / PAGE_SIZE) + 1;
  }, [currentVideo, videos]);

  useEffect(() => {
    setPage(currentVideoPage);
  }, [currentVideoPage]);

  const safePage = Math.min(page, totalPages);
  const startIndex = (safePage - 1) * PAGE_SIZE;
  const pageVideos = videos.slice(startIndex, startIndex + PAGE_SIZE);

  const goToPage = (nextPage) => {
    const boundedPage = Math.min(Math.max(nextPage, 1), totalPages);
    setPage(boundedPage);
  };

  return (
    <aside className="video-sidebar">
      <div className="video-sidebar__header">
        <h2 className="video-sidebar__title">Verfuegbare Spielzuege ({videos.length})</h2>
        <div className="video-sidebar__pager">
          <button
            type="button"
            className="video-sidebar__pager-button"
            onClick={() => goToPage(safePage - 1)}
            disabled={safePage <= 1}
            style={{ touchAction: 'manipulation' }}
          >
            ◀
          </button>
          <span className="video-sidebar__pager-label">
            {safePage} / {totalPages}
          </span>
          <button
            type="button"
            className="video-sidebar__pager-button"
            onClick={() => goToPage(safePage + 1)}
            disabled={safePage >= totalPages}
            style={{ touchAction: 'manipulation' }}
          >
            ▶
          </button>
        </div>
      </div>

      <div className="video-sidebar__list">
        {pageVideos.map((videoName) => (
          <button
            type="button"
            key={videoName}
            className={`video-sidebar__item${currentVideo === videoName ? ' video-sidebar__item--active' : ''}`}
            onClick={() => onSelectVideo(videoName)}
            style={{ touchAction: 'manipulation' }}
          >
            {videoName}
          </button>
        ))}

        {pageVideos.length === 0 ? (
          <div className="video-sidebar__empty">Keine Videos zum Anzeigen.</div>
        ) : null}
      </div>
    </aside>
  );
}