import { useEffect, useState } from 'react';
import { api } from '../api';
import './SourceDataView.css';

function SourceDataView({ reportId, onError }) {
  const [activeTab, setActiveTab] = useState('pdf');
  const [pages, setPages] = useState([]);
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lightboxPage, setLightboxPage] = useState(null);

  useEffect(() => {
    if (!reportId) return;

    async function load() {
      setLoading(true);
      try {
        const [pagesData, metaData] = await Promise.all([
          api.reportPages(reportId),
          api.reportMetadata(reportId).catch(() => null),
        ]);
        setPages(pagesData.pages || []);
        setMetadata(metaData);
      } catch (e) {
        onError?.(String(e?.message || e));
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [reportId]);

  if (!reportId) {
    return <div className="source-data-view">No report selected.</div>;
  }

  return (
    <div className="source-data-view">
      <div className="source-tabs">
        <button
          className={`source-tab ${activeTab === 'pdf' ? 'active' : ''}`}
          onClick={() => setActiveTab('pdf')}
        >
          Original PDF
        </button>
        <button
          className={`source-tab ${activeTab === 'pages' ? 'active' : ''}`}
          onClick={() => setActiveTab('pages')}
        >
          Extracted Pages ({pages.length})
        </button>
        {metadata && (
          <button
            className={`source-tab ${activeTab === 'meta' ? 'active' : ''}`}
            onClick={() => setActiveTab('meta')}
          >
            Metadata
          </button>
        )}
      </div>

      <div className="source-content">
        {activeTab === 'pdf' && <PdfViewer reportId={reportId} />}
        {activeTab === 'pages' && (
          <PageGrid
            reportId={reportId}
            pages={pages}
            loading={loading}
            onSelect={setLightboxPage}
          />
        )}
        {activeTab === 'meta' && metadata && <MetadataView metadata={metadata} />}
      </div>

      {lightboxPage !== null && (
        <Lightbox
          reportId={reportId}
          pages={pages}
          currentPage={lightboxPage}
          onClose={() => setLightboxPage(null)}
          onNavigate={setLightboxPage}
        />
      )}
    </div>
  );
}

function PdfViewer({ reportId }) {
  const pdfUrl = api.reportOriginalUrl(reportId);

  return (
    <div className="pdf-viewer">
      <div className="pdf-viewer-header">
        <a href={pdfUrl} target="_blank" rel="noopener noreferrer" className="pdf-download-link">
          Open in new tab / Download
        </a>
      </div>
      <object data={pdfUrl} type="application/pdf" className="pdf-embed">
        <div className="pdf-fallback">
          <p>Your browser doesn't support inline PDF viewing.</p>
          <a href={pdfUrl} target="_blank" rel="noopener noreferrer">
            Click here to download the PDF
          </a>
        </div>
      </object>
    </div>
  );
}

function PageGrid({ reportId, pages, loading, onSelect }) {
  if (loading) {
    return <div className="page-grid-loading">Loading pages...</div>;
  }

  if (!pages.length) {
    return (
      <div className="page-grid-empty">
        <p>No extracted page images found.</p>
        <p className="muted">Try running Re-extract (OCR) on the report.</p>
      </div>
    );
  }

  return (
    <div className="page-grid">
      {pages.map((p) => (
        <button
          key={p.page}
          className="page-thumbnail"
          onClick={() => onSelect(p.page)}
          title={`Page ${p.page + 1}`}
        >
          <img
            src={api.reportPageUrl(reportId, p.page)}
            alt={`Page ${p.page + 1}`}
            loading="lazy"
          />
          <span className="page-number">Page {p.page + 1}</span>
        </button>
      ))}
    </div>
  );
}

function MetadataView({ metadata }) {
  return (
    <div className="metadata-view">
      <pre>{JSON.stringify(metadata, null, 2)}</pre>
    </div>
  );
}

function Lightbox({ reportId, pages, currentPage, onClose, onNavigate }) {
  const currentIndex = pages.findIndex((p) => p.page === currentPage);
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex < pages.length - 1;

  // Keyboard navigation
  useEffect(() => {
    function handleKey(e) {
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowLeft' && hasPrev) {
        onNavigate(pages[currentIndex - 1].page);
      } else if (e.key === 'ArrowRight' && hasNext) {
        onNavigate(pages[currentIndex + 1].page);
      }
    }

    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [currentPage, pages, hasPrev, hasNext, onClose, onNavigate]);

  return (
    <div className="lightbox-overlay" onClick={onClose}>
      <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
        <div className="lightbox-header">
          <span className="lightbox-title">Page {currentPage + 1} of {pages.length}</span>
          <button className="lightbox-close" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="lightbox-image-container">
          {hasPrev && (
            <button
              className="lightbox-nav lightbox-prev"
              onClick={() => onNavigate(pages[currentIndex - 1].page)}
              aria-label="Previous page"
            >
              &lt;
            </button>
          )}

          <img
            src={api.reportPageUrl(reportId, currentPage)}
            alt={`Page ${currentPage + 1}`}
            className="lightbox-image"
          />

          {hasNext && (
            <button
              className="lightbox-nav lightbox-next"
              onClick={() => onNavigate(pages[currentIndex + 1].page)}
              aria-label="Next page"
            >
              &gt;
            </button>
          )}
        </div>

        <div className="lightbox-footer">
          <span className="muted">Use arrow keys to navigate, Esc to close</span>
        </div>
      </div>
    </div>
  );
}

export default SourceDataView;
