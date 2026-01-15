import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api';
import './PatientPage.css';
import ResizeHandle from './ResizeHandle';

// Panel size persistence
const STORAGE_KEY = 'qeeg-patient-panel-sizes';
const DEFAULT_LEFT_COL_PERCENT = 50;
const MIN_COL_PERCENT = 25;
const MAX_COL_PERCENT = 75;
const DEFAULT_RUN_HISTORY_HEIGHT = 280;
const MIN_RUN_HISTORY_HEIGHT = 120;
const MAX_RUN_HISTORY_HEIGHT = 600;

function loadPanelSizes() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch {
    return {};
  }
}

function savePanelSizes(sizes) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sizes));
  } catch {
    // ignore
  }
}

function PatientPage({
  patientId,
  discoveredModels,
  modelMetaById,
  defaultConsolidator,
  onSelectRun,
  onRefreshGlobal,
  onError,
}) {
  const [patient, setPatient] = useState(null);
  const [reports, setReports] = useState([]);
  const [runs, setRuns] = useState([]);
  const [patientFiles, setPatientFiles] = useState([]);
  const [uploadPreview, setUploadPreview] = useState('');
  const [uploading, setUploading] = useState(false);
  const [fileUploading, setFileUploading] = useState(false);
  const [videoModalFile, setVideoModalFile] = useState(null);

  const reportInputRef = useRef(null);
  const patientFileInputRef = useRef(null);

  const [editLabel, setEditLabel] = useState('');
  const [editNotes, setEditNotes] = useState('');

  const [selectedReportId, setSelectedReportId] = useState('');
  const [selectedCouncilIds, setSelectedCouncilIds] = useState([]);
  const [selectedConsolidator, setSelectedConsolidator] = useState('');
  const [starting, setStarting] = useState(false);
  const [geminiProjectId, setGeminiProjectId] = useState('');

  // Resizable panel state
  const [leftColPercent, setLeftColPercent] = useState(() => {
    const sizes = loadPanelSizes();
    return sizes.leftColPercent ?? DEFAULT_LEFT_COL_PERCENT;
  });
  const [runHistoryHeight, setRunHistoryHeight] = useState(() => {
    const sizes = loadPanelSizes();
    return sizes.runHistoryHeight ?? DEFAULT_RUN_HISTORY_HEIGHT;
  });
  const [gridContainerWidth, setGridContainerWidth] = useState(0);
  const gridContainerRef = useRef(null);

  // Measure grid container width for percentage-based column resizing
  useEffect(() => {
    const el = gridContainerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setGridContainerWidth(entry.contentRect.width);
      }
    });
    observer.observe(el);
    // Initial measurement
    setGridContainerWidth(el.offsetWidth);
    return () => observer.disconnect();
  }, []);

  const handleColumnResize = useCallback((delta) => {
    if (gridContainerWidth <= 0) return;
    const percentDelta = (delta / gridContainerWidth) * 100;
    setLeftColPercent((prev) => {
      const next = Math.max(MIN_COL_PERCENT, Math.min(MAX_COL_PERCENT, prev + percentDelta));
      return next;
    });
  }, [gridContainerWidth]);

  const handleColumnResizeEnd = useCallback(() => {
    savePanelSizes({ ...loadPanelSizes(), leftColPercent });
  }, [leftColPercent]);

  const handleRunHistoryResize = useCallback((delta) => {
    setRunHistoryHeight((prev) => {
      // Negative delta means dragging up (making taller), positive means dragging down (making shorter)
      const next = Math.max(MIN_RUN_HISTORY_HEIGHT, Math.min(MAX_RUN_HISTORY_HEIGHT, prev - delta));
      return next;
    });
  }, []);

  const handleRunHistoryResizeEnd = useCallback(() => {
    savePanelSizes({ ...loadPanelSizes(), runHistoryHeight });
  }, [runHistoryHeight]);

  const discoveredOptions = useMemo(
    () =>
      discoveredModels.map((id) => {
        const meta = modelMetaById?.get?.(id);
        const label = meta?.name ? `${meta.name} (${id})` : id;
        return { id, label };
      }),
    [discoveredModels, modelMetaById]
  );

  const hasGemini = useMemo(
    () => (discoveredModels || []).some((id) => String(id).toLowerCase().includes('gemini')),
    [discoveredModels]
  );

  const explainerVideo = useMemo(() => {
    for (const f of patientFiles || []) {
      const mt = String(f?.mime_type || '').toLowerCase();
      const name = String(f?.filename || '').toLowerCase();
      if (mt === 'video/mp4' || name.endsWith('.mp4')) return f;
    }
    return null;
  }, [patientFiles]);

  useEffect(() => {
    if (!videoModalFile) return;
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setVideoModalFile(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [videoModalFile]);

  async function refresh() {
    if (!patientId) return;
    const [p, r, ru, pf] = await Promise.all([
      api.getPatient(patientId),
      api.listReports(patientId),
      api.listRuns(patientId),
      api.listPatientFiles(patientId),
    ]);
    setPatient(p);
    setEditLabel(p.label || '');
    setEditNotes(p.notes || '');
    setReports(r);
    setRuns(ru);
    setPatientFiles(pf);
    if (r.length && !selectedReportId) setSelectedReportId(r[0].id);
  }

  useEffect(() => {
    setPatient(null);
    setReports([]);
    setRuns([]);
    setPatientFiles([]);
    setUploadPreview('');
    setSelectedReportId('');
    setSelectedCouncilIds([]);
    setSelectedConsolidator('');
    setVideoModalFile(null);
    (async () => {
      try {
        await refresh();
      } catch (e) {
        onError(String(e?.message || e));
      }
    })();
  }, [patientId]);

  useEffect(() => {
    if (!selectedConsolidator) setSelectedConsolidator(defaultConsolidator || '');
  }, [defaultConsolidator, selectedConsolidator]);

  if (!patientId) return <div className="page">Select or create a patient.</div>;
  if (!patient) return <div className="page">Loading…</div>;

  return (
    <div className="page">
      {videoModalFile ? (
        <div
          className="video-modal-overlay"
          onClick={(e) => {
            if (e.target === e.currentTarget) setVideoModalFile(null);
          }}
        >
          <div className="video-modal">
            <div className="video-modal-header">
              <div className="video-modal-title">Explainer Video</div>
              <div className="video-modal-actions">
                <button onClick={() => window.open(api.patientFileUrl(videoModalFile.id), '_blank')}>
                  Open
                </button>
                <button onClick={() => setVideoModalFile(null)}>Close</button>
              </div>
            </div>
            <div className="video-modal-body">
              <video
                className="video-modal-player"
                controls
                autoPlay
                crossOrigin="anonymous"
                src={api.patientFileUrl(videoModalFile.id)}
              />
            </div>
          </div>
        </div>
      ) : null}

      <div className="card">
        <div className="card-title-row">
          <div className="card-title">Patient</div>
          {explainerVideo ? (
            <div className="explainer-banner">
              <div className="explainer-label">Explainer Video</div>
              <button
                className="primary explainer-play"
                onClick={() => setVideoModalFile(explainerVideo)}
              >
                Play
              </button>
            </div>
          ) : null}
        </div>
        <div className="row">
          <label>
            Label
            <input value={editLabel} onChange={(e) => setEditLabel(e.target.value)} />
          </label>
          <button
            onClick={async () => {
              try {
                await api.updatePatient(patientId, { label: editLabel, notes: editNotes });
                await refresh();
                await onRefreshGlobal();
              } catch (e) {
                onError(String(e?.message || e));
              }
            }}
          >
            Save
          </button>
        </div>
        <label>
          Notes
          <textarea value={editNotes} onChange={(e) => setEditNotes(e.target.value)} />
        </label>
      </div>

      <div className="grid-row" ref={gridContainerRef}>
        <div className="card" style={{ width: `${leftColPercent}%`, flexShrink: 0 }}>
          <div className="card-title">Reports</div>
          <div className="row">
            <button
              className="primary"
              onClick={() => reportInputRef.current?.click()}
              disabled={uploading}
            >
              {uploading ? 'Uploading…' : 'Upload report…'}
            </button>
            <input
              ref={reportInputRef}
              className="hidden-file-input"
              type="file"
              accept=".pdf,.txt,text/plain,application/pdf"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                if (!file) return;
                setUploading(true);
                try {
                  const res = await api.uploadReport(patientId, file);
                  setUploadPreview(res.preview || '');
                  await refresh();
                } catch (err) {
                  onError(String(err?.message || err));
                } finally {
                  setUploading(false);
                  e.target.value = '';
                }
              }}
              disabled={uploading}
            />
            {selectedReportId ? (
              <>
                <button
                  onClick={() => {
                    window.open(api.reportExtractedUrl(selectedReportId), '_blank');
                  }}
                >
                  View extracted
                </button>
                <button
                  onClick={async () => {
                    try {
                      await api.reextractReport(selectedReportId);
                      onError('Re-extracted report text (OCR if available). Refreshing models…');
                      await refresh();
                      await onRefreshGlobal();
                    } catch (e) {
                      onError(String(e?.message || e));
                    }
                  }}
                >
                  Re-extract (OCR)
                </button>
              </>
            ) : null}
          </div>

          {uploadPreview ? (
            <div className="preview">
              <div className="preview-title">Extracted preview</div>
              <pre>{uploadPreview}</pre>
            </div>
          ) : null}

          <div className="list">
            {reports.map((r) => (
              <button
                key={r.id}
                className={`list-item ${selectedReportId === r.id ? 'active' : ''}`}
                onClick={() => setSelectedReportId(r.id)}
              >
                <div className="list-item-title">{r.filename}</div>
                <div className="list-item-sub">{new Date(r.created_at).toLocaleString()}</div>
              </button>
            ))}
            {!reports.length ? <div className="muted">No reports yet.</div> : null}
          </div>

          <div className="section-divider" />
          <div className="subsection-title">Patient Files</div>
          <div className="muted" style={{ fontSize: 13 }}>
            PDFs, Markdown, and MP4 videos (e.g. explainer videos).
          </div>

          <div className="row" style={{ marginTop: 10 }}>
            <button onClick={() => patientFileInputRef.current?.click()} disabled={fileUploading}>
              {fileUploading ? 'Uploading…' : 'Upload file…'}
            </button>
            <input
              ref={patientFileInputRef}
              className="hidden-file-input"
              type="file"
              accept=".pdf,.md,.markdown,.mp4,application/pdf,text/markdown,video/mp4"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                if (!file) return;
                setFileUploading(true);
                try {
                  await api.uploadPatientFile(patientId, file);
                  await refresh();
                  await onRefreshGlobal?.();
                } catch (err) {
                  onError(String(err?.message || err));
                } finally {
                  setFileUploading(false);
                  e.target.value = '';
                }
              }}
              disabled={fileUploading}
            />
          </div>

          <div className="list">
            {patientFiles.map((f) => (
              <div key={f.id} className="file-row">
                <button
                  className="list-item file-open"
                  onClick={() => window.open(api.patientFileUrl(f.id), '_blank')}
                >
                  <div className="list-item-title">{f.filename}</div>
                  <div className="list-item-sub">
                    {f.mime_type}
                    {typeof f.size_bytes === 'number' && f.size_bytes > 0
                      ? ` • ${Math.round(f.size_bytes / 1024)} KB`
                      : ''}
                    {' • '}
                    {new Date(f.created_at).toLocaleString()}
                  </div>
                </button>
                <button
                  className="file-delete"
                  onClick={async () => {
                    const ok = window.confirm(`Delete "${f.filename}"?`);
                    if (!ok) return;
                    try {
                      await api.deletePatientFile(f.id);
                      await refresh();
                      await onRefreshGlobal?.();
                    } catch (err) {
                      onError(String(err?.message || err));
                    }
                  }}
                >
                  Delete
                </button>
              </div>
            ))}
            {!patientFiles.length ? <div className="muted">No files yet.</div> : null}
          </div>
        </div>

        <ResizeHandle
          direction="horizontal"
          onResize={handleColumnResize}
          onResizeEnd={handleColumnResizeEnd}
        />

        <div className="card" style={{ flex: 1, minWidth: 0 }}>
          <div className="card-title">New Run</div>
          <label>
            Report
            <select value={selectedReportId} onChange={(e) => setSelectedReportId(e.target.value)}>
              <option value="" disabled>
                Select…
              </option>
              {reports.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.filename} ({r.id.slice(0, 8)})
                </option>
              ))}
            </select>
          </label>

          <label>
            Council models (multi-select)
            <select
              multiple
              className="multi-select"
              value={selectedCouncilIds}
              onChange={(e) => {
                const opts = Array.from(e.target.selectedOptions).map((o) => o.value);
                setSelectedCouncilIds(opts);
                if (!opts.includes(selectedConsolidator)) setSelectedConsolidator(opts[0] || '');
              }}
            >
              {discoveredOptions.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Consolidator
            <select
              value={selectedConsolidator}
              onChange={(e) => setSelectedConsolidator(e.target.value)}
            >
              <option value="" disabled>
                Select…
              </option>
              {(selectedCouncilIds.length ? selectedCouncilIds : discoveredModels).map((id) => (
                <option key={id} value={id}>
                  {modelMetaById?.get?.(id)?.name ? `${modelMetaById.get(id).name} (${id})` : id}
                </option>
              ))}
            </select>
          </label>

          <button
            disabled={
              starting ||
              !selectedReportId ||
              selectedCouncilIds.length === 0 ||
              !selectedConsolidator
            }
            onClick={async () => {
              setStarting(true);
              try {
                const run = await api.createRun({
                  patient_id: patientId,
                  report_id: selectedReportId,
                  council_model_ids: selectedCouncilIds,
                  consolidator_model_id: selectedConsolidator,
                });
                await api.startRun(run.id);
                await refresh();
                onSelectRun(run.id);
              } catch (e) {
                onError(String(e?.message || e));
              } finally {
                setStarting(false);
              }
            }}
          >
            Create + Start
          </button>

          <div style={{ marginTop: 12 }}>
            {!hasGemini ? (
              <div className="muted" style={{ marginBottom: 8 }}>
                No Gemini models detected. Run Google/Gemini login to enable them.
              </div>
            ) : (
              <div className="muted" style={{ marginBottom: 8 }}>
                Need to re-auth Gemini? Run Google/Gemini login.
              </div>
            )}
            <button
              onClick={async () => {
                try {
                  await api.cliproxyLogin({
                    mode: 'gemini',
                    project_id: geminiProjectId.trim() || null,
                  });
                  onError(
                    'Gemini login launched. Complete it in the opened browser window, then click Refresh.'
                  );
                } catch (e) {
                  onError(String(e?.message || e));
                }
              }}
            >
              Gemini Login
            </button>
            <input
              className="inline-input"
              placeholder="project_id (optional)"
              value={geminiProjectId}
              onChange={(e) => setGeminiProjectId(e.target.value)}
            />
            <button
              style={{ marginLeft: 10 }}
              onClick={async () => {
                try {
                  await onRefreshGlobal();
                  await refresh();
                } catch (e) {
                  onError(String(e?.message || e));
                }
              }}
            >
              Refresh
            </button>
          </div>
        </div>
      </div>

      <ResizeHandle
        direction="vertical"
        onResize={handleRunHistoryResize}
        onResizeEnd={handleRunHistoryResizeEnd}
      />

      <div className="card run-history-card" style={{ height: runHistoryHeight, flexShrink: 0 }}>
        <div className="card-title">Run History</div>
        <div className="list run-history-list">
          {runs.map((r) => (
            <button key={r.id} className="list-item" onClick={() => onSelectRun(r.id)}>
              <div className="list-item-title">
                {r.id.slice(0, 8)} — {r.status}
              </div>
              <div className="list-item-sub">{new Date(r.created_at).toLocaleString()}</div>
            </button>
          ))}
          {!runs.length ? <div className="muted">No runs yet.</div> : null}
        </div>
      </div>
    </div>
  );
}

export default PatientPage;
