import { useMemo, useRef, useState } from 'react';
import { api } from '../api';
import './BulkUploadPage.css';

function filenameStem(name) {
  const raw = String(name || '').trim();
  if (!raw) return '';
  const lastDot = raw.lastIndexOf('.');
  if (lastDot <= 0) return raw;
  return raw.slice(0, lastDot).trim();
}

function formatBytes(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = n;
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
}

function BulkUploadPage({ patients, onSelectPatient, onClose, onError, onRefreshPatients }) {
  const fileInputRef = useRef(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);

  const existingLabelSet = useMemo(() => {
    const set = new Set();
    for (const p of patients || []) {
      const label = String(p?.label || '').trim().toLowerCase();
      if (label) set.add(label);
    }
    return set;
  }, [patients]);

  const selectionPreview = useMemo(() => {
    const items = [];
    const seen = new Set();
    for (const f of selectedFiles || []) {
      const stem = filenameStem(f?.name);
      const key = stem.toLowerCase();
      const duplicateInBatch = Boolean(key && seen.has(key));
      if (key) seen.add(key);
      const exists = Boolean(key && existingLabelSet.has(key));
      items.push({
        name: f?.name || 'upload',
        size: formatBytes(f?.size),
        patientLabel: stem,
        duplicateInBatch,
        exists,
      });
    }
    return items;
  }, [selectedFiles, existingLabelSet]);

  const warnings = useMemo(() => {
    let empty = 0;
    let dupBatch = 0;
    let exists = 0;
    for (const item of selectionPreview) {
      if (!item.patientLabel) empty += 1;
      if (item.duplicateInBatch) dupBatch += 1;
      if (item.exists) exists += 1;
    }
    return { empty, dupBatch, exists };
  }, [selectionPreview]);

  return (
    <div className="page">
      <div className="card">
        <div className="bulk-header">
          <div className="card-title">Bulk Upload</div>
          <div className="bulk-actions">
            <button onClick={onClose}>Close</button>
          </div>
        </div>

        <div className="muted bulk-help">
          Each file creates a new patient (label = filename without extension) and uploads that file as the
          patient’s qEEG report. Existing patient labels are skipped and reported.
        </div>

        <div className="row bulk-controls">
          <button
            className="primary"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            Choose files…
          </button>
          <div className="muted">
            {selectedFiles.length ? `${selectedFiles.length} selected` : 'No files selected'}
          </div>
          <button
            onClick={async () => {
              if (!selectedFiles.length) return;
              setUploading(true);
              setResult(null);
              try {
                const res = await api.bulkUploadPatients(selectedFiles);
                setResult(res);
                await onRefreshPatients?.();
              } catch (e) {
                onError(String(e?.message || e));
              } finally {
                setUploading(false);
              }
            }}
            disabled={!selectedFiles.length || uploading}
          >
            {uploading ? 'Uploading…' : 'Upload'}
          </button>

          <input
            ref={fileInputRef}
            className="bulk-hidden-input"
            type="file"
            accept=".pdf,.txt,application/pdf,text/plain"
            multiple
            onChange={(e) => {
              const files = Array.from(e.target.files || []);
              setSelectedFiles(files);
              setResult(null);
              // Allow selecting the same file again later
              e.target.value = '';
            }}
          />
        </div>

        {selectedFiles.length ? (
          <div className="bulk-preview">
            <div className="bulk-preview-title">Selected</div>
            {warnings.empty || warnings.dupBatch || warnings.exists ? (
              <div className="bulk-warnings">
                {warnings.empty ? (
                  <div className="warn-banner">Some files have empty filename stems (will error).</div>
                ) : null}
                {warnings.dupBatch ? (
                  <div className="warn-banner">Duplicate filename stems in selection (later ones will skip).</div>
                ) : null}
                {warnings.exists ? (
                  <div className="warn-banner">
                    Some filename stems match existing patient labels (will skip).
                  </div>
                ) : null}
              </div>
            ) : null}
            <div className="list bulk-file-list">
              {selectionPreview.map((item) => (
                <div key={`${item.name}-${item.patientLabel}`} className="bulk-file-row">
                  <div className="bulk-file-name">{item.name}</div>
                  <div className="bulk-file-meta">
                    <span className="muted">{item.size}</span>
                    <span className="muted">→</span>
                    <span className="bulk-file-label">{item.patientLabel || '[invalid]'}</span>
                    {item.exists ? <span className="bulk-tag bulk-tag-skip">exists</span> : null}
                    {item.duplicateInBatch ? <span className="bulk-tag bulk-tag-skip">dup</span> : null}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {result ? (
          <div className="bulk-result">
            <div className="bulk-result-title">Result</div>
            <div className="row bulk-counts">
              <div className="muted">Created: {result?.counts?.created ?? 0}</div>
              <div className="muted">Skipped: {result?.counts?.skipped ?? 0}</div>
              <div className="muted">Errors: {result?.counts?.errors ?? 0}</div>
            </div>

            {result.created?.length ? (
              <>
                <div className="bulk-subtitle">Created</div>
                <div className="list">
                  {result.created.map((c) => (
                    <div key={c.report?.id || c.filename} className="bulk-created-row">
                      <div className="bulk-created-main">
                        <div className="bulk-created-title">{c.patient?.label}</div>
                        <div className="muted">
                          {c.filename} ({c.patient?.id?.slice?.(0, 8)})
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          const id = c.patient?.id;
                          if (!id) return;
                          onSelectPatient?.(id);
                          onClose?.();
                        }}
                      >
                        Open
                      </button>
                    </div>
                  ))}
                </div>
              </>
            ) : null}

            {result.skipped?.length ? (
              <>
                <div className="bulk-subtitle">Skipped</div>
                <div className="list">
                  {result.skipped.map((s, idx) => (
                    <div key={`${s.filename}-${idx}`} className="bulk-skip-row">
                      <div>
                        <div className="bulk-skip-title">{s.patient_label}</div>
                        <div className="muted">
                          {s.filename} — {s.reason}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : null}

            {result.errors?.length ? (
              <>
                <div className="bulk-subtitle">Errors</div>
                <div className="list">
                  {result.errors.map((er, idx) => (
                    <div key={`${er.filename}-${idx}`} className="bulk-error-row">
                      <div>
                        <div className="bulk-skip-title">{er.patient_label || er.filename}</div>
                        <div className="muted">{er.error}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default BulkUploadPage;

