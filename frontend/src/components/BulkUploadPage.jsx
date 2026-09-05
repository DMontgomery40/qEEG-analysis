import { useMemo, useRef, useState } from 'react';
import { api } from '../api';
import './BulkUploadPage.css';

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

const DOB_RE = /^\d{1,2}-\d{1,2}-\d{4}$/;

function isComplete(identity) {
  return Boolean(
    identity
      && String(identity.first_name || '').trim()
      && String(identity.last_name || '').trim()
      && DOB_RE.test(String(identity.birthdate || '').trim()),
  );
}

function clinicId(identity) {
  if (!isComplete(identity)) return '';
  const [mm, dd, yyyy] = String(identity.birthdate).trim().split('-');
  const first = String(identity.first_name).trim()[0].toUpperCase();
  const last = String(identity.last_name).trim()[0].toUpperCase();
  return `${first}${last}_${mm.padStart(2, '0')}-${dd.padStart(2, '0')}-${yyyy}`;
}

function BulkUploadPage({ onSelectPatient, onClose, onError, onRefreshPatients }) {
  const fileInputRef = useRef(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [identities, setIdentities] = useState({});
  const [readings, setReadings] = useState({});
  const [conflicts, setConflicts] = useState({});
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);

  const missingIdentity = useMemo(
    () => selectedFiles.filter((f) => !isComplete(identities[f?.name])).length,
    [selectedFiles, identities],
  );

  const setField = (filename, field, value) => {
    setIdentities((prev) => ({
      ...prev,
      [filename]: { ...(prev[filename] || {}), [field]: value },
    }));
  };

  // The operator answers a name conflict on the file it belongs to, then
  // uploads again. Clearing the conflict takes the prompt off that row.
  const resolveConflict = (filename, resolution) => {
    setIdentities((prev) => ({
      ...prev,
      [filename]: { ...(prev[filename] || {}), ...resolution },
    }));
    setConflicts((prev) => {
      const next = { ...prev };
      delete next[filename];
      return next;
    });
  };

  const readReport = async (file) => {
    setReadings((prev) => ({ ...prev, [file.name]: { loading: true, text: '' } }));
    try {
      const res = await api.previewReport(file);
      setReadings((prev) => ({
        ...prev,
        [file.name]: { loading: false, text: res?.text || res?.preview || '' },
      }));
    } catch (e) {
      setReadings((prev) => ({ ...prev, [file.name]: { loading: false, text: '' } }));
      onError(e, { action: 'preview_report', filename: file.name });
    }
  };

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
          Read each report, then give the patient’s name and date of birth. That is what files the
          report — under the clinic patient id, on the patient already on file when there is one.
          Reading a report costs nothing and files nothing.
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
                const submittedFiles = selectedFiles;
                const payload = submittedFiles.map((f) => ({
                  filename: f.name,
                  ...(identities[f.name] || {}),
                }));
                const res = await api.bulkUploadPatients(submittedFiles, payload);
                const acceptedFiles = new Set(
                  (res?.created || [])
                    .filter((row) => Number.isInteger(row.file_index)
                      && row.file_index >= 0 && row.file_index < submittedFiles.length)
                    .map((row) => submittedFiles[row.file_index]),
                );
                setSelectedFiles((current) => current.filter((file) => !acceptedFiles.has(file)));
                setResult(res);
                setConflicts(
                  Object.fromEntries(
                    (res?.errors || [])
                      .filter((e) => e?.conflict === 'identity_name_mismatch')
                      .map((e) => [e.filename, e]),
                  ),
                );
                await onRefreshPatients?.();
              } catch (e) {
                onError(e, { action: 'bulk_upload_patients', fileCount: selectedFiles.length });
              } finally {
                setUploading(false);
              }
            }}
            disabled={!selectedFiles.length || uploading || missingIdentity > 0}
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
              setIdentities({});
              setReadings({});
              setConflicts({});
              setResult(null);
              // Allow selecting the same file again later
              e.target.value = '';
            }}
          />
        </div>

        {selectedFiles.length ? (
          <div className="bulk-preview">
            <div className="bulk-preview-title">Selected</div>
            {missingIdentity ? (
              <div className="bulk-warnings">
                <div className="warn-banner">
                  {missingIdentity} file(s) still need a name and a date of birth as MM-DD-YYYY.
                </div>
              </div>
            ) : null}
            <div className="list bulk-file-list">
              {selectedFiles.map((file, idx) => {
                const identity = identities[file.name] || {};
                const reading = readings[file.name];
                const conflict = conflicts[file.name];
                return (
                  <div key={`${file.name}-${idx}`} className="bulk-file-row">
                    <div className="bulk-file-name">{file.name}</div>
                    <div className="bulk-file-meta">
                      <span className="muted">{formatBytes(file.size)}</span>
                      <input
                        placeholder="First name"
                        value={identity.first_name || ''}
                        onChange={(e) => setField(file.name, 'first_name', e.target.value)}
                      />
                      <input
                        placeholder="Last name"
                        value={identity.last_name || ''}
                        onChange={(e) => setField(file.name, 'last_name', e.target.value)}
                      />
                      <input
                        placeholder="MM-DD-YYYY"
                        value={identity.birthdate || ''}
                        onChange={(e) => setField(file.name, 'birthdate', e.target.value)}
                      />
                      <span className="bulk-file-label">{clinicId(identity) || '—'}</span>
                      <button onClick={() => readReport(file)} disabled={reading?.loading}>
                        {reading?.loading ? 'Reading…' : 'Read report'}
                      </button>
                    </div>
                    {conflict ? (
                      <div className="bulk-conflict">
                        <div>
                          Already on file with these initials and this date of birth:
                        </div>
                        <div className="bulk-conflict-actions">
                          {(conflict.candidates || []).map((c) => (
                            <button
                              key={c.patient_id}
                              onClick={() =>
                                resolveConflict(file.name, {
                                  attach_to: c.patient_id,
                                  force_new: false,
                                })
                              }
                            >
                              Same as {c.name} ({c.patient_id})
                            </button>
                          ))}
                          <button
                            onClick={() =>
                              resolveConflict(file.name, {
                                attach_to: null,
                                force_new: true,
                              })
                            }
                          >
                            Different person
                          </button>
                        </div>
                      </div>
                    ) : null}
                    {identity.attach_to ? (
                      <div className="muted">Filing under {identity.attach_to}.</div>
                    ) : null}
                    {identity.force_new ? (
                      <div className="muted">Filing as a new patient.</div>
                    ) : null}
                    {reading && !reading.loading && reading.text ? (
                      <pre className="bulk-report-text">{reading.text.slice(0, 4000)}</pre>
                    ) : null}
                  </div>
                );
              })}
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
                        <div className="bulk-created-title">{c.patient?.patient_id}</div>
                        <div className="muted">{c.filename}</div>
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
