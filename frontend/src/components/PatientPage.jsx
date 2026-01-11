import { useEffect, useMemo, useState } from 'react';
import { api } from '../api';
import './PatientPage.css';

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
  const [uploadPreview, setUploadPreview] = useState('');
  const [uploading, setUploading] = useState(false);

  const [editLabel, setEditLabel] = useState('');
  const [editNotes, setEditNotes] = useState('');

  const [selectedReportId, setSelectedReportId] = useState('');
  const [selectedCouncilIds, setSelectedCouncilIds] = useState([]);
  const [selectedConsolidator, setSelectedConsolidator] = useState('');
  const [starting, setStarting] = useState(false);
  const [geminiProjectId, setGeminiProjectId] = useState('');

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

  async function refresh() {
    if (!patientId) return;
    const [p, r, ru] = await Promise.all([
      api.getPatient(patientId),
      api.listReports(patientId),
      api.listRuns(patientId),
    ]);
    setPatient(p);
    setEditLabel(p.label || '');
    setEditNotes(p.notes || '');
    setReports(r);
    setRuns(ru);
    if (r.length && !selectedReportId) setSelectedReportId(r[0].id);
  }

  useEffect(() => {
    setPatient(null);
    setReports([]);
    setRuns([]);
    setUploadPreview('');
    setSelectedReportId('');
    setSelectedCouncilIds([]);
    setSelectedConsolidator('');
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
      <div className="card">
        <div className="card-title">Patient</div>
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

      <div className="grid">
        <div className="card">
          <div className="card-title">Reports</div>
          <div className="row">
            <input
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
        </div>

        <div className="card">
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

      <div className="card">
        <div className="card-title">Run History</div>
        <div className="list">
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
