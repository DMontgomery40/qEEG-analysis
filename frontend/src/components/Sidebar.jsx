import { useMemo, useState } from 'react';
import './Sidebar.css';
import { getPatientSidebarSummary } from '../orchestration';

// EEG Wave Icon Component
function EEGWaveIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 40 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M2 12 L8 12 L10 4 L14 20 L18 8 L22 16 L26 6 L30 18 L34 12 L38 12"
        stroke="url(#wave-gradient)"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <defs>
        <linearGradient id="wave-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#00CED1" />
          <stop offset="50%" stopColor="#9370DB" />
          <stop offset="100%" stopColor="#4A90D9" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function Sidebar({ patients, selectedPatientId, onSelectPatient, onCreatePatient, onBulkUpload, style }) {
  const [query, setQuery] = useState('');
  const [newLabel, setNewLabel] = useState('');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return patients;
    return patients.filter((p) => (p.label || '').toLowerCase().includes(q));
  }, [patients, query]);

  return (
    <div className="sidebar" style={style}>
      <div className="sidebar-header">
        <div className="sidebar-logo-container">
          <EEGWaveIcon className="sidebar-logo-icon" />
          <div className="sidebar-logo-text-group">
            <div className="sidebar-logo-text">qEEG Council</div>
            <div className="sidebar-tagline">Multi-Model Analysis</div>
          </div>
        </div>
        <input
          className="sidebar-search"
          placeholder="Search patients…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="sidebar-new">
          <input
            className="sidebar-input"
            placeholder="New patient label…"
            value={newLabel}
            onChange={(e) => setNewLabel(e.target.value)}
          />
          <button
            className="sidebar-button"
            onClick={() => {
              const label = newLabel.trim();
              if (!label) return;
              setNewLabel('');
              onCreatePatient({ label, notes: '' });
            }}
          >
            New
          </button>
        </div>
        <div className="sidebar-bulk">
          <button
            className="sidebar-bulk-button"
            onClick={() => {
              onBulkUpload?.();
            }}
          >
            Bulk Upload
          </button>
        </div>
      </div>

      <div className="sidebar-list">
        {filtered.map((p) => (
          <SidebarPatientItem
            key={p.id}
            patient={p}
            active={p.id === selectedPatientId}
            onSelectPatient={onSelectPatient}
          />
        ))}
        {!filtered.length ? <div className="sidebar-empty">No patients</div> : null}
      </div>
    </div>
  );
}

function SidebarPatientItem({ patient, active, onSelectPatient }) {
  const summary = getPatientSidebarSummary(patient);

  return (
    <button
      className={`sidebar-item ${active ? 'active' : ''}`}
      onClick={() => onSelectPatient(patient.id)}
    >
      <div className="sidebar-item-main">
        <div className="sidebar-item-title">{patient.label}</div>
        <div className="sidebar-item-sub">{patient.id.slice(0, 8)}</div>
        {summary ? (
          <div className="sidebar-item-status">
            <div className="sidebar-item-status-row">
              <span className={`sidebar-status-dot ${summary.tone}`} />
              <span className="sidebar-item-status-label">{summary.label}</span>
              {summary.percent != null ? (
                <span className="sidebar-item-status-percent">{summary.percent}%</span>
              ) : null}
            </div>
            {summary.percent != null ? (
              <div className="sidebar-mini-progress" aria-hidden="true">
                <div className="sidebar-mini-progress-fill" style={{ width: `${summary.percent}%` }} />
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
      <div className="sidebar-item-badges">
        {patient.has_explainer_video ? (
          <div className="sidebar-badge sidebar-badge-mp4" title="Explainer video uploaded">
            MP4
          </div>
        ) : null}
      </div>
    </button>
  );
}

export default Sidebar;
