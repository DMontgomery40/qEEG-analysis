import { useMemo, useState } from 'react';
import './Sidebar.css';

function Sidebar({ patients, selectedPatientId, onSelectPatient, onCreatePatient }) {
  const [query, setQuery] = useState('');
  const [newLabel, setNewLabel] = useState('');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return patients;
    return patients.filter((p) => (p.label || '').toLowerCase().includes(q));
  }, [patients, query]);

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-title">qEEG Council</div>
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
      </div>

      <div className="sidebar-list">
        {filtered.map((p) => (
          <button
            key={p.id}
            className={`sidebar-item ${p.id === selectedPatientId ? 'active' : ''}`}
            onClick={() => onSelectPatient(p.id)}
          >
            <div className="sidebar-item-title">{p.label}</div>
            <div className="sidebar-item-sub">{p.id.slice(0, 8)}</div>
          </button>
        ))}
        {!filtered.length ? <div className="sidebar-empty">No patients</div> : null}
      </div>
    </div>
  );
}

export default Sidebar;

