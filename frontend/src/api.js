const API_BASE = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(text || `Request failed: ${res.status}`);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res.text();
}

export const api = {
  health() {
    return request('/api/health');
  },
  models() {
    return request('/api/models');
  },
  cliproxyStart(payload = {}) {
    return request('/api/cliproxy/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },
  cliproxyLogin(payload = { mode: 'login' }) {
    return request('/api/cliproxy/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },
  cliproxyInstall(payload = {}) {
    return request('/api/cliproxy/install', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  listPatients() {
    return request('/api/patients');
  },
  createPatient(payload) {
    return request('/api/patients', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },
  getPatient(patientId) {
    return request(`/api/patients/${patientId}`);
  },
  updatePatient(patientId, payload) {
    return request(`/api/patients/${patientId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  listReports(patientId) {
    return request(`/api/patients/${patientId}/reports`);
  },
  async uploadReport(patientId, file) {
    const form = new FormData();
    form.append('file', file);
    return request(`/api/patients/${patientId}/reports`, { method: 'POST', body: form });
  },
  reportExtractedUrl(reportId) {
    return `${API_BASE}/api/reports/${reportId}/extracted`;
  },
  reextractReport(reportId) {
    return request(`/api/reports/${reportId}/reextract`, { method: 'POST' });
  },

  listRuns(patientId) {
    return request(`/api/patients/${patientId}/runs`);
  },
  createRun(payload) {
    return request('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },
  startRun(runId) {
    return request(`/api/runs/${runId}/start`, { method: 'POST' });
  },
  getRun(runId) {
    return request(`/api/runs/${runId}`);
  },
  getArtifacts(runId) {
    return request(`/api/runs/${runId}/artifacts`);
  },

  selectFinal(runId, payload) {
    return request(`/api/runs/${runId}/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  exportRun(runId) {
    return request(`/api/runs/${runId}/export`, { method: 'POST' });
  },
  finalMdUrl(runId) {
    return `${API_BASE}/api/runs/${runId}/export/final.md`;
  },
  finalPdfUrl(runId) {
    return `${API_BASE}/api/runs/${runId}/export/final.pdf`;
  },
  streamUrl(runId) {
    return `${API_BASE}/api/runs/${runId}/stream`;
  },
};
