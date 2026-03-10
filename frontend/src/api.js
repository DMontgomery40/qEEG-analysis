import { childLogger, errorMessage, newRequestId, serializeError } from './logger';

const API_BASE = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';
const apiLogger = childLogger({ area: 'api' });

export class ApiError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = details.status ?? null;
    this.path = details.path ?? '';
    this.method = details.method ?? 'GET';
    this.requestId = details.requestId ?? '';
    this.serverRequestId = details.serverRequestId ?? '';
    this.responseBody = details.responseBody ?? '';
    this.cause = details.cause;
  }
}

async function request(path, options = {}) {
  const method = String(options.method || 'GET').toUpperCase();
  const requestId = newRequestId();
  const startedAt = performance.now();
  const headers = new Headers(options.headers || {});
  headers.set('X-Request-ID', requestId);
  const log = apiLogger.child({ method, path, requestId });

  try {
    const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
    const durationMs = Math.round((performance.now() - startedAt) * 100) / 100;
    const serverRequestId = res.headers.get('x-request-id') || requestId;

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      const err = new ApiError(text || `Request failed: ${res.status}`, {
        status: res.status,
        path,
        method,
        requestId,
        serverRequestId,
        responseBody: text,
      });
      log.warn(
        {
          statusCode: res.status,
          durationMs,
          serverRequestId,
          responseBodyPreview: text.slice(0, 500),
        },
        'api_request_failed'
      );
      throw err;
    }

    log.debug(
      {
        statusCode: res.status,
        durationMs,
        serverRequestId,
      },
      'api_request_succeeded'
    );

    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) return res.json();
    return res.text();
  } catch (error) {
    if (error instanceof ApiError) throw error;

    const durationMs = Math.round((performance.now() - startedAt) * 100) / 100;
    log.error(
      {
        durationMs,
        err: serializeError(error),
      },
      'api_request_threw'
    );
    throw new ApiError(errorMessage(error), {
      path,
      method,
      requestId,
      cause: error,
    });
  }
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

  reportOriginalUrl(reportId) {
    return `${API_BASE}/api/reports/${reportId}/original`;
  },
  reportPages(reportId) {
    return request(`/api/reports/${reportId}/pages`);
  },
  reportPageUrl(reportId, pageNum) {
    return `${API_BASE}/api/reports/${reportId}/pages/${pageNum}`;
  },
  reportMetadata(reportId) {
    return request(`/api/reports/${reportId}/metadata`);
  },

  listRuns(patientId) {
    return request(`/api/patients/${patientId}/runs`);
  },
  async bulkUploadPatients(files) {
    const form = new FormData();
    for (const f of files || []) form.append('files', f);
    return request('/api/patients/bulk_upload', { method: 'POST', body: form });
  },

  listPatientFiles(patientId) {
    return request(`/api/patients/${patientId}/files`);
  },
  async uploadPatientFile(patientId, file) {
    const form = new FormData();
    form.append('file', file);
    return request(`/api/patients/${patientId}/files`, { method: 'POST', body: form });
  },
  patientFileUrl(fileId) {
    return `${API_BASE}/api/patient_files/${fileId}`;
  },
  deletePatientFile(fileId) {
    return request(`/api/patient_files/${fileId}`, { method: 'DELETE' });
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
  streamUrl(runId, requestId) {
    const url = new URL(`${API_BASE}/api/runs/${runId}/stream`);
    if (requestId) url.searchParams.set('request_id', requestId);
    return url.toString();
  },
};
