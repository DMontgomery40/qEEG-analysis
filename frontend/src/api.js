import { childLogger, errorMessage, newRequestId, serializeError } from './logger';
import { requestOperatorHint } from './operatorHints';

const RAW_API_BASE = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';
const apiLogger = childLogger({ area: 'api' });

function resolveApiBase(base) {
  const raw = String(base || '').trim();
  if (!raw) {
    return typeof window !== 'undefined' ? window.location.origin : 'http://127.0.0.1:8000';
  }
  if (/^[a-z][a-z\d+.-]*:\/\//i.test(raw)) {
    return raw;
  }
  const origin = typeof window !== 'undefined' ? window.location.origin : 'http://127.0.0.1:8000';
  return new URL(raw.startsWith('/') ? raw : `/${raw}`, origin).toString();
}

const API_BASE = resolveApiBase(RAW_API_BASE);

function buildApiUrl(path, { requestId } = {}) {
  const base = new URL(API_BASE.endsWith('/') ? API_BASE : `${API_BASE}/`);
  let relativePath = String(path || '');
  if (!relativePath.startsWith('/')) {
    relativePath = `/${relativePath}`;
  }

  const basePath = base.pathname.replace(/\/$/, '');
  if (basePath && basePath !== '/' && relativePath.startsWith(`${basePath}/`)) {
    relativePath = relativePath.slice(basePath.length);
  }

  const url = new URL(relativePath.replace(/^\//, ''), base);
  if (requestId) {
    url.searchParams.set('request_id', requestId);
  }
  return url;
}

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
    this.operatorHint = details.operatorHint ?? '';
    this.cause = details.cause;
  }
}

async function request(path, options = {}) {
  const method = String(options.method || 'GET').toUpperCase();
  const requestId = newRequestId();
  const startedAt = performance.now();
  const headers = new Headers(options.headers || {});
  const url = buildApiUrl(path, { requestId });
  const log = apiLogger.child({ method, path, requestId });

  try {
    const res = await fetch(url.toString(), { ...options, headers });
    const durationMs = Math.round((performance.now() - startedAt) * 100) / 100;
    const serverRequestId = res.headers.get('x-request-id') || requestId;

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      const operatorHint = requestOperatorHint(path, { method, phase: 'response' });
      const err = new ApiError(text || `Request failed: ${res.status}`, {
        status: res.status,
        path,
        method,
        requestId,
        serverRequestId,
        responseBody: text,
        operatorHint,
      });
      log.warn(
        {
          statusCode: res.status,
          durationMs,
          serverRequestId,
          responseBodyPreview: text.slice(0, 500),
          operatorHint,
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
    if (ct.includes('application/json')) return await res.json();
    return await res.text();
  } catch (error) {
    if (error instanceof ApiError) throw error;

    const durationMs = Math.round((performance.now() - startedAt) * 100) / 100;
    const operatorHint = requestOperatorHint(path, {
      method,
      phase: error instanceof SyntaxError ? 'parse' : 'transport',
    });
    log.error(
      {
        durationMs,
        operatorHint,
        err: serializeError(error),
      },
      'api_request_threw'
    );
    throw new ApiError(errorMessage(error), {
      path,
      method,
      requestId,
      operatorHint,
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
  getPatientOrchestration(patientId) {
    return request(`/api/patients/${patientId}/orchestration`);
  },
  runPatientAction(patientId, action, payload = {}) {
    return request(`/api/patients/${patientId}/actions/${action}`, {
      method: 'POST',
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
    return buildApiUrl(`/api/reports/${reportId}/extracted`).toString();
  },
  reextractReport(reportId) {
    return request(`/api/reports/${reportId}/reextract`, { method: 'POST' });
  },

  reportOriginalUrl(reportId) {
    return buildApiUrl(`/api/reports/${reportId}/original`).toString();
  },
  reportPages(reportId) {
    return request(`/api/reports/${reportId}/pages`);
  },
  reportPageUrl(reportId, pageNum) {
    return buildApiUrl(`/api/reports/${reportId}/pages/${pageNum}`).toString();
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
    return buildApiUrl(`/api/patient_files/${fileId}`).toString();
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
    return buildApiUrl(`/api/runs/${runId}/export/final.md`).toString();
  },
  finalPdfUrl(runId) {
    return buildApiUrl(`/api/runs/${runId}/export/final.pdf`).toString();
  },
  streamUrl(runId, requestId) {
    return buildApiUrl(`/api/runs/${runId}/stream`, { requestId }).toString();
  },
};
