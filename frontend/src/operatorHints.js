export function requestOperatorHint(path, { method = 'GET', phase = 'response' } = {}) {
  const normalizedPath = String(path || '');

  if (normalizedPath === '/api/patients' && method === 'GET') {
    return 'App bootstrap refreshes health, models, and patients together; verify the backend is reachable and returning JSON instead of an HTML or proxy fallback.';
  }

  if (normalizedPath === '/api/patients' && method === 'POST') {
    return 'Sidebar create-patient path depends on a unique patient label; verify the label is not already claimed case-insensitively.';
  }

  if (normalizedPath === '/api/runs' && method === 'POST') {
    return 'PatientPage Create + Start pairs patient_id and report_id; verify the selected report still belongs to the current patient and model ids are non-empty.';
  }

  if (/^\/api\/reports\/[^/]+\/metadata$/.test(normalizedPath)) {
    return 'SourceDataView loads /api/reports/{reportId}/metadata alongside page images; verify metadata.json exists and is valid JSON for the selected report.';
  }

  if (/^\/api\/runs\/[^/]+\/export$/.test(normalizedPath)) {
    return 'Run export reads run.selected_artifact_id and expects a Stage 6 markdown final draft on the same run.';
  }

  if (/^\/api\/runs\/[^/]+\/stream$/.test(normalizedPath)) {
    if (phase === 'parse') {
      return 'RunPage EventSource onmessage expects JSON payloads from /api/runs/{runId}/stream; inspect broker.publish payload serialization if evt.data is not valid JSON.';
    }
    return 'RunPage EventSource listens on /api/runs/{runId}/stream; verify the run still exists and the backend broker can subscribe before the first heartbeat.';
  }

  if (phase === 'parse') {
    return `request(${method} ${normalizedPath}) expected response parsing to succeed after fetch(); inspect the backend response body for truncation or content-type drift.`;
  }

  return `request(${method} ${normalizedPath}) failed before the caller boundary could use the response; inspect the backend route and response for this request.`;
}
