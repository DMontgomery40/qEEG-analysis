const ACTION_DEFINITIONS = [
  {
    id: 'refresh',
    label: 'Refresh state',
    requestLabel: 'refresh',
    successMessage: 'Refreshed orchestration state.',
  },
  {
    id: 'sync_portal',
    label: 'Sync portal',
    requestLabel: 'portal sync',
    successMessage: 'Portal sync requested.',
  },
  {
    id: 'rerun_pipeline',
    label: 'Rerun council batch',
    requestLabel: 'council batch rerun',
    successMessage: 'Council batch rerun requested.',
  },
  {
    id: 'regenerate_patient_facing',
    label: 'Regenerate patient-facing',
    requestLabel: 'patient-facing regeneration',
    successMessage: 'Patient-facing regeneration requested.',
  },
  {
    id: 'prepare_cathode_handoff',
    label: 'Prepare Cathode handoff',
    requestLabel: 'Cathode handoff preparation',
    successMessage: 'Cathode handoff preparation requested.',
  },
  {
    id: 'export_council_artifacts',
    label: 'Export council artifacts',
    requestLabel: 'council export',
    successMessage: 'Council artifacts exported.',
  },
];

const SECTION_DEFINITIONS = [
  { id: 'council', title: 'Council run', aliases: ['council', 'council_run', 'run', 'current_run', 'latest_run'] },
  { id: 'portal', title: 'Portal sync', aliases: ['portal_sync', 'portal', 'sync'] },
  {
    id: 'patientFacing',
    title: 'Patient-facing',
    aliases: ['patient_facing', 'patientFacing', 'patient_pdf', 'patient_report'],
  },
  {
    id: 'worker',
    title: 'Worker / pipeline',
    aliases: ['worker', 'pipeline', 'pipeline_worker', 'job', 'pipeline_job', 'pipeline_state'],
  },
  { id: 'cathode', title: 'Cathode', aliases: ['cathode', 'cathode_handoff', 'handoff'] },
];

function isRecord(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function toArray(value) {
  if (Array.isArray(value)) return value;
  return [];
}

function pickFromSources(sources, keys) {
  for (const source of sources) {
    if (!isRecord(source)) continue;
    for (const key of keys) {
      if (!(key in source)) continue;
      const value = source[key];
      if (value == null) continue;
      if (typeof value === 'string' && !value.trim()) continue;
      return value;
    }
  }
  return null;
}

function pickNumber(sources, keys) {
  const value = pickFromSources(sources, keys);
  if (value == null) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function pickString(sources, keys) {
  const value = pickFromSources(sources, keys);
  if (value == null) return '';
  return String(value).trim();
}

function clampPercent(value) {
  if (!Number.isFinite(value)) return null;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function titleCase(value) {
  return String(value || '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function humanizeStatus(value) {
  if (!value) return 'Unknown';
  return titleCase(value);
}

function shortenId(value) {
  const text = String(value || '').trim();
  if (!text) return '';
  return text.length > 12 ? text.slice(0, 8) : text;
}

function formatDateTime(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function formatElapsed(seconds, rawValue = '') {
  if (Number.isFinite(seconds)) {
    const rounded = Math.max(0, Math.round(seconds));
    const hours = Math.floor(rounded / 3600);
    const minutes = Math.floor((rounded % 3600) / 60);
    const secs = rounded % 60;
    if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  }
  if (!rawValue) return '';
  return String(rawValue);
}

function getStatusTone(...values) {
  const haystack = values
    .filter(Boolean)
    .map((value) => String(value).toLowerCase())
    .join(' ');

  if (!haystack) return 'neutral';
  if (/(fail|error|stalled|blocked|missing|invalid|denied)/.test(haystack)) return 'error';
  if (/(warn|degrad|partial|retry|lag|attention)/.test(haystack)) return 'warning';
  if (/(run|progress|queue|working|syncing|generating|preparing|processing|starting)/.test(haystack)) {
    return 'active';
  }
  if (/(complete|completed|ready|synced|available|healthy|published|done|success)/.test(haystack)) {
    return 'success';
  }
  return 'neutral';
}

function getSources(source) {
  const sources = [];
  if (isRecord(source?.progress)) sources.push(source.progress);
  if (isRecord(source?.progress_state)) sources.push(source.progress_state);
  if (isRecord(source?.progress_summary)) sources.push(source.progress_summary);
  if (isRecord(source?.progress_details)) sources.push(source.progress_details);
  if (isRecord(source?.sync_entry)) sources.push(source.sync_entry);
  if (isRecord(source)) sources.push(source);
  return sources;
}

function getProgressInfo(source) {
  if (!isRecord(source)) {
    return {
      percent: null,
      determinate: false,
      phase: '',
      task: '',
      model: '',
      chunkLabel: '',
      elapsed: '',
      stageNum: null,
      stageName: '',
      statusLabel: '',
      headline: '',
    };
  }

  const sources = getSources(source);
  const statusLabel = pickString(sources, ['status', 'state']);
  const stageNum = pickNumber(sources, ['stage_num', 'current_stage_num', 'stage']);
  const stageName = pickString(sources, ['stage_name', 'current_stage_name']);
  const phase = pickString(sources, ['phase', 'phase_label', 'current_phase', 'step_group']);
  const task = pickString(sources, ['task', 'current_task', 'step', 'current_step', 'activity']);
  const model = pickString(sources, ['model', 'model_id', 'current_model', 'current_model_id']);

  const elapsedSeconds =
    pickNumber(sources, ['elapsed_seconds', 'elapsed_s', 'runtime_seconds', 'duration_seconds']) ??
    (() => {
      const elapsedMs = pickNumber(sources, ['elapsed_ms', 'runtime_ms', 'duration_ms']);
      return Number.isFinite(elapsedMs) ? elapsedMs / 1000 : null;
    })();
  const elapsedRaw = pickString(sources, ['elapsed', 'runtime', 'duration']);
  const elapsed = formatElapsed(elapsedSeconds, elapsedRaw);

  const chunkTotal = pickNumber(sources, ['total_chunks', 'chunks_total', 'chunk_total']);
  const chunkCurrent =
    pickNumber(sources, ['current_chunk', 'chunk_index', 'completed_chunks', 'chunks_completed']) ??
    null;
  const chunkLabel =
    Number.isFinite(chunkCurrent) && Number.isFinite(chunkTotal) && chunkTotal > 0
      ? `${Math.round(chunkCurrent)} of ${Math.round(chunkTotal)}`
      : '';

  const candidatePairs = [
    {
      completeKeys: ['completed', 'completed_steps', 'steps_completed', 'done', 'processed', 'finished'],
      totalKeys: ['total', 'total_steps', 'steps_total'],
    },
    {
      completeKeys: ['current', 'current_step', 'step_index'],
      totalKeys: ['total', 'total_steps', 'steps_total'],
    },
    {
      completeKeys: ['completed_chunks', 'chunks_completed', 'current_chunk', 'chunk_index'],
      totalKeys: ['total_chunks', 'chunks_total', 'chunk_total'],
    },
    {
      completeKeys: ['completed_models', 'models_completed'],
      totalKeys: ['total_models', 'models_total'],
    },
  ];

  let percent = null;
  let determinate = false;

  for (const pair of candidatePairs) {
    const completed = pickNumber(sources, pair.completeKeys);
    const total = pickNumber(sources, pair.totalKeys);
    if (Number.isFinite(completed) && Number.isFinite(total) && total > 0) {
      determinate = true;
      percent = clampPercent((completed / total) * 100);
      break;
    }
  }

  if (!determinate) {
    const explicitPercent = pickNumber(sources, ['percent', 'progress_percent', 'completion_percent']);
    const explicitDeterminacy = pickFromSources(sources, [
      'determinate',
      'is_determinate',
      'has_total',
      'has_real_denominator',
      'percent_is_reliable',
    ]);
    if (
      Number.isFinite(explicitPercent) &&
      (explicitDeterminacy === true || String(explicitDeterminacy).toLowerCase() === 'true')
    ) {
      determinate = true;
      percent = clampPercent(explicitPercent);
    }
  }

  const headlineParts = [];
  for (const part of [stageName && humanizeStatus(stageName), phase && humanizeStatus(phase), task]) {
    if (!part) continue;
    if (headlineParts.includes(part)) continue;
    headlineParts.push(part);
  }
  const headline =
    pickString(sources, ['summary', 'message', 'detail', 'description']) ||
    headlineParts.join(' • ') ||
    humanizeStatus(statusLabel);

  return {
    percent,
    determinate,
    phase,
    task,
    model,
    chunkLabel,
    elapsed,
    stageNum,
    stageName,
    statusLabel,
    headline,
  };
}

function getSectionRaw(orchestration, definition) {
  if (!isRecord(orchestration)) return null;

  const sectionsSource = orchestration.sections;
  if (Array.isArray(sectionsSource)) {
    const arrayMatch = sectionsSource.find((section) => {
      if (!isRecord(section)) return false;
      return [section.id, section.key, section.name].some(
        (value) => definition.aliases.includes(value) || value === definition.id
      );
    });
    if (isRecord(arrayMatch)) return arrayMatch;
  }

  const sectionSources = [];
  if (isRecord(sectionsSource)) sectionSources.push(sectionsSource);
  sectionSources.push(orchestration);

  for (const source of sectionSources) {
    for (const alias of definition.aliases) {
      if (isRecord(source?.[alias])) return source[alias];
    }
  }

  return null;
}

function buildFallbackCouncilSection(runs) {
  const latestRun = getLatestRun(runs);
  if (!latestRun) return null;
  return {
    status: latestRun.status,
    run_id: latestRun.id,
    created_at: latestRun.created_at,
    progress: latestRun.progress,
    progress_state: latestRun.progress_state,
    progress_summary: latestRun.progress_summary,
    progress_details: latestRun.progress_details,
    current_stage_num: latestRun.current_stage_num,
    current_stage_name: latestRun.current_stage_name,
    current_phase: latestRun.current_phase,
    current_task: latestRun.current_task,
    current_model_id: latestRun.current_model_id,
    elapsed_seconds: latestRun.elapsed_seconds,
    error_message: latestRun.error_message,
  };
}

function buildSection(definition, rawValue) {
  const raw = isRecord(rawValue) ? rawValue : null;
  const progress = getProgressInfo(raw || {});
  const sources = raw ? getSources(raw) : [];
  const statusLabel = pickString(sources, ['status', 'state', 'health']) || progress.statusLabel;
  const error = pickString(sources, ['last_error', 'error', 'error_message', 'failure_reason', 'reason']);
  const updatedAt = formatDateTime(
    pickString(sources, ['updated_at', 'last_updated', 'completed_at', 'finished_at', 'started_at', 'timestamp'])
  );
  const runId = pickString(sources, ['run_id', 'current_run_id', 'latest_run_id']);
  const lastSynced = formatDateTime(pickString(sources, ['synced_at', 'last_synced_at']));
  const lastPublished = formatDateTime(pickString(sources, ['published_at', 'prepared_at']));
  const queueDepth = pickNumber(sources, ['queue_depth', 'queued', 'queued_jobs']);
  const summary =
    pickString(sources, ['summary', 'message', 'detail', 'description']) ||
    progress.headline ||
    (raw ? humanizeStatus(statusLabel || definition.title) : 'No data yet');

  const details = [];
  if (progress.phase) details.push({ label: 'Phase', value: humanizeStatus(progress.phase) });
  if (progress.task) details.push({ label: 'Task', value: progress.task });
  if (progress.model) details.push({ label: 'Model', value: progress.model });
  if (progress.stageNum) {
    const stageLabel = progress.stageName
      ? `Stage ${Math.round(progress.stageNum)} • ${humanizeStatus(progress.stageName)}`
      : `Stage ${Math.round(progress.stageNum)}`;
    details.push({ label: 'Stage', value: stageLabel });
  }
  if (progress.chunkLabel) details.push({ label: 'Chunk', value: progress.chunkLabel });
  if (progress.elapsed) details.push({ label: 'Elapsed', value: progress.elapsed });
  if (runId) details.push({ label: 'Run', value: shortenId(runId) });
  if (Number.isFinite(queueDepth)) details.push({ label: 'Queue', value: String(Math.round(queueDepth)) });
  if (lastSynced) details.push({ label: 'Synced', value: lastSynced });
  if (lastPublished) details.push({ label: 'Published', value: lastPublished });
  if (updatedAt) details.push({ label: 'Updated', value: updatedAt });

  return {
    id: definition.id,
    title: definition.title,
    raw,
    summary,
    statusLabel: humanizeStatus(statusLabel || (raw ? 'unknown' : 'unavailable')),
    tone: getStatusTone(statusLabel, summary, error),
    error,
    progress,
    details,
    runId,
    available: Boolean(raw),
  };
}

function getLatestRun(runs) {
  const sorted = [...toArray(runs)].sort((left, right) => {
    return new Date(right.created_at || 0).getTime() - new Date(left.created_at || 0).getTime();
  });
  return sorted[0] || null;
}

function getSidebarSummary(summary) {
  if (!isRecord(summary)) return null;
  const progress = getProgressInfo(summary);
  const label =
    pickString(getSources(summary), ['label', 'summary', 'message']) ||
    [progress.phase && humanizeStatus(progress.phase), progress.task].filter(Boolean).join(' • ') ||
    humanizeStatus(pickString(getSources(summary), ['status', 'state']) || 'status');

  return {
    label,
    tone: getStatusTone(summary.status, summary.state, label),
    percent: progress.determinate ? progress.percent : null,
  };
}

function toActionAliases(actionId) {
  const snake = actionId.replace(/-/g, '_');
  const camel = snake.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
  return [actionId, snake, camel];
}

function getActionState(orchestration, actionId) {
  const sources = [];
  if (isRecord(orchestration?.actions)) sources.push(orchestration.actions);
  if (Array.isArray(orchestration?.actions)) {
    const arrayMatch = orchestration.actions.find((action) => {
      if (!isRecord(action)) return false;
      return toActionAliases(actionId).includes(action.id) || toActionAliases(actionId).includes(action.key);
    });
    if (isRecord(arrayMatch)) return arrayMatch;
  }

  const aliases = toActionAliases(actionId);
  for (const source of sources) {
    for (const alias of aliases) {
      if (isRecord(source[alias])) return source[alias];
      if (source[alias] != null && typeof source[alias] !== 'object') {
        return { enabled: source[alias] };
      }
    }
  }
  return null;
}

export const PATIENT_ACTIONS = ACTION_DEFINITIONS;

export function getPatientSidebarSummary(patient) {
  return getSidebarSummary(patient?.orchestration_summary);
}

export function getPatientOrchestrationSections(orchestration, runs) {
  return SECTION_DEFINITIONS.map((definition) => {
    const raw = getSectionRaw(orchestration, definition);
    if (definition.id === 'council' && !raw) {
      return buildSection(definition, buildFallbackCouncilSection(runs));
    }
    return buildSection(definition, raw);
  });
}

export function getLatestRunProgress(run) {
  return getProgressInfo(run);
}

export function getPatientActionConfig(actionId) {
  return ACTION_DEFINITIONS.find((action) => action.id === actionId) || null;
}

export function getPatientActionState(orchestration, actionId) {
  const state = getActionState(orchestration, actionId);
  if (!state) return { enabled: true, reason: '' };
  const enabled = state.enabled ?? state.available ?? state.allowed ?? true;
  const reason = state.reason || state.message || '';
  return { enabled: Boolean(enabled), reason: String(reason || '') };
}

export function formatStatusDate(value) {
  return formatDateTime(value);
}
