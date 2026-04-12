import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api';
import { childLogger, serializeError, newRequestId } from '../logger';
import { requestOperatorHint } from '../operatorHints';
import { getLatestRunProgress } from '../orchestration';
import Tabs from './Tabs';
import ModelBadge from './ModelBadge';
import SourceDataView from './SourceDataView';
import './RunPage.css';

const STAGES = [
  { num: 1, name: 'initial_analysis', kind: 'analysis', type: 'md' },
  { num: 2, name: 'peer_review', kind: 'peer_review', type: 'json' },
  { num: 3, name: 'revision', kind: 'revision', type: 'md' },
  { num: 4, name: 'consolidation', kind: 'consolidation', type: 'md' },
  { num: 5, name: 'final_review', kind: 'final_review', type: 'json' },
  { num: 6, name: 'final_draft', kind: 'final_draft', type: 'md' },
];
const runPageLogger = childLogger({ area: 'run_page' });

function groupByStage(artifacts) {
  const by = new Map();
  for (const a of artifacts) {
    const key = `${a.stage_num}`;
    if (!by.has(key)) by.set(key, []);
    by.get(key).push(a);
  }
  return by;
}

function tryJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function RunPage({ runId, modelMetaById, onBack, onError }) {
  const [run, setRun] = useState(null);
  const [artifacts, setArtifacts] = useState([]);
  const [exporting, setExporting] = useState(false);
  const [activeStageNum, setActiveStageNum] = useState(1);
  const [showSourceData, setShowSourceData] = useState(false);
  const esRef = useRef(null);
  const initializedStageRef = useRef(false);

  const refresh = useCallback(async () => {
    const [r, a] = await Promise.all([api.getRun(runId), api.getArtifacts(runId)]);
    setRun(r);
    setArtifacts(a);
  }, [runId]);

  useEffect(() => {
    setRun(null);
    setArtifacts([]);
    setActiveStageNum(1);
    setShowSourceData(false);
    initializedStageRef.current = false;
    (async () => {
      try {
        await refresh();
      } catch (e) {
        onError(e, { action: 'load_run', runId });
      }
    })();
  }, [onError, refresh, runId]);

  useEffect(() => {
    if (!runId) return;
    const streamRequestId = newRequestId();
    const log = runPageLogger.child({ runId, streamRequestId });
    log.info('sse_connecting');
    const es = new EventSource(api.streamUrl(runId, streamRequestId));
    esRef.current = es;
    es.onopen = () => {
      log.info('sse_connected');
    };
    es.onmessage = async (evt) => {
      let payload = null;
      try {
        payload = JSON.parse(evt.data);
      } catch (error) {
        log.warn(
          {
            err: serializeError(error),
            operatorHint: requestOperatorHint(`/api/runs/${runId}/stream`, {
              method: 'GET',
              phase: 'parse',
            }),
            payloadPreview: String(evt?.data || '').slice(0, 500),
          },
          'sse_message_parse_failed'
        );
        return;
      }

      try {
        await refresh();
      } catch (error) {
        log.error(
          {
            err: serializeError(error),
            operatorHint:
              typeof error?.operatorHint === 'string'
                ? error.operatorHint
                : requestOperatorHint(`/api/runs/${runId}/stream`, {
                    method: 'GET',
                    phase: 'response',
                  }),
            payloadPreview: JSON.stringify(payload).slice(0, 500),
          },
          'sse_refresh_failed'
        );
        onError(error, { action: 'refresh_run_from_sse', runId });
      }
    };
    es.onerror = () => {
      log.warn('sse_error');
    };
    return () => {
      log.info('sse_closed');
      es.close();
      esRef.current = null;
    };
  }, [onError, refresh, runId]);

  const byStage = useMemo(() => groupByStage(artifacts), [artifacts]);
  const labelMap = run?.label_map || {};
  const runProgress = useMemo(() => getLatestRunProgress(run || {}), [run]);
  const currentStageNum = runProgress.stageNum ?? null;
  const hasDetailedProgress = Boolean(
    runProgress.determinate ||
      runProgress.phase ||
      runProgress.task ||
      runProgress.model ||
      runProgress.chunkLabel ||
      runProgress.elapsed ||
      currentStageNum
  );

  function stageComplete(stageNum) {
    const arts = byStage.get(String(stageNum)) || [];
    return arts.length > 0;
  }

  function inferDefaultStage() {
    if (currentStageNum) return Math.round(currentStageNum);
    for (const s of STAGES) {
      if (!stageComplete(s.num)) return s.num;
    }
    return 6;
  }

  useEffect(() => {
    if (!run) return;
    if (!artifacts.length && !currentStageNum) return;
    if (initializedStageRef.current) return;
    initializedStageRef.current = true;
    setActiveStageNum(inferDefaultStage());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, artifacts, currentStageNum]);

  if (!run) return <div className="page">Loading run…</div>;

  return (
    <div className="page">
      <div className="row space-between">
        <button onClick={onBack}>← Back</button>
        <div className="muted">
          Run {run.id.slice(0, 8)} — {run.status}
          {run.status === 'needs_auth' ? ' (needs auth)' : ''}
        </div>
      </div>

      {run.error_message ? <div className="error-banner">{run.error_message}</div> : null}

      {hasDetailedProgress ? (
        <div className="card">
          <div className="card-title">Live Progress</div>
          <div className="run-progress-header">
            <div>
              <div className="run-progress-summary">{runProgress.headline}</div>
              <div className="run-progress-subtitle">
                {run.status}
                {run.status === 'needs_auth' ? ' • needs auth' : ''}
              </div>
            </div>
            {runProgress.determinate ? (
              <div className="run-progress-percent">{runProgress.percent}%</div>
            ) : null}
          </div>

          {runProgress.determinate ? (
            <div className="run-progress-track" aria-hidden="true">
              <div className="run-progress-fill" style={{ width: `${runProgress.percent}%` }} />
            </div>
          ) : null}

          <div className="run-progress-meta">
            {currentStageNum ? (
              <div className="run-progress-chip">
                <span>Stage</span>
                <strong>
                  {Math.round(currentStageNum)}
                  {runProgress.stageName ? ` • ${runProgress.stageName.replace(/[_-]+/g, ' ')}` : ''}
                </strong>
              </div>
            ) : null}
            {runProgress.phase ? (
              <div className="run-progress-chip">
                <span>Phase</span>
                <strong>{runProgress.phase.replace(/[_-]+/g, ' ')}</strong>
              </div>
            ) : null}
            {runProgress.task ? (
              <div className="run-progress-chip">
                <span>Task</span>
                <strong>{runProgress.task}</strong>
              </div>
            ) : null}
            {runProgress.model ? (
              <div className="run-progress-chip">
                <span>Model</span>
                <strong>{runProgress.model}</strong>
              </div>
            ) : null}
            {runProgress.chunkLabel ? (
              <div className="run-progress-chip">
                <span>Chunk</span>
                <strong>{runProgress.chunkLabel}</strong>
              </div>
            ) : null}
            {runProgress.elapsed ? (
              <div className="run-progress-chip">
                <span>Elapsed</span>
                <strong>{runProgress.elapsed}</strong>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="card">
        <div className="card-title">Stage Progress</div>
        <div className="stage-grid">
          {STAGES.map((s) => (
            <button
              key={s.num}
              className={`stage-pill ${stageComplete(s.num) ? 'done' : ''} ${
                currentStageNum === s.num && !stageComplete(s.num) ? 'running' : ''
              } ${
                activeStageNum === s.num && !showSourceData ? 'active' : ''
              }`}
              onClick={() => {
                setActiveStageNum(s.num);
                setShowSourceData(false);
              }}
              type="button"
            >
              Stage {s.num}: {s.name}
            </button>
          ))}
          <span className="stage-divider" />
          <button
            className={`stage-pill source-data-pill ${showSourceData ? 'active' : ''}`}
            onClick={() => setShowSourceData(true)}
            type="button"
          >
            Source Data
          </button>
        </div>
      </div>

      {showSourceData ? (
        <div className="card source-data-card">
          <div className="card-title">Source Data</div>
          <SourceDataView reportId={run.report_id} onError={onError} />
        </div>
      ) : null}

      {!showSourceData && activeStageNum === 1 ? (
        <StageView
          stageNum={1}
          title="Stage 1: Initial Analysis"
          artifacts={byStage.get('1') || []}
          type="md"
          modelMetaById={modelMetaById}
        />
      ) : null}

      {!showSourceData && activeStageNum === 2 ? (
        <StagePeerReview
          stageNum={2}
          title="Stage 2: Peer Review"
          artifacts={byStage.get('2') || []}
          labelMap={labelMap}
          modelMetaById={modelMetaById}
        />
      ) : null}

      {!showSourceData && activeStageNum === 3 ? (
        <StageView
          stageNum={3}
          title="Stage 3: Revision"
          artifacts={byStage.get('3') || []}
          type="md"
          modelMetaById={modelMetaById}
        />
      ) : null}

      {!showSourceData && activeStageNum === 4 ? (
        <StageConsolidation
          title="Stage 4: Consolidation"
          artifacts={byStage.get('4') || []}
          selectedArtifactId={run.selected_artifact_id}
          onSelect={async (artifactId) => {
            try {
              await api.selectFinal(runId, { artifact_id: artifactId });
              await refresh();
            } catch (e) {
              onError(e, { action: 'select_consolidation', runId, artifactId });
            }
          }}
        />
      ) : null}

      {!showSourceData && activeStageNum === 5 ? (
        <StageFinalReview
          title="Stage 5: Final Review"
          artifacts={byStage.get('5') || []}
          modelMetaById={modelMetaById}
        />
      ) : null}

      {!showSourceData && activeStageNum === 6 ? (
        <StageFinalDrafts
          title="Stage 6: Final Drafts"
          artifacts={byStage.get('6') || []}
          selectedArtifactId={run.selected_artifact_id}
          modelMetaById={modelMetaById}
          onSelect={async (artifactId) => {
            try {
              await api.selectFinal(runId, { artifact_id: artifactId });
              await refresh();
            } catch (e) {
              onError(e, { action: 'select_final_draft', runId, artifactId });
            }
          }}
        />
      ) : null}

      <div className="card">
        <div className="card-title">Selection + Export</div>
        <div className="row">
          <div className="muted">
            Selected artifact: {run.selected_artifact_id ? run.selected_artifact_id.slice(0, 8) : '(none)'}
          </div>
          <button
            disabled={!run.selected_artifact_id || exporting}
            onClick={async () => {
              setExporting(true);
              try {
                await api.exportRun(runId);
                window.open(api.finalMdUrl(runId), '_blank');
                window.open(api.finalPdfUrl(runId), '_blank');
              } catch (e) {
                onError(e, { action: 'export_run', runId });
              } finally {
                setExporting(false);
              }
            }}
          >
            Export MD + PDF
          </button>
        </div>
      </div>
    </div>
  );
}

function StageView({ title, artifacts, type, modelMetaById }) {
  if (!artifacts.length) return null;
  const tabs = artifacts.map((a) => ({
    id: a.id,
    label: (
      <span className="tab-label">
        {modelMetaById?.get?.(a.model_id)?.name || a.model_id}{' '}
        <ModelBadge text={modelMetaById?.get?.(a.model_id)?.source || ''} />
      </span>
    ),
    content: a.content,
    contentType: a.content_type,
  }));

  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <Tabs
        tabs={tabs}
        render={(t) => {
          if (t.contentType?.includes?.('json')) {
            const parsed = tryJson(t.content);
            return <pre>{parsed ? JSON.stringify(parsed, null, 2) : t.content}</pre>;
          }
          return type === 'md' ? <ReactMarkdown>{t.content}</ReactMarkdown> : <pre>{t.content}</pre>;
        }}
      />
    </div>
  );
}

function StageConsolidation({ title, artifacts, selectedArtifactId, onSelect }) {
  const a = artifacts[0];
  if (!a) return null;
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      {onSelect ? (
        <div className="row">
          <button className={a.id === selectedArtifactId ? 'selected' : ''} onClick={() => onSelect(a.id)}>
            {a.id === selectedArtifactId ? 'Selected' : 'Select consolidated report'}
          </button>
        </div>
      ) : null}
      <ReactMarkdown>{a.content}</ReactMarkdown>
      <div className="muted">Model: {a.model_id}</div>
    </div>
  );
}

function StagePeerReview({ title, artifacts, labelMap, modelMetaById }) {
  if (!artifacts.length) return null;
  const tabs = artifacts.map((a) => ({
    id: a.id,
    label: (
      <span className="tab-label">
        {modelMetaById?.get?.(a.model_id)?.name || a.model_id}{' '}
        <ModelBadge text={modelMetaById?.get?.(a.model_id)?.source || ''} />
      </span>
    ),
    json: tryJson(a.content),
    raw: a.content,
  }));

  return (
    <div className="card">
      <div className="card-title">{title}</div>
      {Object.keys(labelMap).length ? (
        <details>
          <summary>Label map (A/B/C → model)</summary>
          <pre>{JSON.stringify(labelMap, null, 2)}</pre>
        </details>
      ) : null}
      <Tabs
        tabs={tabs}
        render={(t) =>
          t.json ? <pre>{JSON.stringify(t.json, null, 2)}</pre> : <pre>{t.raw}</pre>
        }
      />
    </div>
  );
}

function StageFinalReview({ title, artifacts, modelMetaById }) {
  if (!artifacts.length) return null;
  const parsed = artifacts
    .map((a) => ({ ...a, json: tryJson(a.content) }))
    .filter((a) => a.json);
  const votes = parsed.map((a) => a.json.vote).filter(Boolean);
  const approve = votes.filter((v) => v === 'APPROVE').length;
  const revise = votes.filter((v) => v === 'REVISE').length;

  const required = [];
  for (const a of parsed) {
    for (const c of a.json.required_changes || []) required.push(c);
  }

  const tabs = artifacts.map((a) => ({
    id: a.id,
    label: (
      <span className="tab-label">
        {modelMetaById?.get?.(a.model_id)?.name || a.model_id}{' '}
        <ModelBadge text={modelMetaById?.get?.(a.model_id)?.source || ''} />
      </span>
    ),
    json: tryJson(a.content),
    raw: a.content,
  }));

  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div className="row">
        <div className="muted">
          Votes: {approve} APPROVE, {revise} REVISE
        </div>
      </div>
      {required.length ? (
        <details>
          <summary>Required changes (combined)</summary>
          <ul>
            {required.map((c, idx) => (
              <li key={idx}>{c}</li>
            ))}
          </ul>
        </details>
      ) : null}
      <Tabs
        tabs={tabs}
        render={(t) =>
          t.json ? <pre>{JSON.stringify(t.json, null, 2)}</pre> : <pre>{t.raw}</pre>
        }
      />
    </div>
  );
}

function StageFinalDrafts({ title, artifacts, selectedArtifactId, onSelect, modelMetaById }) {
  if (!artifacts.length) return null;
  const tabs = artifacts.map((a) => ({
    id: a.id,
    label: (
      <span className="tab-label">
        {modelMetaById?.get?.(a.model_id)?.name || a.model_id}{' '}
        <ModelBadge text={modelMetaById?.get?.(a.model_id)?.source || ''} />
      </span>
    ),
    content: a.content,
  }));

  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div className="row">
        <div className="muted">Choose a draft to export:</div>
      </div>
      <Tabs
        tabs={tabs}
        render={(t) => (
          <div>
            <div className="row">
              <button
                className={t.id === selectedArtifactId ? 'selected' : ''}
                onClick={() => onSelect(t.id)}
              >
                {t.id === selectedArtifactId ? 'Selected' : 'Select this draft'}
              </button>
            </div>
            <ReactMarkdown>{t.content}</ReactMarkdown>
          </div>
        )}
      />
    </div>
  );
}

export default RunPage;
