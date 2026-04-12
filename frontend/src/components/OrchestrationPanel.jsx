import { useMemo, useState } from 'react';
import { api } from '../api';
import {
  PATIENT_ACTIONS,
  formatStatusDate,
  getPatientActionConfig,
  getPatientActionState,
  getPatientOrchestrationSections,
  getPatientSidebarSummary,
} from '../orchestration';
import './OrchestrationPanel.css';

function ProgressBar({ percent, compact = false }) {
  if (percent == null) return null;
  return (
    <div className={`status-progress ${compact ? 'compact' : ''}`} aria-hidden="true">
      <div className="status-progress-fill" style={{ width: `${percent}%` }} />
    </div>
  );
}

function StatusBadge({ label, tone = 'neutral' }) {
  return <span className={`status-badge status-badge-${tone}`}>{label}</span>;
}

function ReportLifecycleList({ reports }) {
  if (!Array.isArray(reports) || reports.length === 0) return null;

  return (
    <section className="orchestration-report-lifecycle">
      <div className="orchestration-report-header">
        <div className="orchestration-section-title">Report lifecycle</div>
        <div className="orchestration-muted">Uploaded to extracted to council to patient-facing to portal to Cathode.</div>
      </div>
      <div className="orchestration-report-list">
        {reports.map((report) => {
          const lifecycle = report.lifecycle || {};
          const councilLabel = lifecycle.council_status ? `Council: ${lifecycle.council_status}` : 'Council: pending';
          const patientFacingLabel = lifecycle.patient_facing_status
            ? `Patient PDF: ${lifecycle.patient_facing_status}`
            : 'Patient PDF: pending';
          const portalLabel = lifecycle.portal_sync_status ? `Portal: ${lifecycle.portal_sync_status}` : 'Portal: unknown';
          const cathodeLabel = lifecycle.cathode_status ? `Cathode: ${lifecycle.cathode_status}` : 'Cathode: pending';
          return (
            <div key={report.report_id} className="orchestration-report-row">
              <div>
                <div className="orchestration-report-name">{report.filename}</div>
                <div className="orchestration-muted">{formatStatusDate(report.created_at)}</div>
              </div>
              <div className="orchestration-report-badges">
                <StatusBadge label={lifecycle.uploaded ? 'Uploaded' : 'Missing'} tone={lifecycle.uploaded ? 'success' : 'error'} />
                <StatusBadge label={lifecycle.extracted ? 'Extracted' : 'Not extracted'} tone={lifecycle.extracted ? 'success' : 'warning'} />
                <StatusBadge label={councilLabel} tone="neutral" />
                <StatusBadge label={patientFacingLabel} tone="neutral" />
                <StatusBadge label={portalLabel} tone="neutral" />
                <StatusBadge label={cathodeLabel} tone="neutral" />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function OrchestrationSection({ section, onOpenRun }) {
  return (
    <section className={`orchestration-section ${section.available ? '' : 'empty'}`}>
      <div className="orchestration-section-header">
        <div>
          <div className="orchestration-section-title">{section.title}</div>
          <div className="orchestration-section-summary">{section.summary}</div>
        </div>
        <StatusBadge label={section.statusLabel} tone={section.tone} />
      </div>

      <ProgressBar percent={section.progress.determinate ? section.progress.percent : null} />

      {section.details.length ? (
        <div className="orchestration-detail-grid">
          {section.details.map((detail) => (
            <div key={`${section.id}-${detail.label}`} className="orchestration-detail">
              <span>{detail.label}</span>
              <strong>{detail.value}</strong>
            </div>
          ))}
        </div>
      ) : null}

      {section.error ? <div className="orchestration-error">{section.error}</div> : null}

      {section.runId && onOpenRun ? (
        <button className="orchestration-open-run" onClick={() => onOpenRun(section.runId)} type="button">
          Open run {section.runId.slice(0, 8)}
        </button>
      ) : null}
    </section>
  );
}

function OrchestrationPanel({
  patientId,
  patient,
  orchestration,
  orchestrationLoading,
  orchestrationError,
  runs,
  onSelectRun,
  onRefresh,
  onRefreshGlobal,
  onError,
  onNotice,
}) {
  const [busyActionId, setBusyActionId] = useState('');

  const overallSummary = useMemo(() => {
    return getPatientSidebarSummary(patient);
  }, [patient]);

  const sections = useMemo(() => {
    return getPatientOrchestrationSections(orchestration, runs);
  }, [orchestration, runs]);

  const latestUpdatedAt = useMemo(() => {
    return formatStatusDate(orchestration?.updated_at || orchestration?.last_updated || '');
  }, [orchestration]);

  async function runAction(actionId) {
    const config = getPatientActionConfig(actionId);
    if (!config) return;

    setBusyActionId(actionId);
    try {
      const response = await api.runPatientAction(patientId, actionId);
      const message =
        (typeof response === 'string' ? response : response?.message) || config.successMessage;
      onNotice?.(message, { action: config.requestLabel, patientId });
      if (actionId === 'export_council_artifacts' && response?.run_id) {
        window.open(api.finalMdUrl(response.run_id), '_blank');
        window.open(api.finalPdfUrl(response.run_id), '_blank');
      }
      await onRefresh?.();
      await onRefreshGlobal?.();
    } catch (error) {
      onError?.(error, { action: config.requestLabel, patientId });
    } finally {
      setBusyActionId('');
    }
  }

  return (
    <div className="card orchestration-card">
      <div className="orchestration-header">
        <div>
          <div className="card-title">Orchestration + Sync</div>
          <div className="orchestration-subtitle">
            {overallSummary?.label || 'Backend orchestration state'}
          </div>
        </div>
        <div className="orchestration-header-meta">
          {overallSummary ? <StatusBadge label={overallSummary.label} tone={overallSummary.tone} /> : null}
          {overallSummary?.percent != null ? <span className="orchestration-percent">{overallSummary.percent}%</span> : null}
          {latestUpdatedAt ? <span className="orchestration-updated">Updated {latestUpdatedAt}</span> : null}
        </div>
      </div>

      <ProgressBar percent={overallSummary?.percent ?? null} compact />

      <div className="orchestration-actions">
        {PATIENT_ACTIONS.map((action) => {
          const state = getPatientActionState(orchestration, action.id);
          const disabled = busyActionId !== '' || !state.enabled;
          const label = busyActionId === action.id ? 'Working…' : action.label;
          return (
            <button
              key={action.id}
              type="button"
              disabled={disabled}
              title={state.reason || ''}
              onClick={() => runAction(action.id)}
            >
              {label}
            </button>
          );
        })}
      </div>

      {orchestrationLoading ? <div className="orchestration-muted">Loading orchestration state…</div> : null}
      {!orchestrationLoading && orchestrationError ? (
        <div className="orchestration-warning">{orchestrationError}</div>
      ) : null}

      <div className="orchestration-grid">
        {sections.map((section) => (
          <OrchestrationSection key={section.id} section={section} onOpenRun={onSelectRun} />
        ))}
      </div>

      <ReportLifecycleList reports={orchestration?.reports} />
    </div>
  );
}

export default OrchestrationPanel;
