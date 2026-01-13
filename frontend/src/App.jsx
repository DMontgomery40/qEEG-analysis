import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './App.css';
import { api } from './api';
import Sidebar from './components/Sidebar';
import PatientPage from './components/PatientPage';
import RunPage from './components/RunPage';
import ResizeHandle from './components/ResizeHandle';

// Panel size persistence
const STORAGE_KEY = 'qeeg-panel-sizes';
const DEFAULT_SIDEBAR_WIDTH = 300;
const MIN_SIDEBAR_WIDTH = 200;
const MAX_SIDEBAR_WIDTH = 500;

function loadPanelSizes() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch {
    return {};
  }
}

function savePanelSizes(sizes) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sizes));
  } catch {
    // ignore
  }
}

function App() {
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState(null);
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [error, setError] = useState('');
  const autoStartTriedRef = useRef(false);
  const [geminiProjectId, setGeminiProjectId] = useState('');

  // Resizable sidebar
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    const sizes = loadPanelSizes();
    return sizes.sidebarWidth ?? DEFAULT_SIDEBAR_WIDTH;
  });

  const handleSidebarResize = useCallback((delta) => {
    setSidebarWidth((prev) => {
      const next = Math.max(MIN_SIDEBAR_WIDTH, Math.min(MAX_SIDEBAR_WIDTH, prev + delta));
      return next;
    });
  }, []);

  const handleSidebarResizeEnd = useCallback(() => {
    savePanelSizes({ ...loadPanelSizes(), sidebarWidth });
  }, [sidebarWidth]);

  const discoveredModels = useMemo(() => models?.discovered_models || [], [models]);
  const modelMetaById = useMemo(() => {
    const map = new Map();
    for (const m of models?.configured_models || []) map.set(m.id, m);
    return map;
  }, [models]);

  async function refreshModels() {
    const m = await api.models();
    setModels(m);
  }

  async function refreshHealth() {
    const h = await api.health();
    setHealth(h);

    // Best-effort auto-start for local single-user setups.
    if (
      h &&
      !h.cliproxy_reachable &&
      !h.cliproxy_auth_required &&
      !autoStartTriedRef.current &&
      h.cliproxyapi_installed
    ) {
      autoStartTriedRef.current = true;
      try {
        const cfg = h.cliproxy_config_found?.[0];
        await api.cliproxyStart(
          cfg ? { config_path: cfg, force_restart: true } : { force_restart: true }
        );
        await new Promise((r) => setTimeout(r, 1200));
        const h2 = await api.health();
        setHealth(h2);
      } catch {
        // ignore; user can use manual buttons
      }
    }
  }

  async function refreshPatients() {
    const p = await api.listPatients();
    setPatients(p);
    if (p.length && !selectedPatientId) setSelectedPatientId(p[0].id);
  }

  useEffect(() => {
    (async () => {
      try {
        await refreshHealth();
        await refreshModels();
        await refreshPatients();
      } catch (e) {
        setError(String(e?.message || e));
      }
    })();
  }, []);

  const showRun = selectedRunId != null;

  return (
    <div className="app">
      <Sidebar
        patients={patients}
        selectedPatientId={selectedPatientId}
        onSelectPatient={(id) => {
          setSelectedRunId(null);
          setSelectedPatientId(id);
        }}
        onCreatePatient={async (payload) => {
          try {
            const p = await api.createPatient(payload);
            await refreshPatients();
            setSelectedPatientId(p.id);
          } catch (e) {
            setError(String(e?.message || e));
          }
        }}
        style={{ width: sidebarWidth }}
      />

      <ResizeHandle
        direction="horizontal"
        onResize={handleSidebarResize}
        onResizeEnd={handleSidebarResizeEnd}
      />

      <div className="main">
        {error ? <div className="error-banner">{error}</div> : null}
        {health && !health.cliproxy_reachable ? (
          <div className="warn-banner">
            CLIProxyAPI not ready at {health.cliproxy_base_url}
            {health.cliproxy_auth_required ? ' (login required)' : ''}.
            <button
              style={{ marginLeft: 10 }}
              onClick={async () => {
                setError('');
                try {
                  await refreshHealth();
                  await refreshModels();
                  await refreshPatients();
                } catch (e) {
                  setError(String(e?.message || e));
                }
              }}
            >
              Retry
            </button>
            {!health.cliproxy_auth_required ? (
              <button
                style={{ marginLeft: 10 }}
                onClick={async () => {
                  setError('');
                  try {
                    if (!health.cliproxyapi_installed && health.brew_installed) {
                      await api.cliproxyInstall({});
                      setError('Installing CLIProxyAPI… check data/cliproxy_install.log then retry.');
                      return;
                    }
                    const cfg = health.cliproxy_config_found?.[0];
                    await api.cliproxyStart(cfg ? { config_path: cfg, force_restart: true } : { force_restart: true });
                    await refreshHealth();
                    await refreshModels();
                    await refreshPatients();
                  } catch (e) {
                    setError(String(e?.message || e));
                  }
                }}
              >
                {health.cliproxyapi_installed ? 'Start Proxy' : health.brew_installed ? 'Install Proxy' : 'Start Proxy'}
              </button>
            ) : (
              <>
                <button
                  style={{ marginLeft: 10 }}
                  onClick={async () => {
                    setError('');
                    try {
                      await api.cliproxyLogin({ mode: 'login' });
                    } catch (e) {
                      setError(String(e?.message || e));
                    }
                  }}
                >
                  Login (All)
                </button>
                <button
                  style={{ marginLeft: 10 }}
                  onClick={async () => {
                    setError('');
                    try {
                      await api.cliproxyLogin({ mode: 'claude' });
                    } catch (e) {
                      setError(String(e?.message || e));
                    }
                  }}
                >
                  Claude Login
                </button>
                <button
                  style={{ marginLeft: 10 }}
                  onClick={async () => {
                    setError('');
                    try {
                      await api.cliproxyLogin({ mode: 'codex' });
                    } catch (e) {
                      setError(String(e?.message || e));
                    }
                  }}
                >
                  Codex Login
                </button>
                <button
                  style={{ marginLeft: 10 }}
                  onClick={async () => {
                    setError('');
                    try {
                      await api.cliproxyLogin({
                        mode: 'gemini',
                        project_id: geminiProjectId.trim() || null,
                      });
                    } catch (e) {
                      setError(String(e?.message || e));
                    }
                  }}
                >
                  Gemini Login
                </button>
                <input
                  className="inline-input"
                  placeholder="Gemini project_id (optional)"
                  value={geminiProjectId}
                  onChange={(e) => setGeminiProjectId(e.target.value)}
                />
              </>
            )}
            <details style={{ marginTop: 10 }}>
              <summary>Troubleshooting</summary>
              <div style={{ marginTop: 8 }}>
                {health.cliproxy_error_kind ? (
                  <div className="muted">Error kind: {health.cliproxy_error_kind}</div>
                ) : null}
                {health.cliproxy_error ? <pre>{health.cliproxy_error}</pre> : null}
                <div className="muted" style={{ marginTop: 8 }}>
                  Suggested commands:
                </div>
                <pre>{(health.suggested_commands || []).join('\n')}</pre>
              </div>
            </details>
          </div>
        ) : null}

        {showRun ? (
          <RunPage
            runId={selectedRunId}
            modelMetaById={modelMetaById}
            onBack={() => setSelectedRunId(null)}
            onError={(msg) => setError(msg)}
          />
        ) : (
          <PatientPage
            patientId={selectedPatientId}
            discoveredModels={discoveredModels}
            modelMetaById={modelMetaById}
            defaultConsolidator={discoveredModels[0] || ''}
            onSelectRun={(runId) => setSelectedRunId(runId)}
            onRefreshGlobal={async () => {
              try {
                await refreshHealth();
                await refreshModels();
                await refreshPatients();
              } catch (e) {
                setError(String(e?.message || e));
              }
            }}
            onError={(msg) => setError(msg)}
          />
        )}
      </div>
    </div>
  );
}

export default App;
