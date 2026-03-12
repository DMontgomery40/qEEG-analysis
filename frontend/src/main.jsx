import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import ErrorBoundary from './components/ErrorBoundary.jsx';
import { childLogger, serializeError } from './logger.js';
import './index.css';

const bootLogger = childLogger({ area: 'bootstrap' });

const WINDOW_HANDLER_KEY = '__QEEG_BOOTSTRAP_LOG_HANDLERS__';

function installWindowLogging() {
  if (typeof window === 'undefined' || window[WINDOW_HANDLER_KEY]) {
    return;
  }

  const onError = (event) => {
    bootLogger.error(
      {
        err: serializeError(event.error || event.message),
        source: event.filename,
        lineno: event.lineno,
        colno: event.colno,
      },
      'window_error'
    );
  };

  const onUnhandledRejection = (event) => {
    bootLogger.error(
      {
        err: serializeError(event.reason),
      },
      'unhandled_rejection'
    );
  };

  window.addEventListener('error', onError);
  window.addEventListener('unhandledrejection', onUnhandledRejection);
  window[WINDOW_HANDLER_KEY] = { onError, onUnhandledRejection };
}

function removeWindowLogging() {
  if (typeof window === 'undefined') {
    return;
  }
  const handlers = window[WINDOW_HANDLER_KEY];
  if (!handlers) {
    return;
  }
  window.removeEventListener('error', handlers.onError);
  window.removeEventListener('unhandledrejection', handlers.onUnhandledRejection);
  delete window[WINDOW_HANDLER_KEY];
}

installWindowLogging();

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    removeWindowLogging();
  });
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>
);
