import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import ErrorBoundary from './components/ErrorBoundary.jsx';
import { childLogger, serializeError } from './logger.js';
import './index.css';

const bootLogger = childLogger({ area: 'bootstrap' });

window.addEventListener('error', (event) => {
  bootLogger.error(
    {
      err: serializeError(event.error || event.message),
      source: event.filename,
      lineno: event.lineno,
      colno: event.colno,
    },
    'window_error'
  );
});

window.addEventListener('unhandledrejection', (event) => {
  bootLogger.error(
    {
      err: serializeError(event.reason),
    },
    'unhandled_rejection'
  );
});

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>
);
