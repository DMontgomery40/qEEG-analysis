import pino from 'pino';

const LOG_LEVEL = import.meta.env.VITE_LOG_LEVEL || (import.meta.env.DEV ? 'debug' : 'info');

export function newRequestId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID().replace(/-/g, '');
  }
  return `${Date.now().toString(16)}${Math.random().toString(16).slice(2, 10)}`;
}

export function errorMessage(error) {
  if (error instanceof Error) return error.message || error.name || 'Unknown error';
  if (typeof error === 'string') return error;
  if (error == null) return 'Unknown error';
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}

export function serializeError(error) {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
      cause: error.cause ? serializeError(error.cause) : undefined,
    };
  }
  if (typeof error === 'string') return { message: error };
  if (error == null) return undefined;
  try {
    return JSON.parse(JSON.stringify(error));
  } catch {
    return { message: String(error) };
  }
}

export const logger = pino({
  name: 'qeeg-frontend',
  level: LOG_LEVEL,
  base: { app: 'qeeg-frontend' },
  timestamp: pino.stdTimeFunctions.isoTime,
  serializers: {
    err: pino.stdSerializers.err,
  },
  browser: {
    asObject: true,
  },
});

export function childLogger(bindings) {
  return logger.child(bindings || {});
}
