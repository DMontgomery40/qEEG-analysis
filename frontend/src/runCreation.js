// Keep the original UI admission identity until Create + Start is acknowledged.
// The server binds this operation_id to the immutable request and original run.
const PREFIX = 'qeeg-create-start:';

export async function retainRunCreation(payload) {
  const request = JSON.stringify(payload);
  const key = PREFIX + request;
  if (!navigator.locks?.request) throw new Error("This browser cannot safely retain a run request. Use a browser with Web Locks support.");
  return navigator.locks.request(key, () => {
    const saved = localStorage.getItem(key);
    if (saved !== null) {
      const record = JSON.parse(saved);
      if (record.request !== request || typeof record.operationId !== 'string' || !record.operationId) {
        throw new Error('The saved run request could not be read. Its original retry identity is still retained in this browser.');
      }
      return { key, operationId: record.operationId };
    }
    const operationId = crypto.randomUUID();
    localStorage.setItem(key, JSON.stringify({ request, operationId }));
    return { key, operationId };
  });
}

export async function completeRunCreation(retained) {
  return navigator.locks.request(retained.key, () => {
    const saved = localStorage.getItem(retained.key);
    if (saved !== null && JSON.parse(saved).operationId === retained.operationId) {
      localStorage.removeItem(retained.key);
    }
  });
}
