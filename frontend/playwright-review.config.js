import config from './playwright.config.js';
export default {
  ...config,
  use: { ...config.use, baseURL: 'http://127.0.0.1:5197' },
  webServer: { ...config.webServer, command: 'npm run dev -- --host 127.0.0.1 --port 5197 --strictPort', url: 'http://127.0.0.1:5197', reuseExistingServer: false },
};
