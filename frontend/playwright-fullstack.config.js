// @ts-check
import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright config for Tier 3: Full-stack E2E testing.
 *
 * These tests run with:
 * - Real FastAPI backend (with mocked CLIProxyAPI via QEEG_MOCK_LLM=1)
 * - Real Vite frontend
 * - Real SQLite database (ephemeral for tests)
 *
 * This exercises the full stack without making real LLM calls.
 */
export default defineConfig({
  testDir: './tests',
  testMatch: 'fullstack.spec.js',

  /* Run tests in files in parallel */
  fullyParallel: false, // Sequential for fullstack to avoid port conflicts

  /* Fail the build on CI if you accidentally left test.only in the source code */
  forbidOnly: !!process.env.CI,

  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,

  /* Reporter to use */
  reporter: [['list'], ['html', { outputFolder: 'playwright-report-fullstack' }]],

  /* Shared settings for all projects */
  use: {
    /* Base URL for the frontend */
    baseURL: 'http://localhost:5173',

    /* Collect trace when retrying the failed test */
    trace: 'on-first-retry',

    /* Screenshot on failure */
    screenshot: 'only-on-failure',
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  /* Run both backend and frontend before starting tests */
  webServer: [
    {
      command: 'cd .. && QEEG_MOCK_LLM=1 uv run python -m backend.main',
      url: 'http://localhost:8000/api/health',
      reuseExistingServer: !process.env.CI,
      timeout: 60000,
      env: {
        QEEG_MOCK_LLM: '1',
      },
    },
    {
      command: 'npm run dev',
      url: 'http://localhost:5173',
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
    },
  ],

  /* Timeouts */
  timeout: 60000, // Longer for fullstack tests
  expect: {
    timeout: 10000,
  },
});
