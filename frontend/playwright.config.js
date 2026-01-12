// @ts-check
import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright config for Tier 2: Frontend E2E with mocked API.
 *
 * These tests mock all /api/* endpoints using route interception,
 * allowing frontend testing without a running backend.
 */
export default defineConfig({
  testDir: './tests',
  testMatch: 'e2e-mocked.spec.js',

  /* Run tests in parallel */
  fullyParallel: true,

  /* Fail the build on CI if you accidentally left test.only in the source code */
  forbidOnly: !!process.env.CI,

  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,

  /* Reporter to use */
  reporter: [['list'], ['html', { outputFolder: 'playwright-report' }]],

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

  /* Run local dev server before starting tests */
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
  },

  /* Timeouts */
  timeout: 30000,
  expect: {
    timeout: 5000,
  },
});
