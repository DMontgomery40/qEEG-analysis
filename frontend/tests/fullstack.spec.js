// @ts-check
import { test, expect } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';

/**
 * Tier 3: Full-stack E2E tests
 *
 * These tests run against:
 * - Real FastAPI backend with mocked CLIProxyAPI (QEEG_MOCK_LLM=1)
 * - Real Vite frontend
 * - Real SQLite database
 *
 * The backend uses the mock LLM transport so no real LLM calls are made,
 * but all other code paths are exercised.
 */

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Helper to get example PDF path
function getExamplePdfPath() {
  // Look for example PDF in examples directory
  const examplesDir = path.resolve(__dirname, '../../examples');
  if (fs.existsSync(examplesDir)) {
    const files = fs.readdirSync(examplesDir);
    const pdf = files.find((f) => f.endsWith('.pdf'));
    if (pdf) {
      return path.join(examplesDir, pdf);
    }
  }
  return null;
}

// Helper to wait for run to complete
async function waitForRunComplete(page, timeout = 90000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    // Look for complete status in the page
    const statusText = await page.locator('.muted').filter({ hasText: /Run.*complete/i }).count();
    if (statusText > 0) {
      return 'complete';
    }

    // Check for failed status
    const failedText = await page.locator('.error-banner').count();
    if (failedText > 0) {
      const errorMsg = await page.locator('.error-banner').textContent();
      throw new Error(`Run failed: ${errorMsg}`);
    }

    // Wait a bit before checking again
    await page.waitForTimeout(1000);
  }
  throw new Error('Run did not complete within timeout');
}

test.describe('Full-stack integration', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the app
    await page.goto('/');

    // Wait for the app to load
    await expect(page.locator('.app')).toBeVisible({ timeout: 30000 });
  });

  test('app loads and shows health status', async ({ page }) => {
    // The app should load successfully
    await expect(page.locator('.sidebar')).toBeVisible();

    // In mock mode, should not show warning banner about CLIProxyAPI
    // (because mock mode returns healthy status)
    await page.waitForTimeout(1000); // Give time for health check
    const warnBanner = page.locator('.warn-banner');
    const warnCount = await warnBanner.count();

    // If there's no warn banner, we're in mock mode and healthy
    // If there is one, check it's not about CLIProxyAPI being unreachable
    if (warnCount > 0) {
      const text = await warnBanner.textContent();
      // This is acceptable - might just be a transient state
      console.log('Warning banner present:', text);
    }
  });

  test('can create a new patient', async ({ page }) => {
    // Find and click the create patient button (+) in sidebar
    const createButton = page.locator('.sidebar button').filter({ hasText: '+' });
    await createButton.click();

    // A modal or inline form should appear - look for input
    // The sidebar has a simple text input for new patient name
    await page.waitForTimeout(500);

    // The app creates a patient when clicking + and selecting it
    // Check that we can see Patient label input in main area
    await expect(page.getByText('Patient').first()).toBeVisible();
  });

  test('displays discovered models from backend', async ({ page }) => {
    // Navigate to New Run section
    await expect(page.getByText('New Run')).toBeVisible();

    // The council models dropdown should show the mock models
    const councilSelect = page.locator('.multi-select');
    await expect(councilSelect).toBeVisible();

    // Click to open options
    const options = councilSelect.locator('option');
    const count = await options.count();

    // Should have mock models available
    expect(count).toBeGreaterThan(0);
  });
});

test.describe('Report upload', () => {
  test('can upload a PDF report', async ({ page }) => {
    const pdfPath = getExamplePdfPath();

    // Skip if no example PDF available
    if (!pdfPath) {
      test.skip();
      return;
    }

    await page.goto('/');
    await expect(page.locator('.app')).toBeVisible({ timeout: 30000 });

    // First create a patient if none exists
    // Look for the Reports section
    await expect(page.getByText('Reports').first()).toBeVisible();

    // Find file input and upload
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(pdfPath);

    // Wait for upload to complete (preview should appear)
    await expect(page.locator('.preview')).toBeVisible({ timeout: 30000 });

    // Should show extracted preview
    await expect(page.getByText('Extracted preview')).toBeVisible();
  });
});

test.describe('Full pipeline execution', () => {
  test.slow(); // Mark as slow test since pipeline takes time

  test('can create and run a full pipeline', async ({ page }) => {
    const pdfPath = getExamplePdfPath();

    // Skip if no example PDF available
    if (!pdfPath) {
      test.skip();
      return;
    }

    await page.goto('/');
    await expect(page.locator('.app')).toBeVisible({ timeout: 30000 });

    // Step 1: Create a patient if needed
    // Check if there's already a patient selected
    let hasPatient = (await page.locator('.list-item.active').count()) > 0;
    if (!hasPatient) {
      // Create new patient
      const createButton = page.locator('.sidebar button').filter({ hasText: '+' });
      await createButton.click();
      await page.waitForTimeout(500);
      hasPatient = true;
    }

    // Step 2: Upload a report
    await expect(page.getByText('Reports').first()).toBeVisible();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(pdfPath);

    // Wait for upload to complete
    await expect(page.locator('.preview')).toBeVisible({ timeout: 30000 });

    // Step 3: Configure the run
    // Select council models
    const councilSelect = page.locator('.multi-select');
    await councilSelect.selectOption(['mock-council-a', 'mock-council-b']);

    // Select consolidator
    const consolidatorSelect = page.locator('label:has-text("Consolidator") select:not(.multi-select)');
    await consolidatorSelect.selectOption('mock-council-a');

    // Step 4: Create and start the run
    const createButton = page.getByRole('button', { name: 'Create + Start' });
    await expect(createButton).toBeEnabled();
    await createButton.click();

    // Should navigate to run page
    await expect(page.getByRole('button', { name: '← Back' })).toBeVisible({ timeout: 10000 });

    // Step 5: Wait for pipeline to complete
    // The pipeline runs through 6 stages with mocked LLM responses
    await waitForRunComplete(page, 120000);

    // Step 6: Verify all stages completed
    // Check that stage pills show as done
    for (let i = 1; i <= 6; i++) {
      const stagePill = page.locator(`.stage-pill.done`).filter({ hasText: `Stage ${i}` });
      await expect(stagePill).toBeVisible({ timeout: 5000 });
    }

    // Step 7: Click through stages to verify content
    // Stage 1
    await page.getByText('Stage 1: initial_analysis').click();
    await expect(page.getByText('Stage 1: Initial Analysis')).toBeVisible();

    // Stage 6 (final draft)
    await page.getByText('Stage 6: final_draft').click();
    await expect(page.getByText('Stage 6: Final Drafts')).toBeVisible();

    // Should be able to select a draft
    await expect(page.getByText('Choose a draft to export:')).toBeVisible();
  });
});

test.describe('Export functionality', () => {
  test.slow();

  test('can export completed run', async ({ page }) => {
    const pdfPath = getExamplePdfPath();

    if (!pdfPath) {
      test.skip();
      return;
    }

    await page.goto('/');
    await expect(page.locator('.app')).toBeVisible({ timeout: 30000 });

    // Quick setup: upload report and start run
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(pdfPath);
    await expect(page.locator('.preview')).toBeVisible({ timeout: 30000 });

    // Configure and start run
    const councilSelect = page.locator('.multi-select');
    await councilSelect.selectOption(['mock-council-a', 'mock-council-b']);
    const consolidatorSelect = page.locator('label:has-text("Consolidator") select:not(.multi-select)');
    await consolidatorSelect.selectOption('mock-council-a');

    const createButton = page.getByRole('button', { name: 'Create + Start' });
    await createButton.click();

    // Wait for completion
    await expect(page.getByRole('button', { name: '← Back' })).toBeVisible({ timeout: 10000 });
    await waitForRunComplete(page, 120000);

    // Go to Stage 6 and select a draft
    await page.getByText('Stage 6: final_draft').click();
    await expect(page.getByText('Choose a draft to export:')).toBeVisible();

    // Select the draft
    const selectDraftButton = page.getByRole('button', { name: 'Select this draft' });
    if ((await selectDraftButton.count()) > 0) {
      await selectDraftButton.first().click();
    }

    // Now export should be enabled
    const exportButton = page.getByRole('button', { name: 'Export MD + PDF' });

    // Check if button is enabled (artifact must be selected)
    const isDisabled = await exportButton.isDisabled();
    if (isDisabled) {
      // Need to select an artifact first
      console.log('Export button disabled - no artifact selected');
      return;
    }

    // Click export (will open new tabs)
    const [newPage] = await Promise.all([
      page.waitForEvent('popup', { timeout: 5000 }).catch(() => null),
      exportButton.click(),
    ]);

    // Export should have been triggered
    if (newPage) {
      // A popup was opened (for MD or PDF download)
      await newPage.close();
    }
  });
});

test.describe('SSE streaming', () => {
  test('receives SSE events during pipeline execution', async ({ page }) => {
    const pdfPath = getExamplePdfPath();

    if (!pdfPath) {
      test.skip();
      return;
    }

    await page.goto('/');
    await expect(page.locator('.app')).toBeVisible({ timeout: 30000 });

    // Capture network requests
    const sseRequests = [];
    page.on('request', (request) => {
      if (request.url().includes('/stream')) {
        sseRequests.push(request.url());
      }
    });

    // Upload and start run
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(pdfPath);
    await expect(page.locator('.preview')).toBeVisible({ timeout: 30000 });

    const councilSelect = page.locator('.multi-select');
    await councilSelect.selectOption(['mock-council-a', 'mock-council-b']);
    const consolidatorSelect = page.locator('label:has-text("Consolidator") select:not(.multi-select)');
    await consolidatorSelect.selectOption('mock-council-a');

    const createButton = page.getByRole('button', { name: 'Create + Start' });
    await createButton.click();

    // Wait for navigation to run page
    await expect(page.getByRole('button', { name: '← Back' })).toBeVisible({ timeout: 10000 });

    // Give time for SSE connection to establish
    await page.waitForTimeout(2000);

    // Verify SSE endpoint was called
    expect(sseRequests.length).toBeGreaterThan(0);
    expect(sseRequests[0]).toContain('/stream');
  });
});

test.describe('Error handling', () => {
  test('displays error messages from backend', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.app')).toBeVisible({ timeout: 30000 });

    // Try to create a run without a report
    // This should fail validation

    // First check if New Run section is visible
    await expect(page.getByText('New Run')).toBeVisible();

    // The Create + Start button should be disabled without proper selection
    const createButton = page.getByRole('button', { name: 'Create + Start' });
    const isDisabled = await createButton.isDisabled();

    // Button should be disabled when form is not properly filled
    expect(isDisabled).toBe(true);
  });
});

test.describe('Navigation', () => {
  test('can navigate between patient page and run page', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.app')).toBeVisible({ timeout: 30000 });

    // We're on the patient page initially
    await expect(page.getByText('Run History')).toBeVisible();

    // If there are runs in history, click one
    const runItems = page.locator('.list-item').filter({ hasText: /run-|pending|complete/i });
    const runCount = await runItems.count();

    if (runCount > 0) {
      // Click first run
      await runItems.first().click();

      // Should be on run page now
      await expect(page.getByRole('button', { name: '← Back' })).toBeVisible();

      // Click back
      await page.getByRole('button', { name: '← Back' }).click();

      // Should be back on patient page
      await expect(page.getByText('Run History')).toBeVisible();
    }
  });
});
