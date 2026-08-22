import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:5173',
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      command: '../.venv/bin/python ../scripts/e2e-job-harness.py --host 127.0.0.1 --port 8876',
      url: 'http://127.0.0.1:8876/api/health',
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: 'env VIDEOSCOPE_HOST=127.0.0.1 VIDEOSCOPE_PORT=8876 pnpm dev',
      url: 'http://127.0.0.1:5173',
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile', use: { ...devices['iPhone 13'], browserName: 'chromium' } },
  ],
})
