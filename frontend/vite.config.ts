import react from '@vitejs/plugin-react'
import { loadEnv } from 'vite'
import { defineConfig } from 'vitest/config'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '..', '')
  const backendHost = env.VIDEOSCOPE_HOST || '127.0.0.1'
  const backendPort = env.VIDEOSCOPE_PORT || '8765'

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': `http://${backendHost}:${backendPort}`,
      },
    },
    test: {
      environment: 'jsdom',
      globals: true,
      include: ['src/**/*.test.{ts,tsx}'],
      setupFiles: './src/test/setup.ts',
      coverage: {
        provider: 'v8',
        thresholds: {
          lines: 80,
          functions: 80,
          statements: 80,
          branches: 75,
        },
      },
    },
  }
})
