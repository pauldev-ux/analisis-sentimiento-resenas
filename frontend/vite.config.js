import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Redirige /api/* a la API de FastAPI (evita problemas de CORS en desarrollo)
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        rewrite: (ruta) => ruta.replace(/^\/api/, ''),
      },
    },
  },
})
