import { defineConfig } from 'vite';
import fs from 'node:fs';

// Изоляция страницы (COOP/COEP) включена для WebLLM. Режим COEP «credentialless» вместо «require-corp»:
// при require-corp браузер блокировал стили Google Fonts (они не присылают заголовок CORP), и шрифты
// интерфейса не загружались; credentialless пускает такие ресурсы без кук, изоляция сохраняется.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless'
};

export default defineConfig({
  root: fs.realpathSync(process.cwd()),
  server: {
    host: '0.0.0.0',
    port: 3000,
    headers: isolation
  },
  preview: {
    host: '0.0.0.0',
    port: 3000,
    headers: isolation
  },
  build: {
    target: 'esnext'
  }
});
