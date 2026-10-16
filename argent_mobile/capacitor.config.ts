import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.argent.mobile',
  appName: 'ArgentMobile',
  webDir: 'dist',
  server: {
    androidScheme: 'https'
  },
  android: {
    // Страница приложения открыта как https://localhost, а домашняя Ollama/LM Studio отвечает по
    // http://192.168.x.x — без этого WebView блокирует такой запрос как смешанное содержимое
    // (вместе с android:usesCleartextTraffic в AndroidManifest.xml).
    allowMixedContent: true
  }
};

export default config;
