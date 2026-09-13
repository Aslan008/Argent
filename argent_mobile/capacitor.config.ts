import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.argent.mobile',
  appName: 'ArgentMobile',
  webDir: 'dist',
  server: {
    androidScheme: 'https'
  }
};

export default config;
