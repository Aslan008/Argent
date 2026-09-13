import { registerPlugin, Capacitor, PluginListenerHandle } from '@capacitor/core';

export interface ScannedModel {
  name: string;
  path: string;
  size: number;
  sizeFormatted: string;
}

export interface ArgentLlamaPluginInterface {
  checkStoragePermission(): Promise<{ granted: boolean }>;
  requestStoragePermission(): Promise<{ opened: boolean }>;
  scanForModels(): Promise<{ models: ScannedModel[]; permissionRequired?: boolean }>;
  resolvePath(options: { path: string }): Promise<{ found: boolean; path: string; name?: string; size?: number }>;
  loadModel(options: { path: string; threads?: number; context?: number }): Promise<{ success: boolean; path: string; reused?: boolean }>;
  generateStream(options: { prompt: string; temperature?: number; max_tokens?: number }): Promise<{ success: boolean }>;
  stopGeneration(): Promise<{ stopped: boolean }>;
  unloadModel(): Promise<{ unloaded: boolean }>;
  addListener(eventName: 'token', listenerFunc: (data: { token: string }) => void): Promise<PluginListenerHandle>;
}

const ArgentLlama = registerPlugin<ArgentLlamaPluginInterface>('ArgentLlama');

export class LlamaNativeService {
  private static tokenListenerHandle: PluginListenerHandle | null = null;
  private static isCurrentlyGenerating = false;

  public static isAvailable(): boolean {
    return Capacitor.isNativePlatform() && Capacitor.isPluginAvailable('ArgentLlama');
  }

  public static async checkStoragePermission(): Promise<boolean> {
    if (!this.isAvailable()) return false;
    try {
      const res = await ArgentLlama.checkStoragePermission();
      return res.granted;
    } catch {
      return false;
    }
  }

  public static async requestStoragePermission(): Promise<void> {
    if (!this.isAvailable()) return;
    try {
      await ArgentLlama.requestStoragePermission();
    } catch (err) {
      console.warn('Ошибка вызова requestStoragePermission:', err);
    }
  }

  public static async scanForModels(): Promise<{ models: ScannedModel[]; permissionRequired?: boolean }> {
    if (!this.isAvailable()) return { models: [] };
    try {
      return await ArgentLlama.scanForModels();
    } catch (err) {
      console.warn('Ошибка scanForModels:', err);
      return { models: [] };
    }
  }

  public static async resolvePath(path: string): Promise<{ found: boolean; path: string; name?: string; size?: number }> {
    if (!this.isAvailable()) return { found: false, path };
    try {
      return await ArgentLlama.resolvePath({ path });
    } catch {
      return { found: false, path };
    }
  }

  public static async loadModel(path: string, threads = 4, context = 2048): Promise<boolean> {
    if (!this.isAvailable()) {
      throw new Error('Нативный движок llama.cpp доступен только в Android-приложении (APK).');
    }
    const res = await ArgentLlama.loadModel({ path, threads, context });
    return res.success;
  }

  public static async generateStream(
    prompt: string,
    temperature = 0.6,
    maxTokens = 1024,
    onToken: (token: string) => void,
    abortSignal?: AbortSignal
  ): Promise<void> {
    if (!this.isAvailable()) {
      throw new Error('Нативный движок llama.cpp доступен только в скомпилированном APK.');
    }

    if (this.isCurrentlyGenerating) {
      await this.stopGeneration();
    }

    this.isCurrentlyGenerating = true;

    // Очищаем старую подписку
    if (this.tokenListenerHandle) {
      await this.tokenListenerHandle.remove();
      this.tokenListenerHandle = null;
    }

    // Подписываемся на поток токенов
    this.tokenListenerHandle = await ArgentLlama.addListener('token', (data) => {
      if (data && typeof data.token === 'string') {
        onToken(data.token);
      }
    });

    const abortHandler = () => {
      this.stopGeneration().catch(console.warn);
    };

    if (abortSignal) {
      abortSignal.addEventListener('abort', abortHandler, { once: true });
    }

    try {
      await ArgentLlama.generateStream({
        prompt,
        temperature,
        max_tokens: maxTokens
      });
    } finally {
      this.isCurrentlyGenerating = false;
      if (abortSignal) {
        abortSignal.removeEventListener('abort', abortHandler);
      }
      if (this.tokenListenerHandle) {
        await this.tokenListenerHandle.remove();
        this.tokenListenerHandle = null;
      }
    }
  }

  public static async stopGeneration(): Promise<void> {
    if (!this.isAvailable()) return;
    try {
      await ArgentLlama.stopGeneration();
    } catch (err) {
      console.warn('Ошибка stopGeneration:', err);
    } finally {
      this.isCurrentlyGenerating = false;
    }
  }

  public static async unloadModel(): Promise<void> {
    if (!this.isAvailable()) return;
    try {
      await ArgentLlama.unloadModel();
    } catch (err) {
      console.warn('Ошибка unloadModel:', err);
    }
  }
}
