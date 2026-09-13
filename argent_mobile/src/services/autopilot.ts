import { registerPlugin } from '@capacitor/core';

export interface UIElement {
  text: string;
  desc: string;
  id: string;
  clickable: boolean;
  editable: boolean;
  x: number;
  y: number;
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface AutopilotPluginInterface {
  checkAccessibilityStatus(): Promise<{ enabled: boolean }>;
  openAccessibilitySettings(): Promise<{ opened: boolean }>;
  getScreenElements(): Promise<{ elements: UIElement[] }>;
  clickCoordinates(options: { x: number; y: number }): Promise<{ success: boolean }>;
  clickText(options: { text: string }): Promise<{ success: boolean }>;
  typeText(options: { text: string }): Promise<{ success: boolean }>;
  scroll(options: { direction: 'up' | 'down' }): Promise<{ success: boolean }>;
  launchApp(options: { app: string }): Promise<{ launched: boolean }>;
  pressHome(): Promise<{ success: boolean }>;
  pressBack(): Promise<{ success: boolean }>;
}

// Регистрируем нативный плагин или создаем безопасную веб-заглушку
const NativeAutopilot = registerPlugin<AutopilotPluginInterface>('ArgentAutopilot', {
  web: {
    checkAccessibilityStatus: async () => ({ enabled: false }),
    openAccessibilitySettings: async () => {
      alert('В веб-режиме настройки спец. возможностей недоступны. Установите APK для полного доступа.');
      return { opened: false };
    },
    getScreenElements: async () => ({ elements: [] }),
    clickCoordinates: async () => ({ success: true }),
    clickText: async () => ({ success: true }),
    typeText: async () => ({ success: true }),
    scroll: async () => ({ success: true }),
    launchApp: async (opts: { app: string }) => {
      console.log('Симуляция запуска приложения:', opts.app);
      return { launched: true };
    },
    pressHome: async () => ({ success: true }),
    pressBack: async () => ({ success: true })
  }
});

export class AutopilotService {
  /**
   * Проверяет, включена ли служба спец. возможностей в Android
   */
  public static async isServiceEnabled(): Promise<boolean> {
    try {
      const res = await NativeAutopilot.checkAccessibilityStatus();
      return Boolean(res.enabled);
    } catch {
      return false;
    }
  }

  /**
   * Открывает экран настроек спец. возможностей Android
   */
  public static async openSettings(): Promise<void> {
    try {
      await NativeAutopilot.openAccessibilitySettings();
    } catch (e) {
      console.warn('Не удалось открыть настройки спец. возможностей:', e);
    }
  }

  /**
   * Запускает установленное приложение на телефоне
   */
  public static async launchApp(appName: string): Promise<boolean> {
    try {
      const res = await NativeAutopilot.launchApp({ app: appName });
      return res.launched;
    } catch (err: any) {
      throw new Error(`Не удалось запустить ${appName}: ${err.message}`);
    }
  }

  /**
   * Кликает по тексту или названию кнопки
   */
  public static async click(target: string): Promise<boolean> {
    try {
      const res = await NativeAutopilot.clickText({ text: target });
      return res.success;
    } catch (err: any) {
      throw new Error(`Ошибка нажатия на '${target}': ${err.message}`);
    }
  }

  /**
   * Вводит текст в активное поле ввода
   */
  public static async type(text: string): Promise<boolean> {
    try {
      const res = await NativeAutopilot.typeText({ text });
      return res.success;
    } catch (err: any) {
      throw new Error(`Ошибка ввода текста: ${err.message}`);
    }
  }

  /**
   * Возвращает на главный экран телефона
   */
  public static async pressHome(): Promise<boolean> {
    try {
      const res = await NativeAutopilot.pressHome();
      return res.success;
    } catch {
      return false;
    }
  }

  /**
   * Нажимает системную кнопку Назад
   */
  public static async pressBack(): Promise<boolean> {
    try {
      const res = await NativeAutopilot.pressBack();
      return res.success;
    } catch {
      return false;
    }
  }

  /**
   * Считывает элементы на экране
   */
  public static async getScreen(): Promise<UIElement[]> {
    try {
      const res = await NativeAutopilot.getScreenElements();
      return res.elements || [];
    } catch {
      return [];
    }
  }
}
