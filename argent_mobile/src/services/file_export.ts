import { registerPlugin, Capacitor } from '@capacitor/core';

interface ArgentFilesPlugin {
  saveText(options: { name: string; content: string }): Promise<{ path: string }>;
}

const ArgentFiles = registerPlugin<ArgentFilesPlugin>('ArgentFiles');

/**
 * Сохранение файлов и буфер обмена — так, чтобы работало и в браузере, и в APK.
 *
 * В WebView приложения `<a download>` с blob-ссылкой ничего не скачивает, поэтому в APK файл пишет нативный
 * плагин ArgentFiles в «Загрузки/Argent». navigator.clipboard существует только в защищённом контексте
 * (https / localhost): по http://192.168.x.x:3000 кнопка «Копировать» падала с ошибкой.
 */
export class FileExport {
  /** Сохранить текст в файл. Возвращает, куда он попал (для сообщения пользователю). */
  public static async saveText(filename: string, content: string): Promise<string> {
    if (Capacitor.isNativePlatform() && Capacitor.isPluginAvailable('ArgentFiles')) {
      const res = await ArgentFiles.saveText({ name: filename, content });
      return res.path;
    }
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // отзыв ссылки откладывается: некоторые браузеры начинают скачивание асинхронно
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    return 'папку загрузок браузера';
  }

  /** Скопировать текст; true — получилось. */
  public static async copy(text: string): Promise<boolean> {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch {
      // запасной путь ниже
    }
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand('copy');
    } catch {
      ok = false;
    }
    document.body.removeChild(area);
    return ok;
  }
}
