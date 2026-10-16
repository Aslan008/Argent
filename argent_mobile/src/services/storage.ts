import { AppSettings, ChatSession, VirtualFile, ChatMessage } from '../types';

const STORAGE_KEYS = {
  SETTINGS: 'argent_mobile_settings_v1',
  SESSIONS: 'argent_mobile_sessions_v1',
  CURRENT_SESSION_ID: 'argent_mobile_active_session_id',
  FILES: 'argent_mobile_files_v1'
};

const DEFAULT_SETTINGS: AppSettings = {
  mode: 'online',
  online: {
    provider: 'deepseek',
    endpoint: 'https://api.deepseek.com/v1',
    apiKey: '',
    model: 'deepseek-chat'
  },
  offline: {
    model: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC'
  },
  temperature: 0.6,
  enableRethink: true
};

/**
 * Сохранённые настройки поверх значений по умолчанию — с вложенными разделами.
 * Слияние `{...DEFAULT, ...saved}` было мелким: сохранённый раздел `online`/`offline` заменял раздел по
 * умолчанию целиком, и поле, добавленное в новой версии, у обновившегося пользователя было undefined.
 * Значения неверного типа (испорченное хранилище) не принимаются.
 */
export function mergeSettings(defaults: AppSettings, saved: any): AppSettings {
  const out: AppSettings = {
    ...defaults,
    online: { ...defaults.online },
    offline: { ...defaults.offline },
  };
  if (!saved || typeof saved !== 'object') return out;
  for (const key of Object.keys(saved)) {
    const value = saved[key];
    if (key === 'online' || key === 'offline') {
      if (value && typeof value === 'object') Object.assign(out[key], value);
    } else if (key in defaults && typeof value === typeof (defaults as any)[key]) {
      (out as any)[key] = value;
    }
  }
  if (out.mode !== 'online' && out.mode !== 'offline') out.mode = defaults.mode;
  return out;
}

/** Событие для интерфейса: сохранить не удалось (раньше это видно было только в консоли). */
export const STORAGE_PROBLEM_EVENT = 'argent-storage-problem';
function reportStorageProblem(message: string): void {
  try {
    window.dispatchEvent(new CustomEvent(STORAGE_PROBLEM_EVENT, { detail: message }));
  } catch {
    // нет window (тесты) — сообщить некому
  }
}

export class StorageService {
  // Настройки
  public static getSettings(): AppSettings {
    try {
      const raw = localStorage.getItem(STORAGE_KEYS.SETTINGS);
      return mergeSettings(DEFAULT_SETTINGS, raw ? JSON.parse(raw) : null);
    } catch {
      return mergeSettings(DEFAULT_SETTINGS, null);
    }
  }

  public static saveSettings(settings: AppSettings): void {
    try {
      localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
    } catch (e) {
      console.warn('Не удалось сохранить настройки в localStorage:', e);
    }
  }

  // Сессии диалогов
  public static getSessions(): ChatSession[] {
    try {
      const raw = localStorage.getItem(STORAGE_KEYS.SESSIONS);
      if (!raw) return [];
      return JSON.parse(raw);
    } catch {
      return [];
    }
  }

  public static saveSessions(sessions: ChatSession[]): void {
    try {
      localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(sessions));
    } catch (e) {
      console.warn('Превышена квота localStorage. Оптимизация истории сообщений:', e);
      try {
        const currentActiveId = this.getCurrentSessionId();
        // Если квота превышена, сжимаем историю неактивных старых сессий
        const trimmed = sessions.map((s, idx) => {
          const isActive = currentActiveId ? s.id === currentActiveId : idx === 0;
          if (isActive) return s; // Активную сессию не обрезаем
          return {
            ...s,
            messages: s.messages.slice(-20) // Оставляем последние 20 сообщений в архиве
          };
        });
        localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(trimmed));
      } catch (err) {
        console.error('Критическая ошибка сохранения сессий:', err);
        reportStorageProblem('Память браузера заполнена: последние сообщения не сохранились. Удалите старые диалоги в меню.');
      }
    }
  }

  public static getCurrentSessionId(): string | null {
    try {
      return localStorage.getItem(STORAGE_KEYS.CURRENT_SESSION_ID);
    } catch {
      return null;
    }
  }

  public static setCurrentSessionId(id: string): void {
    try {
      localStorage.setItem(STORAGE_KEYS.CURRENT_SESSION_ID, id);
    } catch (e) {
      console.warn('Не удалось сохранить текущий ID сессии:', e);
    }
  }

  public static getOrCreateActiveSession(): ChatSession {
    const sessions = this.getSessions();
    const currentId = this.getCurrentSessionId();
    let session = sessions.find(s => s.id === currentId);

    if (!session) {
      session = {
        id: 'chat_' + Date.now(),
        title: 'Новый диалог',
        messages: [],
        updatedAt: Date.now()
      };
      sessions.unshift(session);
      this.saveSessions(sessions);
      this.setCurrentSessionId(session.id);
    }
    return session;
  }

  public static updateSessionMessages(sessionId: string, messages: ChatMessage[], autoTitle?: string): void {
    const sessions = this.getSessions();
    const idx = sessions.findIndex(s => s.id === sessionId);
    if (idx !== -1) {
      sessions[idx].messages = messages;
      sessions[idx].updatedAt = Date.now();
      if (autoTitle && sessions[idx].title === 'Новый диалог') {
        sessions[idx].title = autoTitle.slice(0, 30);
      }
      this.saveSessions(sessions);
    }
  }

  // Файлы проекта
  public static getFiles(): VirtualFile[] {
    try {
      const raw = localStorage.getItem(STORAGE_KEYS.FILES);
      if (!raw) return [];
      return JSON.parse(raw);
    } catch {
      return [];
    }
  }

  public static saveFile(file: VirtualFile): void {
    try {
      const files = this.getFiles();
      const idx = files.findIndex(f => f.id === file.id);
      if (idx !== -1) {
        files[idx] = file;
      } else {
        files.push(file);
      }
      localStorage.setItem(STORAGE_KEYS.FILES, JSON.stringify(files));
    } catch (e) {
      console.warn('Не удалось сохранить файл в хранилище:', e);
      reportStorageProblem('Файл не сохранён: память браузера заполнена.');
    }
  }

  public static deleteFile(fileId: string): void {
    try {
      const files = this.getFiles().filter(f => f.id !== fileId);
      localStorage.setItem(STORAGE_KEYS.FILES, JSON.stringify(files));
    } catch (e) {
      console.warn('Не удалось удалить файл из хранилища:', e);
    }
  }
}
