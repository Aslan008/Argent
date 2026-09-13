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

export class StorageService {
  // Настройки
  public static getSettings(): AppSettings {
    try {
      const raw = localStorage.getItem(STORAGE_KEYS.SETTINGS);
      if (!raw) return { ...DEFAULT_SETTINGS };
      return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
    } catch {
      return { ...DEFAULT_SETTINGS };
    }
  }

  public static saveSettings(settings: AppSettings): void {
    localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
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
    localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(sessions));
  }

  public static getCurrentSessionId(): string | null {
    return localStorage.getItem(STORAGE_KEYS.CURRENT_SESSION_ID);
  }

  public static setCurrentSessionId(id: string): void {
    localStorage.setItem(STORAGE_KEYS.CURRENT_SESSION_ID, id);
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
    const files = this.getFiles();
    const idx = files.findIndex(f => f.id === file.id);
    if (idx !== -1) {
      files[idx] = file;
    } else {
      files.push(file);
    }
    localStorage.setItem(STORAGE_KEYS.FILES, JSON.stringify(files));
  }

  public static deleteFile(fileId: string): void {
    const files = this.getFiles().filter(f => f.id !== fileId);
    localStorage.setItem(STORAGE_KEYS.FILES, JSON.stringify(files));
  }

  // Скачивание файла на физический накопитель смартфона
  public static downloadFileToDevice(filename: string, content: string): void {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}
