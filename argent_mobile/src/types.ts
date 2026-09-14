export type AIMode = 'online' | 'offline';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  thinking?: string;
  rethink?: string;
  timestamp: number;
  durationMs?: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  updatedAt: number;
}

export interface VirtualFile {
  id: string;
  name: string;
  content: string;
  updatedAt: number;
}

export interface AppSettings {
  mode: AIMode;
  online: {
    provider: 'deepseek' | 'openrouter' | 'groq' | 'openai' | 'ollama' | 'custom';
    endpoint: string;
    apiKey: string;
    model: string;
  };
  offline: {
    model: string;
    localFileName?: string;
    localFileSize?: string;
    customModelId?: string;
    threads?: number;
  };
  temperature: number;
  enableRethink: boolean;
}

export interface StreamCallbacks {
  onChunk?: (chunk: string) => void;
  onContent: (contentDelta: string, fullContent: string) => void;
  onThinking?: (thinkDelta: string, fullThinking: string) => void;
  onRethink?: (rethinkDelta: string, fullRethink: string) => void;
  onDone: (fullContent: string, fullThinking: string, fullRethink: string) => void;
  onError: (err: Error) => void;
  onPhase?: (phase: string, detail?: string) => void;
}

export interface DownloadProgress {
  progress: number;
  text: string;
}
