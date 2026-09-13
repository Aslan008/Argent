import { AppSettings, ChatMessage, StreamCallbacks, DownloadProgress } from '../types';
import { StreamTagParser } from './fsm_parser';
import * as webllm from '@mlc-ai/web-llm';

export class AIEngine {
  private static offlineEngine: webllm.MLCEngine | null = null;
  private static currentModelName: string = '';
  private static activeAbortController: AbortController | null = null;

  public static abortCurrent(): void {
    if (this.activeAbortController) {
      this.activeAbortController.abort();
      this.activeAbortController = null;
    }
  }

  public static async generate(
    messages: ChatMessage[],
    settings: AppSettings,
    callbacks: StreamCallbacks,
    onProgress?: (p: DownloadProgress) => void
  ): Promise<void> {
    this.abortCurrent();
    const abortController = new AbortController();
    this.activeAbortController = abortController;

    const parser = new StreamTagParser(
      (delta) => callbacks.onContent(delta, parser.fullContent),
      (delta) => callbacks.onThinking && callbacks.onThinking(delta, parser.fullThinking),
      (delta) => callbacks.onRethink && callbacks.onRethink(delta, parser.fullRethink)
    );

    // Подготовка системного промпта в стиле Argent
    let systemPrompt = 'Ты — Argent Mobile, автономный интеллектуальный ассистент и инженер.\n' +
      'Отвечай пользователю структурированно, грамотно, без лишней воды и строго на русском языке.\n';

    if (settings.enableRethink) {
      systemPrompt += '\nВсегда проводи глубокий анализ перед ответом:\n' +
        '1. Сначала открой тег <think> и подробно распиши ход своих мыслей, риски и граничные случаи, затем закрой </think>.\n' +
        '2. Затем открой тег <rethink>, критически перепроверь своё рассуждение на возможные ошибки, затем закрой </rethink>.\n' +
        '3. Затем выведи окончательный чистый ответ пользователю.';
    }

    const formattedMessages = [
      { role: 'system', content: systemPrompt },
      ...messages.map(m => ({ role: m.role, content: m.content }))
    ];

    try {
      if (settings.mode === 'online') {
        await this.generateOnline(formattedMessages, settings, parser, abortController, callbacks);
      } else {
        await this.generateOffline(formattedMessages, settings, parser, abortController, onProgress);
      }

      parser.flush();
      callbacks.onDone(parser.fullContent, parser.fullThinking, parser.fullRethink);
    } catch (err: any) {
      if (abortController.signal.aborted) {
        parser.flush();
        callbacks.onDone(parser.fullContent, parser.fullThinking, parser.fullRethink);
        return;
      }
      callbacks.onError(err);
    } finally {
      this.activeAbortController = null;
    }
  }

  /**
   * Облачная потоковая генерация через OpenAI-совместимый API (DeepSeek, OpenRouter, Ollama и др.)
   */
  private static async generateOnline(
    messages: { role: string; content: string }[],
    settings: AppSettings,
    parser: StreamTagParser,
    abortController: AbortController,
    callbacks: StreamCallbacks
  ): Promise<void> {
    let endpoint = settings.online.endpoint.trim();
    if (!endpoint.endsWith('/chat/completions')) {
      endpoint = endpoint.replace(/\/+$/, '') + '/chat/completions';
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };

    if (settings.online.apiKey.trim()) {
      headers['Authorization'] = `Bearer ${settings.online.apiKey.trim()}`;
    }

    const response = await fetch(endpoint, {
      method: 'POST',
      headers,
      signal: abortController.signal,
      body: JSON.stringify({
        model: settings.online.model.trim(),
        messages,
        temperature: settings.temperature,
        stream: true
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ошибка сервера (${response.status}): ${errorText || response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Ответ сервера не поддерживает потоковую передачу данных (ReadableStream).');
    }

    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;
        if (trimmed === 'data: [DONE]') break;

        try {
          const json = JSON.parse(trimmed.slice(5).trim());
          const delta = json.choices?.[0]?.delta;
          
          // Некоторые провайдеры (DeepSeek reasoner) возвращают reasoning_content
          if (delta?.reasoning_content) {
            callbacks.onThinking && callbacks.onThinking(delta.reasoning_content, parser.fullThinking + delta.reasoning_content);
          } else if (delta?.content) {
            parser.feed(delta.content);
          }
        } catch {
          // Игнорируем неполные строки чанков
        }
      }
    }
  }

  /**
   * Локальная автономная генерация прямо на процессоре/видеочипе смартфона без интернета (WebLLM)
   */
  private static async generateOffline(
    messages: { role: string; content: string }[],
    settings: AppSettings,
    parser: StreamTagParser,
    abortController: AbortController,
    onProgress?: (p: DownloadProgress) => void
  ): Promise<void> {
    const targetModel = settings.offline.model;

    // Проверяем поддержку WebGPU на устройстве
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      const isHttp = typeof window !== 'undefined' && window.location.protocol === 'http:';
      let msg = 'Оффлайн-режим на чипе телефона требует графического ускорения WebGPU.\n\n';
      if (isHttp) {
        msg += '👉 Причина: Chrome на Android блокирует WebGPU при обычном подключении http:// (требуется HTTPS или скомпилированный APK).\n\n';
      }
      msg += 'Как решить прямо сейчас:\n' +
        '1. Переключите тумблер вверху на "🌐 Онлайн" — там модель на вашем ПК (Ollama) ответит моментально без требований к WebGPU;\n' +
        '2. Либо для оффлайна откройте в Chrome на телефоне адрес chrome://flags/#enable-unsafe-webgpu и выберите "Enabled".';
      throw new Error(msg);
    }

    // Проверяем формат файла модели
    if (targetModel.endsWith('.gguf') || targetModel.endsWith('.bin')) {
      throw new Error(
        `Файл "${targetModel}" имеет формат GGUF (движок llama.cpp / Ollama).\n\n` +
        `Встроенный оффлайн-движок смартфона работает через графический чип WebGPU и использует оптимизированные модели WebLLM (например, Qwen 2.5 1.5B или Llama 3.2 1B).\n\n` +
        `Как решить:\n` +
        `1. В Настройках выберите из списка готовую оффлайн-модель (рекомендуется "Qwen 2.5 1.5B" — она отлично знает русский);\n` +
        `2. Либо запустите этот .gguf файл на компьютере через Ollama и переключите тумблер вверху в "🌐 Онлайн".`
      );
    }

    // Инициализируем или переиспользуем загруженную модель
    if (!this.offlineEngine || this.currentModelName !== targetModel) {
      this.offlineEngine = await webllm.CreateMLCEngine(targetModel, {
        initProgressCallback: (report) => {
          if (onProgress) {
            onProgress({
              progress: Math.round(report.progress * 100),
              text: report.text
            });
          }
        }
      });
      this.currentModelName = targetModel;
    }

    const chunks = await this.offlineEngine.chat.completions.create({
      messages: messages as any,
      temperature: settings.temperature,
      stream: true
    });

    for await (const chunk of chunks) {
      if (abortController.signal.aborted) break;
      const text = chunk.choices[0]?.delta?.content || '';
      if (text) {
        parser.feed(text);
      }
    }
  }

  public static async preloadOfflineModel(
    modelName: string,
    onProgress: (p: DownloadProgress) => void
  ): Promise<void> {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      throw new Error('WebGPU не поддерживается этим устройством.');
    }

    if (modelName.endsWith('.gguf') || modelName.endsWith('.bin')) {
      throw new Error(
        `Файл "${modelName}" имеет формат GGUF (для llama.cpp / Ollama).\n` +
        `Для встроенного оффлайна на телефоне выберите модель из списка (например, Qwen 2.5 1.5B).`
      );
    }

    this.offlineEngine = await webllm.CreateMLCEngine(modelName, {
      initProgressCallback: (report) => {
        onProgress({
          progress: Math.round(report.progress * 100),
          text: report.text
        });
      }
    });
    this.currentModelName = modelName;
  }
}
