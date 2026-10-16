import { AppSettings, ChatMessage, StreamCallbacks, DownloadProgress } from '../types';
import { StreamTagParser } from './fsm_parser';
import { LlamaNativeService } from './llama_native';
import { SseParser } from './sse';
import { fitToBudget, mergeConsecutive, PromptMessage } from './context';

let webllmModule: typeof import('@mlc-ai/web-llm') | null = null;
async function getWebLLM(): Promise<typeof import('@mlc-ai/web-llm')> {
  if (!webllmModule) {
    webllmModule = await import('@mlc-ai/web-llm');
  }
  return webllmModule;
}

/** История для облачной модели: окна у них 32k–128k, берём консервативно. */
export const ONLINE_HISTORY_TOKENS = 32000;
/** Модели WebLLM из списка собраны с окном 4096; часть уходит под ответ. */
export const WEBLLM_CONTEXT = 4096;
export const WEBLLM_REPLY_TOKENS = 1024;
/** Окно нативного llama.cpp по умолчанию (настраивается в offline.context). */
export const NATIVE_DEFAULT_CONTEXT = 2048;

/**
 * Файл модели для нативного llama.cpp (GGUF), а не id модели WebLLM.
 * Id моделей WebLLM по соглашению MLC содержат «MLC» (Qwen2.5-1.5B-Instruct-q4f16_1-MLC); всё остальное —
 * имя или путь файла (раньше это угадывалось по слову «nanbeige» в имени, и любая другая модель без
 * расширения уходила в WebLLM с ошибкой «модель не найдена»).
 */
export function isNativeModel(model: string, localFileName = ''): boolean {
  const m = (model || '').trim().toLowerCase();
  const f = (localFileName || '').trim().toLowerCase();
  if (m.endsWith('.gguf') || m.endsWith('.bin') || f.endsWith('.gguf') || f.endsWith('.bin') ||
      m.startsWith('/') || m.startsWith('file://') || m === 'custom_local') {
    return true;
  }
  return m.length > 0 && !m.includes('mlc');
}

/** Сколько токенов нативная модель может написать в ответ при окне `ctx`. */
export function nativeReplyTokens(ctx: number): number {
  return Math.max(128, Math.min(1024, Math.floor(ctx / 2)));
}

export function buildSystemPrompt(enableRethink: boolean): string {
  let systemPrompt = 'Ты — Argent Mobile, автономный интеллектуальный ассистент и инженер.\n' +
    'Отвечай пользователю структурированно, грамотно, без лишней воды и строго на русском языке.\n';
  if (enableRethink) {
    systemPrompt += '\nВсегда проводи глубокий анализ перед ответом:\n' +
      '1. Сначала открой тег <think> и подробно распиши ход своих мыслей, риски и граничные случаи, затем закрой </think>.\n' +
      '2. Затем открой тег <rethink>, критически перепроверь своё рассуждение на возможные ошибки, затем закрой </rethink>.\n' +
      '3. Затем выведи окончательный чистый ответ пользователю.';
  }
  return systemPrompt;
}

/** Запасной шаблон, когда модель не сообщает свой: ChatML (Qwen, Nanbeige и многие другие). */
export function chatmlPrompt(messages: PromptMessage[]): string {
  let out = '';
  for (const msg of messages) {
    out += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
  }
  return out + '<|im_start|>assistant\n';
}

export class AIEngine {
  private static offlineEngine: any = null;
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

    const system: PromptMessage = { role: 'system', content: buildSystemPrompt(settings.enableRethink) };
    const history: PromptMessage[] = mergeConsecutive(messages
      .filter(m => (m.role === 'user' || m.role === 'assistant') && m.content.trim())
      .map(m => ({ role: m.role, content: m.content })));

    try {
      callbacks.onPhase?.('Подготовка запроса...', settings.mode === 'online' ? settings.online.model : settings.offline.model);
      if (settings.mode === 'online') {
        const fit = this.fit(system, history, ONLINE_HISTORY_TOKENS, callbacks);
        await this.generateOnline(fit, settings, parser, abortController, callbacks);
      } else {
        await this.generateOffline(system, history, settings, parser, abortController, callbacks, onProgress);
      }

      parser.flush();
      callbacks.onDone(parser.fullContent, parser.fullThinking, parser.fullRethink);
    } catch (err: any) {
      if (abortController.signal.aborted) {
        parser.flush();
        callbacks.onDone(parser.fullContent, parser.fullThinking, parser.fullRethink);
        return;
      }
      callbacks.onError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      // Только свой: новая генерация, начатая до конца этой, уже поставила свой контроллер,
      // и обнулять его нельзя — иначе кнопка «Стоп» перестаёт её останавливать.
      if (this.activeAbortController === abortController) {
        this.activeAbortController = null;
      }
    }
  }

  private static fit(system: PromptMessage, history: PromptMessage[], budget: number,
                     callbacks: StreamCallbacks): PromptMessage[] {
    const fit = fitToBudget(system, history, budget);
    if (fit.dropped > 0 || fit.truncated) {
      const parts = [];
      if (fit.dropped > 0) parts.push(`${fit.dropped} старых сообщ. не вошли в окно модели`);
      if (fit.truncated) parts.push('последнее сообщение укорочено');
      callbacks.onNotice?.(parts.join('; '));
    }
    return fit.messages;
  }

  /**
   * Облачная потоковая генерация через OpenAI-совместимый API (DeepSeek, OpenRouter, Ollama и др.)
   */
  private static async generateOnline(
    messages: PromptMessage[],
    settings: AppSettings,
    parser: StreamTagParser,
    abortController: AbortController,
    callbacks: StreamCallbacks
  ): Promise<void> {
    callbacks.onPhase?.('Подключение к серверу...', `${settings.online.provider}: ${settings.online.model}`);
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

    callbacks.onPhase?.('Отправка запроса в облако...', 'Ожидание первого токена ответа');

    let response: Response;
    try {
      response = await fetch(endpoint, {
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
    } catch (err: any) {
      if (abortController.signal.aborted) throw err;
      throw new Error(explainFetchFailure(endpoint, err));
    }

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ошибка сервера (${response.status}): ${errorText || response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Ответ сервера не поддерживает потоковую передачу данных (ReadableStream).');
    }

    const decoder = new TextDecoder('utf-8');
    const sse = new SseParser();
    const apply = (deltas: { content?: string; reasoning?: string }[]) => {
      for (const d of deltas) {
        if (d.reasoning) parser.appendReasoning(d.reasoning);
        if (d.content) parser.feed(d.content);
      }
    };

    while (!sse.done) {
      const { done, value } = await reader.read();
      if (done) break;
      apply(sse.push(decoder.decode(value, { stream: true })));
    }
    apply(sse.push(decoder.decode()));
    apply(sse.finish());
    if (sse.done) {
      try { await reader.cancel(); } catch { /* поток уже закрыт */ }
    }
  }

  /**
   * Локальная генерация без интернета: GGUF через нативный llama.cpp (APK) или модель WebLLM (WebGPU).
   */
  private static async generateOffline(
    system: PromptMessage,
    history: PromptMessage[],
    settings: AppSettings,
    parser: StreamTagParser,
    abortController: AbortController,
    callbacks: StreamCallbacks,
    onProgress?: (p: DownloadProgress) => void
  ): Promise<void> {
    const targetModel = (settings.offline.model || '').trim();
    const localFileName = (settings.offline.localFileName || '').trim();

    if (isNativeModel(targetModel, localFileName)) {
      if (!LlamaNativeService.isAvailable()) {
        throw new Error(
          `Файл "${localFileName || targetModel}" (.gguf) запускается через нативный движок llama.cpp внутри установленного приложения Android (APK).\n\n` +
          `В браузере переключитесь на "🌐 Онлайн" (Ollama на ПК) или выберите модель из списка WebGPU.`
        );
      }

      callbacks.onPhase?.('Проверка доступа к памяти...', 'Системные разрешения Android');
      onProgress?.({ progress: 10, text: `Проверка доступа к памяти устройства...` });

      const hasPerm = await LlamaNativeService.checkStoragePermission();
      if (!hasPerm) {
        await LlamaNativeService.requestStoragePermission();
        throw new Error(
          '⚠️ Требуется доступ к файлам телефона:\n\n' +
          'Для чтения модели GGUF из памяти Android требует системное разрешение "Доступ ко всем файлам".\n' +
          'Мы открыли настройки Android. Пожалуйста, включите переключатель "Разрешить доступ к управлению всеми файлами" для Argent Mobile, затем вернитесь в приложение и повторите отправку.'
        );
      }

      const queryModelName = localFileName || targetModel;
      callbacks.onPhase?.('Поиск файла модели на диске...', queryModelName);
      onProgress?.({ progress: 25, text: `Поиск файла модели ${queryModelName}...` });

      let pathToLoad = targetModel;
      const resolved = await LlamaNativeService.resolvePath(queryModelName);
      if (resolved.found && resolved.path) {
        pathToLoad = resolved.path;
      }

      const threadsCount = settings.offline.threads || 2;
      const ctx = settings.offline.context || NATIVE_DEFAULT_CONTEXT;
      callbacks.onPhase?.('Загрузка в память (mmap)...', `llama.cpp • ${resolved.name || queryModelName}`);
      onProgress?.({ progress: 60, text: `Загрузка модели через llama.cpp...` });

      await LlamaNativeService.loadModel(pathToLoad, threadsCount, ctx);
      onProgress?.({ progress: 100, text: 'Модель готова к работе' });

      const replyTokens = nativeReplyTokens(ctx);
      const fitted = this.fit(system, history, ctx - replyTokens - 32, callbacks);
      // Шаблон чата берётся из самой модели (GGUF хранит его в метаданных): ChatML, вшитый в код, ломал
      // Llama 3, Gemma и Phi. Если модель шаблона не сообщает — ChatML как раньше.
      const prompt = (await LlamaNativeService.formatChat(fitted)) ?? chatmlPrompt(fitted);

      callbacks.onPhase?.('Генерация ответа...', `llama.cpp • ${threadsCount} потока`);

      const genResult = await LlamaNativeService.generateStream(
        prompt,
        settings.temperature,
        replyTokens,
        (delta) => parser.feed(delta),
        abortController.signal
      );

      if (genResult.tokensGenerated === 0 && !parser.fullContent && !parser.fullThinking && !abortController.signal.aborted) {
        throw new Error(
          '⚠️ Модель не выдала ни одного токена ответа. Возможно, повреждён файл модели (.gguf) или не хватает свободной оперативной памяти устройства.'
        );
      }
      return;
    }

    // Проверяем поддержку WebGPU на устройстве для моделей WebLLM
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

    // Инициализируем или переиспользуем загруженную модель
    const webllm = await getWebLLM();
    if (!this.offlineEngine || this.currentModelName !== targetModel) {
      this.offlineEngine = await webllm.CreateMLCEngine(targetModel, {
        initProgressCallback: (report: any) => {
          onProgress?.({
            progress: Math.round(report.progress * 100),
            text: report.text
          });
        }
      });
      this.currentModelName = targetModel;
    }

    const fitted = this.fit(system, history, WEBLLM_CONTEXT - WEBLLM_REPLY_TOKENS, callbacks);
    const chunks = await this.offlineEngine.chat.completions.create({
      messages: fitted as any,
      temperature: settings.temperature,
      max_tokens: WEBLLM_REPLY_TOKENS,
      stream: true
    });

    for await (const chunk of chunks) {
      if (abortController.signal.aborted) {
        try {
          await this.offlineEngine.interruptGenerate();
        } catch {
          // Игнорируем ошибки при прерывании
        }
        break;
      }
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

    if (isNativeModel(modelName)) {
      throw new Error(
        `Файл "${modelName}" имеет формат GGUF (для llama.cpp).\n\n` +
        `Модели формата .gguf запускаются нативно на процессоре смартфона напрямую из файлов устройства без интернета и не требуют загрузки в веб-кэш.`
      );
    }

    const webllm = await getWebLLM();
    this.offlineEngine = await webllm.CreateMLCEngine(modelName, {
      initProgressCallback: (report: any) => {
        onProgress({
          progress: Math.round(report.progress * 100),
          text: report.text
        });
      }
    });
    this.currentModelName = modelName;
  }
}

/** Понятное объяснение, почему запрос не ушёл (fetch бросает безликий TypeError "Failed to fetch"). */
export function explainFetchFailure(endpoint: string, err: any): string {
  const base = `Не удалось связаться с сервером ${endpoint} (${err?.message || err}).`;
  let url: URL | null = null;
  try { url = new URL(endpoint); } catch { /* не URL */ }
  if (!url) return `${base} Проверьте адрес сервера в настройках.`;
  const lan = /^(localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(url.hostname);
  if (lan) {
    return `${base}\n\nЭто адрес в домашней сети: проверьте, что телефон в том же Wi-Fi, сервер запущен и ` +
      `слушает сеть (для Ollama: переменная OLLAMA_HOST=0.0.0.0 и OLLAMA_ORIGINS=*), а брандмауэр ПК пропускает порт ${url.port || '80'}.`;
  }
  return `${base} Проверьте интернет, адрес и ключ API.`;
}
