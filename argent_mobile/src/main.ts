import { marked } from 'marked';
// Только распространённые языки: полный highlight.js — это ~1 МБ из 1,08 МБ основного бандла
import hljs from 'highlight.js/lib/common';
import 'highlight.js/styles/atom-one-dark.min.css';
import { Capacitor } from '@capacitor/core';
import { StorageService, STORAGE_PROBLEM_EVENT } from './services/storage';
import { AIEngine, isNativeModel } from './services/ai_engine';
import { AutopilotService } from './services/autopilot';
import { LlamaNativeService } from './services/llama_native';
import { FileExport } from './services/file_export';
import { AppSettings, ChatMessage, ChatSession, VirtualFile } from './types';

// Конфигурация Markdown рендерера с подсветкой синтаксиса
marked.setOptions({
  breaks: true,
  gfm: true
});

/**
 * Один ответ ассистента: его пузырь, таймер и сессия.
 * Раньше всё это жило в полях приложения (activeTextContent и т. п.): ответ, прерванный кнопкой «Стоп»,
 * продолжал присылать события уже ПОСЛЕ отправки следующего сообщения и писал в чужой пузырь, а его
 * сообщение вставало в историю после нового вопроса.
 */
interface Turn {
  bubble: HTMLElement;
  thinking: HTMLElement | null;
  rethink: HTMLElement | null;
  text: HTMLElement | null;
  timer: ReturnType<typeof setInterval> | null;
  startedAt: number;
  session: ChatSession;
  userMsgId: string;
  assistantMsgId: string;
  firstToken: boolean;
}

/** Как подписать движок, честно для каждого режима (раньше везде было «Helio G99 ARM NEON», даже для облака). */
function engineInfo(settings: AppSettings): { chip: string; phase: string; detail: string; name: string } {
  if (settings.mode === 'online') {
    return {
      chip: `🌐 ${settings.online.provider.toUpperCase()}`,
      phase: 'Соединение с сервером...',
      detail: 'Ожидание ответа модели...',
      name: settings.online.model,
    };
  }
  const name = settings.offline.localFileName || settings.offline.model;
  if (isNativeModel(settings.offline.model, settings.offline.localFileName)) {
    return { chip: '⚡ llama.cpp', phase: 'Запуск нативного движка...', detail: 'Вычисление на процессоре телефона', name };
  }
  return { chip: '⚡ WebLLM', phase: 'Запуск WebLLM...', detail: 'Вычисление на видеочипе (WebGPU)', name };
}

class ArgentMobileApp {
  private settings: AppSettings;
  private currentSession: ChatSession;
  private isStreaming: boolean = false;
  private isAutopilotEnabled: boolean = false;
  private turn: Turn | null = null;

  constructor() {
    this.settings = StorageService.getSettings();
    this.currentSession = StorageService.getOrCreateActiveSession();

    this.initDOM();
    this.renderCurrentSession();
    this.updateUIState();

    let lastWarned = 0;
    window.addEventListener(STORAGE_PROBLEM_EVENT, (e) => {
      if (Date.now() - lastWarned < 60_000) return;     // не чаще раза в минуту
      lastWarned = Date.now();
      alert((e as CustomEvent<string>).detail);
    });
  }

  private initDOM(): void {
    // Режим работы (Онлайн / Оффлайн)
    const btnOnline = document.getElementById('mode-online-btn');
    const btnOffline = document.getElementById('mode-offline-btn');
    btnOnline?.addEventListener('click', () => this.switchMode('online'));
    btnOffline?.addEventListener('click', () => this.switchMode('offline'));

    // Режим перепроверки (Rethink toggle)
    const toggleRethink = document.getElementById('toggle-rethink-btn');
    toggleRethink?.addEventListener('click', () => {
      this.settings.enableRethink = !this.settings.enableRethink;
      toggleRethink.classList.toggle('active', this.settings.enableRethink);
      StorageService.saveSettings(this.settings);
    });

    // Режим автопилота телефона
    const toggleAutopilot = document.getElementById('toggle-autopilot-btn');
    toggleAutopilot?.addEventListener('click', async () => {
      this.isAutopilotEnabled = !this.isAutopilotEnabled;
      toggleAutopilot.classList.toggle('active', this.isAutopilotEnabled);

      if (this.isAutopilotEnabled) {
        const isRunning = await AutopilotService.isServiceEnabled();
        if (!isRunning) {
          const confirmOpen = confirm(
            'Для работы автопилота необходимо включить службу "Argent Autopilot Service" в специальных возможностях Android.\n\nОткрыть настройки телефона прямо сейчас?'
          );
          if (confirmOpen) {
            await AutopilotService.openSettings();
          }
        }
      }
    });

    // Отправка сообщений
    const btnSend = document.getElementById('btn-send');
    const userInput = document.getElementById('user-input') as HTMLTextAreaElement;

    btnSend?.addEventListener('click', () => {
      if (this.isStreaming) {
        this.abortGeneration();
      } else {
        this.handleSend();
      }
    });

    userInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleSend();
      }
    });

    // Авто-высота textarea
    userInput?.addEventListener('input', () => {
      userInput.style.height = 'auto';
      userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    });

    // Быстрые подсказки на экране приветствия
    document.querySelectorAll('.quick-prompt-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const prompt = (e.currentTarget as HTMLElement).dataset.prompt;
        if (prompt && userInput) {
          userInput.value = prompt;
          this.handleSend();
        }
      });
    });

    // Модальные окна и шторки
    this.initModals();
  }

  private switchMode(mode: 'online' | 'offline'): void {
    this.settings.mode = mode;
    StorageService.saveSettings(this.settings);
    this.updateUIState();
  }

  private updateUIState(): void {
    const btnOnline = document.getElementById('mode-online-btn');
    const btnOffline = document.getElementById('mode-offline-btn');
    const modelTag = document.getElementById('active-model-tag');
    const toggleRethink = document.getElementById('toggle-rethink-btn');

    if (this.settings.mode === 'online') {
      btnOnline?.classList.add('active');
      btnOffline?.classList.remove('active');
      if (modelTag) modelTag.textContent = this.settings.online.model || 'cloud-api';
    } else {
      btnOnline?.classList.remove('active');
      btnOffline?.classList.add('active');
      let shortName = this.settings.offline.localFileName || this.settings.offline.model;
      if (shortName.endsWith('.gguf')) {
        shortName = shortName.replace('.gguf', '') + ' (gguf)';
      } else {
        shortName = shortName.split('-')[0] + ' (local)';
      }
      if (modelTag) modelTag.textContent = shortName;
    }

    if (toggleRethink) {
      toggleRethink.classList.toggle('active', this.settings.enableRethink);
    }
  }

  private async handleSend(): Promise<void> {
    const input = document.getElementById('user-input') as HTMLTextAreaElement;
    const text = input.value.trim();
    if (!text || this.isStreaming) return;

    input.value = '';
    input.style.height = 'auto';

    // Скрываем карточку приветствия
    const welcomeCard = document.getElementById('welcome-card');
    if (welcomeCard) welcomeCard.classList.add('hidden');

    // Сообщение пользователя сохраняется сразу: при ошибке или закрытии приложения посреди ответа
    // вопрос раньше пропадал из истории (он записывался только вместе с готовым ответом).
    const session = this.currentSession;
    const userMsg: ChatMessage = {
      id: 'msg_' + Date.now(),
      role: 'user',
      content: text,
      timestamp: Date.now()
    };
    session.messages.push(userMsg);
    StorageService.updateSessionMessages(session.id, session.messages, userMsg.content);
    this.renderMessage(userMsg);

    // Заготовка под ответ ассистента
    const turn = this.prepareAssistantBubble(session, userMsg.id, 'msg_' + (Date.now() + 1));

    // Проверка команд автопилота телефона
    const lowerText = text.toLowerCase().trim();
    const isExplicitNavigation = lowerText === 'домой' || lowerText === 'назад' || lowerText.includes('нажми домой');
    const isExplicitAppLaunch = lowerText.startsWith('открой приложение ') ||
                                lowerText.startsWith('запусти приложение ') ||
                                lowerText.includes('открой приложение на телефоне');
    const hasAutopilotPrefix = lowerText.startsWith('🚀') || lowerText.startsWith('автопилот:') || lowerText.startsWith('автопилот,');

    const shouldHandleWithAutopilot = isExplicitNavigation || isExplicitAppLaunch || (this.isAutopilotEnabled && hasAutopilotPrefix);

    if (shouldHandleWithAutopilot && await this.runAutopilotCommand(turn, text, lowerText)) {
      return;
    }

    this.setStreamingState(true);
    const banner = document.getElementById('model-download-banner');
    const bannerPercent = document.getElementById('banner-percent-text');
    const bannerFill = document.getElementById('banner-progress-fill');
    const bannerText = document.getElementById('banner-status-text');
    const engine = engineInfo(this.settings);
    const mode = this.settings.mode;

    let liveTokenCount = 0;
    let lastContentRenderTime = 0;
    const statusEl = turn.bubble.querySelector('.thinking-label');
    const detailEl = turn.bubble.querySelector('.thinking-detail-text');
    const firstToken = () => {
      if (!turn.firstToken) {
        turn.firstToken = true;
        this.vibrate(15);
      }
    };
    const speed = () => (liveTokenCount / Math.max(0.1, (Date.now() - turn.startedAt) / 1000)).toFixed(1);

    try {
      await AIEngine.generate(
        session.messages,
        this.settings,
        {
          onPhase: (phase, detail) => {
            if (statusEl && phase) statusEl.textContent = phase;
            if (detailEl && detail) detailEl.textContent = detail;
          },
          onNotice: (note) => {
            this.addTurnNote(turn, note);
          },
          onContent: (_delta, fullContent) => {
            liveTokenCount++;
            firstToken();
            if (statusEl) statusEl.textContent = 'Вывод ответа...';
            if (detailEl) detailEl.textContent = `Токен #${liveTokenCount} (${speed()} т/с)`;

            if (turn.text) {
              turn.text.classList.remove('is-generating');
              const now = Date.now();
              // Троттлинг отрисовки во время стриминга (~80мс), чтобы не перегружать поток интерфейса
              if (now - lastContentRenderTime > 80) {
                lastContentRenderTime = now;
                turn.text.innerHTML = this.renderMarkdown(fullContent, true);
                this.scrollToBottom();
              }
            }
          },
          onThinking: (_delta, fullThinking) => {
            liveTokenCount++;
            firstToken();
            if (turn.thinking) {
              turn.thinking.textContent = fullThinking;
              turn.thinking.closest('.thought-card')?.classList.remove('hidden');
              if (statusEl) statusEl.textContent = 'Формирование хода мыслей...';
              if (detailEl) detailEl.textContent = `Мысль #${liveTokenCount} (${speed()} т/с)`;
              this.scrollToBottom();
            }
          },
          onRethink: (_delta, fullRethink) => {
            if (turn.rethink) {
              turn.rethink.textContent = fullRethink;
              turn.rethink.closest('.thought-card')?.classList.remove('hidden');
              this.scrollToBottom();
            }
          },
          onDone: (fullContent, fullThinking, fullRethink) => {
            this.stopTurnTimer(turn);

            if (!fullContent.trim() && !fullThinking.trim()) {
              this.showTurnProblem(turn, '⚠️ Ответ не сгенерирован (0 токенов)',
                'Модель завершила вычисления, но не сформировала текст. Повторите отправку; если это локальная модель — проверьте файл модели и свободную память.');
              this.finalizeAssistantBubble(turn);
              return;
            }

            if (turn.text) {
              turn.text.classList.remove('is-generating');
              if (fullContent.trim()) {
                turn.text.innerHTML = this.renderMarkdown(fullContent, false);
                this.attachCodeActionListeners(turn.text);
              } else {
                turn.text.innerHTML = `
                  <div class="notice-box info">
                    🧠 <b>Рассуждения сформированы:</b> модель закончила ход мыслей, но вывод ответа был прерван. Раскройте блок «Ход мыслей» выше.
                  </div>
                `;
              }
              this.scrollToBottom();
            }

            const durationMs = Date.now() - turn.startedAt;
            const assistantMsg: ChatMessage = {
              id: turn.assistantMsgId,
              role: 'assistant',
              content: fullContent,
              thinking: fullThinking || undefined,
              rethink: fullRethink || undefined,
              timestamp: Date.now(),
              durationMs,
              engine: engine.name,
              mode
            };
            this.storeAssistantMessage(turn, assistantMsg);
            this.finalizeAssistantBubble(turn, (durationMs / 1000).toFixed(1), engine.name, mode);
            this.vibrate([15, 30, 15]);
          },
          onError: (err) => {
            this.stopTurnTimer(turn);
            this.showTurnProblem(turn, '⚠️ Ошибка:', err.message);
            this.finalizeAssistantBubble(turn);
          }
        },
        (progress) => {
          if (banner && bannerPercent && bannerFill && bannerText) {
            banner.classList.remove('hidden');
            bannerPercent.textContent = `${progress.progress}%`;
            bannerFill.style.width = `${progress.progress}%`;
            bannerText.textContent = progress.text;

            if (progress.progress >= 100) {
              setTimeout(() => banner.classList.add('hidden'), 2000);
            }
          }
        }
      );
    } catch (err: any) {
      this.stopTurnTimer(turn);
      this.showTurnProblem(turn, '⚠️ Сбой:', err?.message || String(err));
      this.finalizeAssistantBubble(turn);
    } finally {
      // Состояние «идёт генерация» принадлежит последнему ответу: прерванный старый, закончившись позже,
      // не должен снимать его с нового.
      if (this.turn === turn) {
        this.setStreamingState(false);
        banner?.classList.add('hidden');
      }
    }
  }

  /** Команда телефону (открыть приложение, «домой», «назад»). true — обработано, в модель не идёт. */
  private async runAutopilotCommand(turn: Turn, text: string, lowerText: string): Promise<boolean> {
    const isRunning = await AutopilotService.isServiceEnabled();
    if (!isRunning) {
      if (turn.text) {
        turn.text.classList.remove('is-generating');
        turn.text.innerHTML = `
          <div class="notice-box warn">
            <div class="notice-title">⚠️ Служба автопилота отключена в системе</div>
            <div class="notice-body">
              Чтобы Argent мог самостоятельно открывать приложения и нажимать кнопки на телефоне, включите службу <b>Argent Autopilot Service</b> в специальных возможностях Android.
            </div>
            <button class="action-btn-sm btn-open-acc-settings">⚙️ Включить в настройках телефона</button>
          </div>
        `;
        turn.text.querySelector('.btn-open-acc-settings')?.addEventListener('click', () => {
          AutopilotService.openSettings();
        });
      }
      this.stopTurnTimer(turn);
      this.finalizeAssistantBubble(turn);
      this.setStreamingState(false);
      return true;
    }

    let reply = '';
    if (lowerText.startsWith('открой ') || lowerText.startsWith('запусти ') || lowerText.includes('открой приложение ')) {
      const appMatch = text.match(/(?:открой|запусти)(?:\s+приложение)?\s+([a-zA-Zа-яА-ЯёЁ0-9\s]+)/i);
      const appName = appMatch ? appMatch[1].replace(/на телефоне/i, '').replace(/приложение/i, '').trim() : '';
      if (appName) {
        if (turn.thinking) {
          turn.thinking.textContent = `Анализ цели: запуск приложения "${appName}" через Android PackageManager.`;
          turn.thinking.closest('.thought-card')?.classList.remove('hidden');
        }
        try {
          await AutopilotService.launchApp(appName);
          reply = `🚀 <b>Автопилот:</b> приложение <b>${this.escapeHtml(appName)}</b> открыто на телефоне.`;
        } catch (err: any) {
          reply = `⚠️ <b>Ошибка автопилота:</b> ${this.escapeHtml(err.message)}`;
        }
      }
    } else if (lowerText.includes('нажми домой') || lowerText === 'домой') {
      reply = await AutopilotService.pressHome()
        ? '🚀 <b>Автопилот:</b> переход на главный экран выполнен.'
        : '⚠️ <b>Автопилот:</b> система не выполнила «Домой».';
    } else if (lowerText.includes('назад')) {
      reply = await AutopilotService.pressBack()
        ? '🚀 <b>Автопилот:</b> нажата кнопка «Назад».'
        : '⚠️ <b>Автопилот:</b> система не выполнила «Назад».';
    }

    if (!reply) return false;
    if (turn.text) {
      turn.text.classList.remove('is-generating');
      turn.text.innerHTML = reply;
    }
    this.storeAssistantMessage(turn, {
      id: turn.assistantMsgId,
      role: 'assistant',
      content: turn.text?.textContent || 'Команда выполнена',
      timestamp: Date.now(),
      engine: 'Автопилот'
    });
    this.stopTurnTimer(turn);
    this.finalizeAssistantBubble(turn);
    this.setStreamingState(false);
    return true;
  }

  /** Ответ встаёт сразу за своим вопросом (а не в конец: после «Стоп» мог уже прийти новый вопрос)
   *  и пишется в СВОЮ сессию — пользователь мог переключиться на другой диалог. */
  private storeAssistantMessage(turn: Turn, msg: ChatMessage): void {
    const messages = turn.session.messages;
    const at = messages.findIndex(m => m.id === turn.userMsgId);
    let insertAt = at === -1 ? messages.length : at + 1;
    while (insertAt < messages.length && messages[insertAt].role === 'assistant') insertAt++;
    messages.splice(insertAt, 0, msg);
    StorageService.updateSessionMessages(turn.session.id, messages);
    // Тот же диалог мог быть открыт заново (другой объект из хранилища): иначе следующее сохранение
    // из него затёрло бы этот ответ.
    if (this.currentSession.id === turn.session.id && this.currentSession !== turn.session) {
      this.currentSession.messages = messages;
    }
  }

  private showTurnProblem(turn: Turn, title: string, body: string): void {
    if (!turn.text) return;
    turn.text.classList.remove('is-generating');
    turn.text.innerHTML = `
      <div class="notice-box error">
        <div class="notice-title">${this.escapeHtml(title)}</div>
        <div class="notice-body">${this.escapeHtml(body)}</div>
      </div>
    `;
  }

  private addTurnNote(turn: Turn, note: string): void {
    const body = turn.bubble.querySelector('.assistant-body');
    if (!body) return;
    const el = document.createElement('div');
    el.className = 'turn-note';
    el.textContent = `ℹ️ ${note}`;
    body.insertBefore(el, body.firstChild);
  }

  private stopTurnTimer(turn: Turn): void {
    if (turn.timer) {
      clearInterval(turn.timer);
      turn.timer = null;
    }
  }

  private vibrate(pattern: number | number[]): void {
    if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
      try { navigator.vibrate(pattern); } catch { /* вибрация недоступна */ }
    }
  }

  private abortGeneration(): void {
    AIEngine.abortCurrent();
    this.setStreamingState(false);
  }

  private setStreamingState(streaming: boolean): void {
    this.isStreaming = streaming;
    const btnSend = document.getElementById('btn-send');
    const iconSend = document.getElementById('icon-send');
    const iconStop = document.getElementById('icon-stop');

    if (streaming) {
      btnSend?.classList.add('streaming');
      iconSend?.classList.add('hidden');
      iconStop?.classList.remove('hidden');
    } else {
      btnSend?.classList.remove('streaming');
      iconSend?.classList.remove('hidden');
      iconStop?.classList.add('hidden');
    }
  }

  private prepareAssistantBubble(session: ChatSession, userMsgId: string, msgId: string): Turn {
    const list = document.getElementById('messages-list');
    const item = document.createElement('div');
    item.className = 'message-item assistant';
    item.id = msgId;
    const engine = engineInfo(this.settings);

    item.innerHTML = `
      <div class="assistant-body">
        <!-- Thinking Card -->
        <div class="thought-card thinking hidden">
          <div class="thought-card-header">
            <span class="thought-card-title">
              <span class="thought-indicator"></span>
              🧠 Ход мыслей
            </span>
            <span class="collapse-icon">▼</span>
          </div>
          <div class="thought-card-content thought-content-text"></div>
        </div>

        <!-- Rethink Card -->
        <div class="thought-card rethink hidden">
          <div class="thought-card-header">
            <span class="thought-card-title">
              <span class="thought-indicator"></span>
              🔍 Перепроверка решения
            </span>
            <span class="collapse-icon">▼</span>
          </div>
          <div class="thought-card-content rethink-content-text"></div>
        </div>

        <!-- Main Answer Content with Animated Thinking State -->
        <div class="assistant-content is-generating">
          <div class="thinking-state-box">
            <div class="thinking-top-row">
              <div class="thinking-orbit">
                <span class="thinking-orbit-icon">🧠</span>
              </div>
              <div class="thinking-info">
                <div class="thinking-status-line">
                  <span class="thinking-label">${this.escapeHtml(engine.phase)}</span>
                  <span class="thinking-timer">0.0с</span>
                </div>
                <div class="thinking-detail-row">
                  <span class="thinking-chip-engine">${this.escapeHtml(engine.chip)}</span>
                  <div class="thinking-dots"><span></span><span></span><span></span></div>
                  <span class="thinking-detail-text">${this.escapeHtml(engine.detail)}</span>
                </div>
              </div>
            </div>
            <div class="thinking-shimmer-bar"></div>
          </div>
        </div>
      </div>
    `;

    list?.appendChild(item);
    const turn: Turn = {
      bubble: item,
      thinking: item.querySelector('.thought-content-text'),
      rethink: item.querySelector('.rethink-content-text'),
      text: item.querySelector('.assistant-content'),
      timer: null,
      startedAt: Date.now(),
      session,
      userMsgId,
      assistantMsgId: msgId,
      firstToken: false,
    };
    this.turn = turn;

    this.attachThoughtCollapsers(item);
    this.scrollToBottom();

    // Живой таймер размышления (100 мс) — у каждого ответа свой
    const timerEl = item.querySelector('.thinking-timer');
    turn.timer = setInterval(() => {
      if (timerEl) timerEl.textContent = `${((Date.now() - turn.startedAt) / 1000).toFixed(1)}с`;
    }, 100);

    this.vibrate(20);
    return turn;
  }

  private finalizeAssistantBubble(turn: Turn, durationSec?: string, engineName?: string,
                                  mode?: 'online' | 'offline'): void {
    this.stopTurnTimer(turn);
    turn.text?.classList.remove('is-generating');

    // Авто-сворачивание хода мыслей после завершения генерации
    turn.bubble.querySelector('.thought-card.thinking')?.classList.add('collapsed');
    turn.bubble.querySelector('.thought-card.rethink')?.classList.add('collapsed');
    turn.bubble.querySelectorAll('.thought-indicator').forEach(el => el.remove());

    if (durationSec && !turn.bubble.querySelector('.message-meta-badge')) {
      turn.bubble.querySelector('.assistant-body')?.appendChild(this.metaBadge(durationSec, engineName, mode));
    }
  }

  private metaBadge(durationSec: string, engineName?: string, mode?: 'online' | 'offline'): HTMLElement {
    const badge = document.createElement('div');
    badge.className = 'message-meta-badge';
    const parts = [`<span class="meta-icon">${mode === 'online' ? '🌐' : '⚡'}</span>`, `<span>${this.escapeHtml(durationSec)}с</span>`];
    if (engineName) parts.push('<span>•</span>', `<span>${this.escapeHtml(engineName)}</span>`);
    badge.innerHTML = parts.join('');
    return badge;
  }

  private attachThoughtCollapsers(container: HTMLElement): void {
    container.querySelectorAll('.thought-card-header').forEach((header) => {
      header.addEventListener('click', () => {
        const card = header.closest('.thought-card');
        card?.classList.toggle('collapsed');
      });
    });
  }

  private renderCurrentSession(): void {
    const list = document.getElementById('messages-list');
    const welcome = document.getElementById('welcome-card');
    if (!list) return;

    list.innerHTML = '';

    if (this.currentSession.messages.length === 0) {
      welcome?.classList.remove('hidden');
    } else {
      welcome?.classList.add('hidden');
      for (const msg of this.currentSession.messages) {
        this.renderMessage(msg);
      }
    }
    this.scrollToBottom();
  }

  private renderMessage(msg: ChatMessage): void {
    const list = document.getElementById('messages-list');
    if (!list) return;

    const item = document.createElement('div');
    item.className = `message-item ${msg.role}`;
    item.id = msg.id;

    if (msg.role === 'user') {
      item.innerHTML = `<div class="user-bubble">${this.escapeHtml(msg.content)}</div>`;
    } else {
      const hasThinking = Boolean(msg.thinking);
      const hasRethink = Boolean(msg.rethink);
      const durationSec = msg.durationMs ? (msg.durationMs / 1000).toFixed(1) : '';

      item.innerHTML = `
        <div class="assistant-body">
          ${hasThinking ? `
            <div class="thought-card thinking collapsed">
              <div class="thought-card-header">
                <span class="thought-card-title">🧠 Ход мыслей</span>
                <span class="collapse-icon">▼</span>
              </div>
              <div class="thought-card-content">${this.escapeHtml(msg.thinking || '')}</div>
            </div>
          ` : ''}

          ${hasRethink ? `
            <div class="thought-card rethink collapsed">
              <div class="thought-card-header">
                <span class="thought-card-title">🔍 Перепроверка</span>
                <span class="collapse-icon">▼</span>
              </div>
              <div class="thought-card-content">${this.escapeHtml(msg.rethink || '')}</div>
            </div>
          ` : ''}

          <div class="assistant-content">${this.renderMarkdown(msg.content)}</div>
        </div>
      `;
      if (durationSec) {
        item.querySelector('.assistant-body')?.appendChild(this.metaBadge(durationSec, msg.engine, msg.mode));
      }

      this.attachThoughtCollapsers(item);
      this.attachCodeActionListeners(item);
    }

    list.appendChild(item);
  }

  private sanitizeMarkdownHtml(html: string): string {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');

    const ALLOWED_TAGS = new Set([
      'p', 'br', 'strong', 'em', 'b', 'i', 'u', 's', 'del', 'mark',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li', 'blockquote', 'pre', 'code', 'kbd',
      'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td',
      'hr', 'span', 'div', 'a'
    ]);

    const ALLOWED_PROTOCOLS = ['http:', 'https:', 'mailto:', 'tel:'];

    const cleanNode = (node: Node) => {
      const children = Array.from(node.childNodes);
      for (const child of children) {
        if (child.nodeType === Node.ELEMENT_NODE) {
          const element = child as HTMLElement;
          const tagName = element.tagName.toLowerCase();

          if (!ALLOWED_TAGS.has(tagName)) {
            if (['script', 'iframe', 'object', 'embed', 'style', 'svg', 'math', 'link', 'meta', 'base', 'form', 'input', 'textarea', 'button', 'img'].includes(tagName)) {
              element.remove();
              continue;
            }
            const textNode = doc.createTextNode(element.textContent || '');
            node.replaceChild(textNode, element);
            continue;
          }

          // Очистка опасных атрибутов (обработчики событий, inline стили, src)
          const attrNames = element.getAttributeNames();
          for (const attr of attrNames) {
            const attrLower = attr.toLowerCase();
            if (attrLower.startsWith('on') || attrLower === 'style' || attrLower === 'src' || attrLower === 'srcset') {
              element.removeAttribute(attr);
              continue;
            }

            if (tagName === 'a' && attrLower === 'href') {
              const val = element.getAttribute('href') || '';
              try {
                const parsed = new URL(val, window.location.origin);
                if (!ALLOWED_PROTOCOLS.includes(parsed.protocol) && !val.startsWith('#')) {
                  element.removeAttribute('href');
                } else {
                  element.setAttribute('target', '_blank');
                  element.setAttribute('rel', 'noopener noreferrer');
                }
              } catch {
                if (!val.startsWith('#')) {
                  element.removeAttribute('href');
                }
              }
            } else if (attrLower !== 'class' && attrLower !== 'title') {
              element.removeAttribute(attr);
            }
          }

          cleanNode(element);
        }
      }
    };

    cleanNode(doc.body);
    return doc.body.innerHTML;
  }

  private renderMarkdown(raw: string, isStreaming: boolean = false): string {
    const rawHtml = marked.parse(raw) as string;
    const cleanHtml = this.sanitizeMarkdownHtml(rawHtml);
    const temp = document.createElement('div');
    temp.innerHTML = cleanHtml;

    // Подсветка и обёртка блоков кода
    temp.querySelectorAll('pre code').forEach((codeBlock) => {
      const codeElement = codeBlock as HTMLElement;
      if (!isStreaming) {
        hljs.highlightElement(codeElement);
      }

      const pre = codeElement.parentElement;
      if (!pre) return;

      const lang = (codeElement.className.match(/language-(\w+)/) || [])[1] || 'code';
      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      const header = document.createElement('div');
      header.className = 'code-header';
      header.innerHTML = `
        <span class="code-lang">${lang}</span>
        <div class="code-actions">
          <button class="code-action-btn btn-copy-code">📋 Копировать</button>
          <button class="code-action-btn btn-save-project" data-lang="${lang}">💾 В проект</button>
        </div>
      `;

      pre.parentNode?.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);
    });

    return temp.innerHTML;
  }

  private attachCodeActionListeners(container: HTMLElement): void {
    container.querySelectorAll('.btn-copy-code').forEach((btn) => {
      btn.addEventListener('click', async (e) => {
        const target = e.currentTarget as HTMLElement;
        const codeWrapper = target.closest('.code-block-wrapper');
        const code = codeWrapper?.querySelector('pre code')?.textContent || '';
        const ok = await FileExport.copy(code);
        target.textContent = ok ? '✓ Скопировано!' : '✕ Не удалось скопировать';
        setTimeout(() => target.textContent = '📋 Копировать', 2000);
      });
    });

    container.querySelectorAll('.btn-save-project').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const target = e.currentTarget as HTMLElement;
        const lang = target.dataset.lang || 'txt';
        const codeWrapper = target.closest('.code-block-wrapper');
        const code = codeWrapper?.querySelector('pre code')?.textContent || '';

        const extMap: Record<string, string> = {
          python: 'py',
          javascript: 'js',
          typescript: 'ts',
          html: 'html',
          css: 'css',
          json: 'json'
        };
        const ext = extMap[lang.toLowerCase()] || lang || 'txt';
        const filename = `snippet_${Date.now().toString().slice(-4)}.${ext}`;

        const newFile: VirtualFile = {
          id: 'file_' + Date.now(),
          name: filename,
          content: code,
          updatedAt: Date.now()
        };

        StorageService.saveFile(newFile);
        target.textContent = '✓ Сохранено в файлы!';
        setTimeout(() => target.textContent = '💾 В проект', 2000);
      });
    });
  }

  private initModals(): void {
    // Шторка меню (список диалогов)
    const btnMenu = document.getElementById('btn-menu');
    const menuDrawer = document.getElementById('menu-drawer');
    const btnCloseMenu = document.getElementById('btn-close-menu');
    const btnDrawerNewChat = document.getElementById('drawer-new-chat-btn');
    const btnNewChat = document.getElementById('btn-new-chat');

    btnMenu?.addEventListener('click', () => {
      this.renderChatsHistory();
      menuDrawer?.classList.remove('hidden');
    });
    btnCloseMenu?.addEventListener('click', () => menuDrawer?.classList.add('hidden'));

    const startNewChat = () => {
      if (this.isStreaming) this.abortGeneration();    // ответ допишется в свой диалог (Turn.session)
      this.currentSession = {
        id: 'chat_' + Date.now(),
        title: 'Новый диалог',
        messages: [],
        updatedAt: Date.now()
      };
      const sessions = StorageService.getSessions();
      sessions.unshift(this.currentSession);
      StorageService.saveSessions(sessions);
      StorageService.setCurrentSessionId(this.currentSession.id);
      this.renderCurrentSession();
      menuDrawer?.classList.add('hidden');
    };

    btnNewChat?.addEventListener('click', startNewChat);
    btnDrawerNewChat?.addEventListener('click', startNewChat);

    // Шторка файлов проекта
    const btnFiles = document.getElementById('btn-files');
    const filesDrawer = document.getElementById('files-drawer');
    const btnCloseFiles = document.getElementById('btn-close-files');
    const btnCreateFile = document.getElementById('btn-create-file');
    const btnExportAll = document.getElementById('btn-export-all');

    btnFiles?.addEventListener('click', () => {
      this.renderFilesList();
      filesDrawer?.classList.remove('hidden');
    });
    btnCloseFiles?.addEventListener('click', () => filesDrawer?.classList.add('hidden'));

    btnCreateFile?.addEventListener('click', () => {
      const name = prompt('Введите имя файла (например, main.py или index.html):');
      if (name) {
        const newFile: VirtualFile = {
          id: 'file_' + Date.now(),
          name: name.trim(),
          content: '',
          updatedAt: Date.now()
        };
        StorageService.saveFile(newFile);
        this.renderFilesList();
        this.openFileEditor(newFile);
      }
    });

    btnExportAll?.addEventListener('click', () => {
      const files = StorageService.getFiles();
      if (files.length === 0) {
        alert('Пока нет созданных файлов.');
        return;
      }
      const combined = files.map(f => `=== FILE: ${f.name} ===\n${f.content}\n`).join('\n\n');
      this.saveToDevice('argent_project_export.txt', combined);
    });

    // Окно настроек
    const btnSettings = document.getElementById('btn-settings');
    const settingsModal = document.getElementById('settings-modal');
    const btnCloseSettings = document.getElementById('btn-close-settings');
    const btnSaveSettings = document.getElementById('btn-save-settings');
    const btnPreloadModel = document.getElementById('btn-preload-model');

    btnSettings?.addEventListener('click', () => {
      this.populateSettingsForm();
      settingsModal?.classList.remove('hidden');
    });
    btnCloseSettings?.addEventListener('click', () => settingsModal?.classList.add('hidden'));

    btnSaveSettings?.addEventListener('click', () => {
      this.saveSettingsFromForm();
      settingsModal?.classList.add('hidden');
      this.updateUIState();
    });

    // Обработка выбора локального файла модели с накопителя телефона
    const btnPickLocal = document.getElementById('btn-pick-local-file');
    const btnScanModels = document.getElementById('btn-scan-models');
    const inputLocalFile = document.getElementById('input-local-model-file') as HTMLInputElement;
    const localFileInfo = document.getElementById('local-model-file-info');
    const optCustomLocal = document.getElementById('opt-custom-local') as HTMLOptionElement;
    const selectOfflineModel = document.getElementById('setting-offline-model') as HTMLSelectElement;
    const manualPathInput = document.getElementById('setting-manual-model-path') as HTMLInputElement;

    btnPickLocal?.addEventListener('click', () => {
      inputLocalFile?.click();
    });

    btnScanModels?.addEventListener('click', () => {
      this.scanAndPopulateGgufModels(true);
    });

    // Ввод пути только показывает выбор в списке; применяется он кнопкой «Сохранить».
    // Раньше каждая нажатая буква записывалась в настройки и переключала режим на оффлайн.
    manualPathInput?.addEventListener('input', () => {
      const val = manualPathInput.value.trim();
      if (val && optCustomLocal) {
        optCustomLocal.style.display = 'block';
        optCustomLocal.textContent = `⚡ ${val.split('/').pop() || val}`;
        optCustomLocal.value = val;
        if (selectOfflineModel) selectOfflineModel.value = val;
      }
    });

    // Выбор в списке переносится в поле пути: иначе сохранённый путь к GGUF перекрывал выбор модели
    // WebLLM, и вернуться с файла на модель из списка было нельзя.
    selectOfflineModel?.addEventListener('change', () => {
      if (!manualPathInput) return;
      const val = selectOfflineModel.value;
      manualPathInput.value = isNativeModel(val) && val !== 'custom_local' ? val : '';
    });

    // Автозаполнение известных адресов при смене провайдера (обработчик ставится один раз)
    const providerSelect = document.getElementById('setting-provider') as HTMLSelectElement;
    providerSelect?.addEventListener('change', () => {
      const presets: Record<string, { endpoint: string; model: string }> = {
        deepseek: { endpoint: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
        openrouter: { endpoint: 'https://openrouter.ai/api/v1', model: 'deepseek/deepseek-chat' },
        groq: { endpoint: 'https://api.groq.com/openai/v1', model: 'llama-3.3-70b-versatile' },
        openai: { endpoint: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
        ollama: { endpoint: 'http://192.168.1.50:11434/v1', model: 'qwen2.5:3b' }
      };
      const preset = presets[providerSelect.value];
      const endpointInput = document.getElementById('setting-endpoint') as HTMLInputElement;
      const modelInput = document.getElementById('setting-model') as HTMLInputElement;
      if (preset && endpointInput && modelInput) {
        endpointInput.value = preset.endpoint;
        modelInput.value = preset.model;
      }
    });

    inputLocalFile?.addEventListener('change', async () => {
      if (inputLocalFile.files && inputLocalFile.files[0]) {
        const file = inputLocalFile.files[0];
        const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
        const sizeStr = file.size > 1024 * 1024 * 1024
          ? (file.size / (1024 * 1024 * 1024)).toFixed(2) + ' ГБ'
          : sizeMb + ' МБ';

        this.settings.offline.localFileName = file.name;
        this.settings.offline.localFileSize = sizeStr;
        this.settings.offline.model = file.name;
        this.switchMode('offline');

        if (manualPathInput) {
          manualPathInput.value = file.name;
        }

        // Проверяем, может ли нативный движок найти абсолютный путь прямо сейчас
        if (LlamaNativeService.isAvailable()) {
          try {
            const res = await LlamaNativeService.resolvePath(file.name);
            if (res.found && res.path) {
              this.settings.offline.model = res.path;
              if (manualPathInput) manualPathInput.value = res.path;
            }
          } catch (e) {
            console.warn('resolvePath error:', e);
          }
        }

        StorageService.saveSettings(this.settings);

        if (optCustomLocal) {
          optCustomLocal.style.display = 'block';
          optCustomLocal.textContent = `⚡ ${file.name} (${sizeStr})`;
          optCustomLocal.value = this.settings.offline.model;
        }
        if (selectOfflineModel) {
          selectOfflineModel.value = this.settings.offline.model;
        }
        if (localFileInfo) {
          localFileInfo.style.display = 'block';
          localFileInfo.style.color = '#10b981';
          localFileInfo.innerHTML = `✅ <b>${this.escapeHtml(file.name)}</b> (${sizeStr}) выбран для нативного движка llama.cpp.<br>` +
            `Модель будет загружена напрямую из памяти устройства с ускорением ARM NEON.`;
        }
      }
    });

    btnPreloadModel?.addEventListener('click', async () => {
      const select = document.getElementById('setting-offline-model') as HTMLSelectElement;
      const modelName = select?.value || '';
      const manualInput = document.getElementById('setting-manual-model-path') as HTMLInputElement;
      const manualPath = manualInput?.value?.trim() || '';
      const customIdInput = document.getElementById('setting-custom-offline-id') as HTMLInputElement;
      const customId = customIdInput?.value?.trim() || '';

      const effectiveModel = manualPath || (modelName !== 'custom_local' ? modelName : '') || customId;
      const lowerModel = effectiveModel.toLowerCase();

      const isGgufOrLocal = modelName === 'custom_local' ||
                            lowerModel.endsWith('.gguf') ||
                            lowerModel.endsWith('.bin') ||
                            lowerModel.includes('nanbeige') ||
                            effectiveModel.startsWith('/') ||
                            effectiveModel.startsWith('file://');

      if (isGgufOrLocal) {
        if (!LlamaNativeService.isAvailable()) {
          alert('Файлы формата .gguf запускаются через нативный движок llama.cpp внутри установленного APK приложения.');
          return;
        }

        const queryName = effectiveModel || this.settings.offline.localFileName || this.settings.offline.model || 'Nanbeige4.2-3B-Q4_K_M.gguf';
        try {
          const res = await LlamaNativeService.resolvePath(queryName);
          if (res.found && res.path) {
            const sizeGb = (res.size ? res.size / (1024 * 1024 * 1024) : 0).toFixed(2);
            alert(
              `✅ Локальная модель GGUF найдена на телефоне!\n\n` +
              `Файл: ${res.name || queryName}\n` +
              `Размер: ${sizeGb} ГБ\n` +
              `Путь: ${res.path}\n\n` +
              `Модель запускается нативно на процессоре смартфона с ускорением ARM NEON.\n` +
              `Скачивание из интернета не требуется!`
            );
          } else {
            alert(
              `ℹ️ Файл модели "${queryName}" не найден в стандартных папках (Загрузки, Telegram, Documents).\n\n` +
              `Нажмите кнопку "📁 Выбрать из папки" выше, чтобы указать файл .gguf с накопителя устройства.`
            );
          }
        } catch (err: any) {
          alert(`Ошибка проверки модели: ${err.message}`);
        }
        return;
      }

      // Для веб-моделей (WebLLM)
      const targetWebModel = customId || modelName;
      if (!targetWebModel || targetWebModel === 'custom_local') {
        alert('Выберите модель WebLLM из списка или введите ID модели Hugging Face.');
        return;
      }

      const banner = document.getElementById('model-download-banner');
      const bannerPercent = document.getElementById('banner-percent-text');
      const bannerFill = document.getElementById('banner-progress-fill');
      const bannerText = document.getElementById('banner-status-text');

      banner?.classList.remove('hidden');
      try {
        await AIEngine.preloadOfflineModel(targetWebModel, (p) => {
          if (bannerPercent && bannerFill && bannerText) {
            bannerPercent.textContent = `${p.progress}%`;
            bannerFill.style.width = `${p.progress}%`;
            bannerText.textContent = p.text;
          }
        });
        alert('Веб-модель успешно загружена в кэш устройства и готова к оффлайн работе!');
      } catch (err: any) {
        alert('Ошибка загрузки модели: ' + err.message);
      } finally {
        banner?.classList.add('hidden');
      }
    });

    // Модальное окно просмотра/редактирования файлов
    const fileModal = document.getElementById('file-modal');
    const btnCloseFileModal = document.getElementById('btn-close-file-modal');
    btnCloseFileModal?.addEventListener('click', () => fileModal?.classList.add('hidden'));
  }

  private renderChatsHistory(): void {
    const list = document.getElementById('chats-history-list');
    if (!list) return;

    const sessions = StorageService.getSessions();
    list.innerHTML = '';

    for (const s of sessions) {
      const item = document.createElement('div');
      item.className = `chat-history-item ${s.id === this.currentSession.id ? 'active' : ''}`;
      item.innerHTML = `
        <span class="item-title">${this.escapeHtml(s.title)}</span>
        <button class="icon-btn-delete" style="background:none;border:none;color:#ef4444;cursor:pointer;">✕</button>
      `;

      item.querySelector('.item-title')?.addEventListener('click', () => {
        if (s.id === this.currentSession.id) {
          document.getElementById('menu-drawer')?.classList.add('hidden');
          return;
        }
        if (this.isStreaming) this.abortGeneration();
        this.currentSession = s;
        StorageService.setCurrentSessionId(s.id);
        this.renderCurrentSession();
        document.getElementById('menu-drawer')?.classList.add('hidden');
      });

      item.querySelector('.icon-btn-delete')?.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!confirm(`Удалить диалог «${s.title}»?`)) return;
        if (s.id === this.currentSession.id && this.isStreaming) this.abortGeneration();
        const filtered = StorageService.getSessions().filter(x => x.id !== s.id);
        StorageService.saveSessions(filtered);
        if (s.id === this.currentSession.id) {
          this.currentSession = StorageService.getOrCreateActiveSession();
          this.renderCurrentSession();
        }
        this.renderChatsHistory();
      });

      list.appendChild(item);
    }
  }

  private renderFilesList(): void {
    const list = document.getElementById('files-list');
    if (!list) return;

    const files = StorageService.getFiles();
    list.innerHTML = '';

    if (files.length === 0) {
      list.innerHTML = '<div style="color:var(--text-muted);font-size:0.8rem;text-align:center;padding:20px;">Нет созданных файлов. Сгенерируйте код в чате или создайте файл вручную.</div>';
      return;
    }

    for (const f of files) {
      const item = document.createElement('div');
      item.className = 'file-item';
      item.innerHTML = `
        <span class="item-title">📄 ${this.escapeHtml(f.name)}</span>
        <div style="display:flex;gap:6px;">
          <button class="action-btn-sm btn-open-file">Открыть</button>
          <button class="action-btn-sm btn-delete-file" style="color:#ef4444;">✕</button>
        </div>
      `;

      item.querySelector('.btn-open-file')?.addEventListener('click', () => {
        this.openFileEditor(f);
      });

      item.querySelector('.btn-delete-file')?.addEventListener('click', () => {
        StorageService.deleteFile(f.id);
        this.renderFilesList();
      });

      list.appendChild(item);
    }
  }

  private openFileEditor(file: VirtualFile): void {
    const modal = document.getElementById('file-modal');
    const title = document.getElementById('file-modal-title');
    const content = document.getElementById('file-modal-content') as HTMLTextAreaElement;
    const btnSave = document.getElementById('btn-save-file-content');
    const btnDownload = document.getElementById('btn-download-file');

    if (title) title.textContent = file.name;
    if (content) content.value = file.content;

    btnSave?.replaceWith(btnSave.cloneNode(true));
    const newBtnSave = document.getElementById('btn-save-file-content');
    newBtnSave?.addEventListener('click', () => {
      file.content = content.value;
      file.updatedAt = Date.now();
      StorageService.saveFile(file);
      alert('Файл успешно сохранен!');
      this.renderFilesList();
    });

    btnDownload?.replaceWith(btnDownload.cloneNode(true));
    const newBtnDownload = document.getElementById('btn-download-file');
    newBtnDownload?.addEventListener('click', () => {
      this.saveToDevice(file.name, content.value);
    });

    modal?.classList.remove('hidden');
  }

  private populateSettingsForm(): void {
    const provider = document.getElementById('setting-provider') as HTMLSelectElement;
    const endpoint = document.getElementById('setting-endpoint') as HTMLInputElement;
    const apiKey = document.getElementById('setting-api-key') as HTMLInputElement;
    const model = document.getElementById('setting-model') as HTMLInputElement;
    const offlineModel = document.getElementById('setting-offline-model') as HTMLSelectElement;
    const temp = document.getElementById('setting-temp') as HTMLInputElement;
    const tempVal = document.getElementById('temp-val');

    if (provider) provider.value = this.settings.online.provider;
    if (endpoint) endpoint.value = this.settings.online.endpoint;
    if (apiKey) apiKey.value = this.settings.online.apiKey;
    if (model) model.value = this.settings.online.model;
    const optCustomLocal = document.getElementById('opt-custom-local') as HTMLOptionElement;
    const localFileInfo = document.getElementById('local-model-file-info');
    const customOfflineInput = document.getElementById('setting-custom-offline-id') as HTMLInputElement;
    const native = isNativeModel(this.settings.offline.model, this.settings.offline.localFileName);

    // В поле пути — только путь к файлу GGUF; id модели WebLLM там не место (иначе он «перекрывал» список)
    const manualInput = document.getElementById('setting-manual-model-path') as HTMLInputElement;
    if (manualInput) {
      manualInput.value = native ? this.settings.offline.model : '';
    }

    const threadsSelect = document.getElementById('setting-offline-threads') as HTMLSelectElement;
    if (threadsSelect) {
      threadsSelect.value = String(this.settings.offline.threads || 2);
    }
    const contextSelect = document.getElementById('setting-offline-context') as HTMLSelectElement;
    if (contextSelect) {
      contextSelect.value = String(this.settings.offline.context || 2048);
    }

    if (native && this.settings.offline.localFileName) {
      if (optCustomLocal) {
        optCustomLocal.style.display = 'block';
        optCustomLocal.textContent = `⚡ ${this.settings.offline.localFileName} (${this.settings.offline.localFileSize || 'llama.cpp'})`;
        optCustomLocal.value = this.settings.offline.model;
      }
      if (localFileInfo) {
        localFileInfo.style.display = 'block';
        localFileInfo.style.color = '#10b981';
        localFileInfo.innerHTML = `✅ Выбран файл: <b>${this.escapeHtml(this.settings.offline.localFileName)}</b> ${this.settings.offline.localFileSize ? `(${this.escapeHtml(this.settings.offline.localFileSize)})` : ''} для нативного llama.cpp`;
      }
    }
    if (offlineModel) offlineModel.value = this.settings.offline.model;

    // Проверка разрешений на файлы и поиск моделей на телефоне (только показывает найденное:
    // раньше открытие настроек само подменяло выбранную модель и включало оффлайн-режим)
    const permBanner = document.getElementById('storage-perm-banner');
    const permStatus = document.getElementById('storage-perm-status');
    const btnReqPerm = document.getElementById('btn-request-storage-perm');

    if (LlamaNativeService.isAvailable()) {
      LlamaNativeService.checkStoragePermission().then((granted) => {
        if (permBanner && permStatus && btnReqPerm) {
          permBanner.style.display = 'block';
          if (granted) {
            permBanner.style.background = 'rgba(16, 185, 129, 0.1)';
            permBanner.style.border = '1px solid rgba(16, 185, 129, 0.3)';
            permStatus.innerHTML = '<b style="color:#10b981;">✅ Доступ к памяти телефона включен.</b> Нативный движок готов читать .gguf модели.';
            btnReqPerm.style.display = 'none';
          } else {
            permBanner.style.background = 'rgba(245, 158, 11, 0.1)';
            permBanner.style.border = '1px solid rgba(245, 158, 11, 0.3)';
            permStatus.innerHTML = '<b style="color:#f59e0b;">⚠️ Доступ ко всем файлам не предоставлен.</b> Android блокирует чтение файлов из папки Загрузки.';
            btnReqPerm.style.display = 'inline-block';
            btnReqPerm.onclick = async () => {
              await LlamaNativeService.requestStoragePermission();
            };
          }
        }
      });

      this.scanAndPopulateGgufModels(false);
    }

    if (customOfflineInput) {
      customOfflineInput.value = this.settings.offline.customModelId || '';
    }

    if (temp) {
      temp.value = this.settings.temperature.toString();
      if (tempVal) tempVal.textContent = temp.value;
      temp.oninput = () => { if (tempVal) tempVal.textContent = temp.value; };
    }
  }

  /**
   * Найти GGUF на телефоне и добавить в список. Выбор меняется только по кнопке «Найти» (userTriggered):
   * в форме, а сохраняется он вместе с остальными настройками.
   */
  private async scanAndPopulateGgufModels(userTriggered = false): Promise<void> {
    if (!LlamaNativeService.isAvailable()) return;
    const localFileInfo = document.getElementById('local-model-file-info');
    const selectOfflineModel = document.getElementById('setting-offline-model') as HTMLSelectElement;
    const manualInput = document.getElementById('setting-manual-model-path') as HTMLInputElement;

    if (userTriggered && localFileInfo) {
      localFileInfo.style.display = 'block';
      localFileInfo.style.color = '#3b82f6';
      localFileInfo.textContent = '🔍 Сканирование памяти устройства на наличие .gguf моделей...';
    }

    try {
      const res = await LlamaNativeService.scanForModels();
      if (res.permissionRequired) {
        if (localFileInfo) {
          localFileInfo.style.display = 'block';
          localFileInfo.style.color = '#f59e0b';
          localFileInfo.innerHTML = '⚠️ Требуется включить "Доступ ко всем файлам" в настройках телефона, чтобы найти модели.';
        }
        return;
      }

      if (res.models && res.models.length > 0) {
        for (const m of res.models) {
          let opt = Array.from(selectOfflineModel.options).find(o => o.value === m.path);
          if (!opt) {
            opt = document.createElement('option');
            opt.value = m.path;
            selectOfflineModel.appendChild(opt);
          }
          opt.textContent = `⚡ ${m.name} (${m.sizeFormatted}, Найдено)`;
        }

        if (userTriggered) {
          const first = res.models[0];
          selectOfflineModel.value = first.path;
          if (manualInput) manualInput.value = first.path;
        }

        if (localFileInfo) {
          localFileInfo.style.display = 'block';
          localFileInfo.style.color = '#10b981';
          localFileInfo.innerHTML = `✅ Найдено моделей на телефоне: <b>${res.models.length}</b>` +
            (userTriggered ? ' — выбрана первая, нажмите «Сохранить».' : '.') + '<br>' +
            res.models.map(m => `• <b>${this.escapeHtml(m.name)}</b> (${this.escapeHtml(m.sizeFormatted)})`).join('<br>');
        }
      } else if (userTriggered && localFileInfo) {
        localFileInfo.style.display = 'block';
        localFileInfo.style.color = '#f59e0b';
        localFileInfo.innerHTML = '⚠️ Файлы .gguf не найдены в папках Загрузки (Download), Документы или Telegram. Проверьте, что файл сохранён.';
      }
    } catch (e) {
      console.warn('Ошибка scanAndPopulateGgufModels:', e);
    }
  }

  private saveSettingsFromForm(): void {
    const provider = (document.getElementById('setting-provider') as HTMLSelectElement)?.value as any;
    const endpoint = (document.getElementById('setting-endpoint') as HTMLInputElement)?.value;
    const apiKey = (document.getElementById('setting-api-key') as HTMLInputElement)?.value;
    const model = (document.getElementById('setting-model') as HTMLInputElement)?.value;
    const offlineModel = (document.getElementById('setting-offline-model') as HTMLSelectElement)?.value;
    const customOfflineInput = document.getElementById('setting-custom-offline-id') as HTMLInputElement;
    const manualPath = (document.getElementById('setting-manual-model-path') as HTMLInputElement)?.value?.trim();
    const temp = parseFloat((document.getElementById('setting-temp') as HTMLInputElement)?.value || '0.6');

    this.settings.online.provider = provider || 'deepseek';
    this.settings.online.endpoint = (endpoint || 'https://api.deepseek.com/v1').trim();
    this.settings.online.apiKey = (apiKey || '').trim();
    this.settings.online.model = (model || 'deepseek-chat').trim();

    const threadsSelect = (document.getElementById('setting-offline-threads') as HTMLSelectElement)?.value;
    const threadsVal = threadsSelect ? parseInt(threadsSelect, 10) : 2;
    this.settings.offline.threads = isNaN(threadsVal) ? 2 : threadsVal;
    const contextSelect = (document.getElementById('setting-offline-context') as HTMLSelectElement)?.value;
    const contextVal = contextSelect ? parseInt(contextSelect, 10) : 2048;
    this.settings.offline.context = isNaN(contextVal) ? 2048 : contextVal;

    // Порядок выбора модели: путь к файлу -> свой id WebLLM -> строка списка.
    const customId = customOfflineInput?.value?.trim() || '';
    this.settings.offline.customModelId = customId || undefined;
    if (manualPath) {
      if (manualPath !== this.settings.offline.model) this.settings.offline.localFileSize = undefined;
      this.settings.offline.model = manualPath;
      this.settings.offline.localFileName = manualPath.split('/').pop() || manualPath;
    } else if (customId) {
      this.settings.offline.model = customId;
      this.settings.offline.localFileName = undefined;
      this.settings.offline.localFileSize = undefined;
    } else if (offlineModel && offlineModel !== 'custom_local') {
      this.settings.offline.model = offlineModel;
      if (isNativeModel(offlineModel)) {
        this.settings.offline.localFileName = offlineModel.split('/').pop() || offlineModel;
      } else {
        this.settings.offline.localFileName = undefined;
        this.settings.offline.localFileSize = undefined;
      }
    }

    this.settings.temperature = isNaN(temp) ? 0.6 : temp;
    StorageService.saveSettings(this.settings);
  }

  private async saveToDevice(filename: string, content: string): Promise<void> {
    try {
      const where = await FileExport.saveText(filename, content);
      alert(`Файл «${filename}» сохранён в ${where}.`);
    } catch (err: any) {
      alert(`Не удалось сохранить файл: ${err?.message || err}`);
    }
  }

  private scrollToBottom(): void {
    const chat = document.getElementById('chat-container');
    if (chat) {
      chat.scrollTop = chat.scrollHeight;
    }
  }

  private escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Запуск приложения
document.addEventListener('DOMContentLoaded', () => {
  new ArgentMobileApp();

  // Service Worker — только для PWA из собранной версии. В APK ресурсы и так лежат в приложении, а кэш
  // воркера после обновления APK отдавал бы старую версию; на dev-сервере он ломал бы перезагрузку модулей.
  if (!('serviceWorker' in navigator)) return;
  if (Capacitor.isNativePlatform() || !import.meta.env.PROD) {
    navigator.serviceWorker.getRegistrations()
      .then((regs) => regs.forEach((r) => r.unregister()))
      .catch(() => { /* нечего снимать */ });
    return;
  }
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((reg) => console.log('Argent Mobile Service Worker зарегистрирован:', reg.scope))
      .catch((err) => console.warn('Ошибка Service Worker:', err));
  });
});
