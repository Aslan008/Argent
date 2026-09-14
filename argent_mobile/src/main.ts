import { marked } from 'marked';
import hljs from 'highlight.js';
import 'highlight.js/styles/atom-one-dark.min.css';
import { StorageService } from './services/storage';
import { AIEngine } from './services/ai_engine';
import { AutopilotService } from './services/autopilot';
import { LlamaNativeService } from './services/llama_native';
import { AppSettings, ChatMessage, ChatSession, VirtualFile } from './types';

// Конфигурация Markdown рендерера с подсветкой синтаксиса
marked.setOptions({
  breaks: true,
  gfm: true
});

class ArgentMobileApp {
  private settings: AppSettings;
  private currentSession: ChatSession;
  private isStreaming: boolean = false;
  private isAutopilotEnabled: boolean = false;
  private activeAssistantBubble: HTMLElement | null = null;
  private activeThinkingContent: HTMLElement | null = null;
  private activeRethinkContent: HTMLElement | null = null;
  private activeTextContent: HTMLElement | null = null;
  private generationStartTime: number = 0;
  private thinkingTimerInterval: any = null;
  private hasReceivedFirstToken: boolean = false;

  constructor() {
    this.settings = StorageService.getSettings();
    this.currentSession = StorageService.getOrCreateActiveSession();

    this.initDOM();
    this.renderCurrentSession();
    this.updateUIState();
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

    // Добавляем сообщение пользователя
    const userMsg: ChatMessage = {
      id: 'msg_' + Date.now(),
      role: 'user',
      content: text,
      timestamp: Date.now()
    };
    this.currentSession.messages.push(userMsg);
    this.renderMessage(userMsg);

    // Добавляем заготовку под ответ ассистента
    const assistantMsgId = 'msg_' + (Date.now() + 1);
    this.prepareAssistantBubble(assistantMsgId);

    // Проверка команд автопилота телефона
    const lowerText = text.toLowerCase().trim();
    const isExplicitNavigation = lowerText === 'домой' || lowerText === 'назад' || lowerText.includes('нажми домой');
    const isExplicitAppLaunch = lowerText.startsWith('открой приложение ') || 
                                lowerText.startsWith('запусти приложение ') || 
                                lowerText.includes('открой приложение на телефоне');
    const hasAutopilotPrefix = lowerText.startsWith('🚀') || lowerText.startsWith('автопилот:') || lowerText.startsWith('автопилот,');

    const shouldHandleWithAutopilot = isExplicitNavigation || isExplicitAppLaunch || (this.isAutopilotEnabled && hasAutopilotPrefix);

    if (shouldHandleWithAutopilot) {
      const isRunning = await AutopilotService.isServiceEnabled();
      if (!isRunning) {
        if (this.activeTextContent) {
          this.activeTextContent.innerHTML = `
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 14px; margin-top: 4px;">
              <div style="font-weight: 600; color: #f59e0b; margin-bottom: 6px;">⚠️ Служба автопилота отключена в системе</div>
              <div style="font-size: 0.82rem; color: #fde68a; margin-bottom: 12px; line-height: 1.4;">
                Чтобы Argent мог самостоятельно открывать приложения и нажимать кнопки на телефоне, включите службу <b>Argent Autopilot Service</b> в специальных возможностях Android.
              </div>
              <button id="btn-open-acc-settings" class="action-btn-sm" style="background: #f59e0b; color: #000; font-weight: 600; border: none; padding: 8px 14px; border-radius: 8px; cursor: pointer;">⚙️ Включить в настройках телефона</button>
            </div>
          `;
          document.getElementById('btn-open-acc-settings')?.addEventListener('click', () => {
            AutopilotService.openSettings();
          });
        }
        this.finalizeAssistantBubble();
        this.setStreamingState(false);
        return;
      }

      // Выполнение действия
      let handled = false;
      if (lowerText.startsWith('открой ') || lowerText.startsWith('запусти ') || lowerText.includes('открой приложение ')) {
        const appMatch = text.match(/(?:открой|запусти)(?:\s+приложение)?\s+([a-zA-Zа-яА-Я0-9\s]+)/i);
        const appName = appMatch ? appMatch[1].replace(/на телефоне/i, '').replace(/приложение/i, '').trim() : '';
        if (appName) {
          handled = true;
          try {
            if (this.activeThinkingContent) {
              this.activeThinkingContent.textContent = `Анализ цели: запуск системного приложения "${appName}" через Android PackageManager.`;
              this.activeThinkingContent.closest('.thought-card')?.classList.remove('hidden');
            }
            await AutopilotService.launchApp(appName);
            if (this.activeTextContent) {
              this.activeTextContent.innerHTML = `🚀 <b>Автопилот:</b> Приложение <b>${this.escapeHtml(appName)}</b> успешно открыто на телефоне!`;
            }
          } catch (err: any) {
            if (this.activeTextContent) {
              this.activeTextContent.innerHTML = `⚠️ <b>Ошибка автопилота:</b> ${this.escapeHtml(err.message)}`;
            }
          }
        }
      } else if (lowerText.includes('нажми домой') || lowerText === 'домой') {
        handled = true;
        await AutopilotService.pressHome();
        if (this.activeTextContent) {
          this.activeTextContent.innerHTML = '🚀 <b>Автопилот:</b> Переход на главный экран телефона выполнен.';
        }
      } else if (lowerText.includes('назад')) {
        handled = true;
        await AutopilotService.pressBack();
        if (this.activeTextContent) {
          this.activeTextContent.innerHTML = '🚀 <b>Автопилот:</b> Нажата кнопка «Назад».';
        }
      }

      if (handled) {
        const assistantMsg: ChatMessage = {
          id: assistantMsgId,
          role: 'assistant',
          content: this.activeTextContent?.textContent || 'Команда выполнена',
          timestamp: Date.now()
        };
        this.currentSession.messages.push(assistantMsg);
        StorageService.updateSessionMessages(this.currentSession.id, this.currentSession.messages);
        this.finalizeAssistantBubble();
        this.setStreamingState(false);
        return;
      }
    }

    const startTime = Date.now();
    this.generationStartTime = startTime;
    this.hasReceivedFirstToken = false;
    this.setStreamingState(true);
    const banner = document.getElementById('model-download-banner');
    const bannerPercent = document.getElementById('banner-percent-text');
    const bannerFill = document.getElementById('banner-progress-fill');
    const bannerText = document.getElementById('banner-status-text');

    try {
      await AIEngine.generate(
        this.currentSession.messages,
        this.settings,
        {
          onPhase: (phase, detail) => {
            const statusEl = this.activeAssistantBubble?.querySelector('#live-thinking-status');
            const detailEl = this.activeAssistantBubble?.querySelector('#live-thinking-detail');
            if (statusEl && phase) statusEl.textContent = phase;
            if (detailEl && detail) detailEl.textContent = detail;
          },
          onContent: (_delta, fullContent) => {
            if (!this.hasReceivedFirstToken) {
              this.hasReceivedFirstToken = true;
              if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
                try { navigator.vibrate(15); } catch {}
              }
            }
            if (this.activeTextContent) {
              this.activeTextContent.classList.remove('is-generating');
              this.activeTextContent.innerHTML = this.renderMarkdown(fullContent);
              this.attachCodeActionListeners(this.activeTextContent);
              this.scrollToBottom();
            }
          },
          onThinking: (_delta, fullThinking) => {
            if (!this.hasReceivedFirstToken) {
              this.hasReceivedFirstToken = true;
              if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
                try { navigator.vibrate(15); } catch {}
              }
            }
            if (this.activeThinkingContent) {
              this.activeThinkingContent.textContent = fullThinking;
              const card = this.activeThinkingContent.closest('.thought-card');
              card?.classList.remove('hidden');
              this.scrollToBottom();
            }
          },
          onRethink: (_delta, fullRethink) => {
            if (this.activeRethinkContent) {
              this.activeRethinkContent.textContent = fullRethink;
              const card = this.activeRethinkContent.closest('.thought-card');
              card?.classList.remove('hidden');
              this.scrollToBottom();
            }
          },
          onDone: (fullContent, fullThinking, fullRethink) => {
            if (this.thinkingTimerInterval) {
              clearInterval(this.thinkingTimerInterval);
              this.thinkingTimerInterval = null;
            }

            if (!fullContent.trim() && !fullThinking.trim()) {
              if (this.activeTextContent) {
                this.activeTextContent.classList.remove('is-generating');
                this.activeTextContent.innerHTML = `
                  <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 12px; color: #fca5a5;">
                    <div style="font-weight: 600; margin-bottom: 4px; color: #ef4444;">⚠️ Ответ не сгенерирован (0 токенов)</div>
                    <div style="font-size: 0.84rem; line-height: 1.4;">
                      Модель завершила вычисления, но не сформировала текст. Проверьте целостность файла модели (.gguf) или повторите отправку.
                    </div>
                  </div>
                `;
              }
              return;
            }

            if (!fullContent.trim() && fullThinking.trim()) {
              if (this.activeTextContent) {
                this.activeTextContent.classList.remove('is-generating');
                this.activeTextContent.innerHTML = `
                  <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 10px; padding: 12px; color: #93c5fd; font-size: 0.84rem; line-height: 1.4;">
                    🧠 <b>Рассуждения сформированы:</b> Модель завершила ход мыслей, но вывод текста был прерван. Вы можете раскрыть блок «Ход мыслей» выше для ознакомления.
                  </div>
                `;
              }
            }

            const durationMs = Date.now() - startTime;
            const durationSec = (durationMs / 1000).toFixed(1);
            const engineName = this.settings.mode === 'offline'
              ? (this.settings.offline.localFileName || this.settings.offline.model)
              : this.settings.online.model;

            const assistantMsg: ChatMessage = {
              id: assistantMsgId,
              role: 'assistant',
              content: fullContent,
              thinking: fullThinking || undefined,
              rethink: fullRethink || undefined,
              timestamp: Date.now(),
              durationMs
            };
            this.currentSession.messages.push(assistantMsg);
            StorageService.updateSessionMessages(
              this.currentSession.id,
              this.currentSession.messages,
              userMsg.content
            );
            this.finalizeAssistantBubble(durationSec, engineName);
            if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
              try { navigator.vibrate([15, 30, 15]); } catch {}
            }
          },
          onError: (err) => {
            if (this.thinkingTimerInterval) {
              clearInterval(this.thinkingTimerInterval);
              this.thinkingTimerInterval = null;
            }
            if (this.activeTextContent) {
              this.activeTextContent.classList.remove('is-generating');
              this.activeTextContent.innerHTML = `
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 12px; color: #fca5a5;">
                  <div style="font-weight: 600; margin-bottom: 4px; color: #ef4444;">⚠️ Ошибка:</div>
                  <div style="font-size: 0.84rem; line-height: 1.4;">${this.escapeHtml(err.message)}</div>
                </div>
              `;
            }
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
      if (this.activeTextContent) {
        this.activeTextContent.classList.remove('is-generating');
        this.activeTextContent.innerHTML = `<div style="color: #ef4444;">⚠️ Сбой: ${this.escapeHtml(err.message)}</div>`;
      }
    } finally {
      this.setStreamingState(false);
      banner?.classList.add('hidden');
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

  private prepareAssistantBubble(msgId: string): void {
    const list = document.getElementById('messages-list');
    if (!list) return;

    const item = document.createElement('div');
    item.className = 'message-item assistant';
    item.id = msgId;

    const isOffline = this.settings.mode === 'offline';
    const engineLabel = isOffline 
      ? (this.settings.offline.model.endsWith('.gguf') || this.settings.offline.model.endsWith('.bin') 
          ? '⚡ llama.cpp ARM NEON' 
          : '⚡ WebLLM GPU')
      : `🌐 ${this.settings.online.provider.toUpperCase()}`;

    const initialPhase = isOffline
      ? 'Запуск нативного C++ движка...'
      : 'Соединение с облачным сервером...';

    const initialDetail = isOffline
      ? 'Вычисление на процессоре Helio G99'
      : 'Ожидание ответа модели...';

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
                  <span class="thinking-label" id="live-thinking-status">${initialPhase}</span>
                  <span class="thinking-timer" id="live-thinking-timer">0.0с</span>
                </div>
                <div class="thinking-detail-row">
                  <span class="thinking-chip-engine">${engineLabel}</span>
                  <div class="thinking-dots"><span></span><span></span><span></span></div>
                  <span class="thinking-detail-text" id="live-thinking-detail">${initialDetail}</span>
                </div>
              </div>
            </div>
            <div class="thinking-shimmer-bar"></div>
          </div>
        </div>
      </div>
    `;

    list.appendChild(item);
    this.activeAssistantBubble = item;
    this.activeThinkingContent = item.querySelector('.thought-content-text');
    this.activeRethinkContent = item.querySelector('.rethink-content-text');
    this.activeTextContent = item.querySelector('.assistant-content');

    this.attachThoughtCollapsers(item);
    this.scrollToBottom();

    // Запуск живого таймера размышления (100 мс)
    if (this.thinkingTimerInterval) clearInterval(this.thinkingTimerInterval);
    const timerEl = item.querySelector('#live-thinking-timer');
    this.thinkingTimerInterval = setInterval(() => {
      if (timerEl) {
        const elapsed = ((Date.now() - this.generationStartTime) / 1000).toFixed(1);
        timerEl.textContent = `${elapsed}с`;
      }
    }, 100);

    // Тактильный виброотклик при старте генерации
    if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
      try { navigator.vibrate(20); } catch {}
    }
  }

  private finalizeAssistantBubble(durationSec?: string, engineName?: string): void {
    if (this.thinkingTimerInterval) {
      clearInterval(this.thinkingTimerInterval);
      this.thinkingTimerInterval = null;
    }

    if (this.activeAssistantBubble) {
      this.activeTextContent?.classList.remove('is-generating');

      // Авто-сворачивание хода мыслей после завершения генерации
      const thinkingCard = this.activeAssistantBubble.querySelector('.thought-card.thinking');
      thinkingCard?.classList.add('collapsed');
      const rethinkCard = this.activeAssistantBubble.querySelector('.thought-card.rethink');
      rethinkCard?.classList.add('collapsed');

      // Убираем пульсирующие точки
      const indicators = this.activeAssistantBubble.querySelectorAll('.thought-indicator');
      indicators.forEach(el => el.remove());

      // Добавляем красивый мета-бейдж времени генерации и модели
      if (durationSec && this.activeTextContent && !this.activeAssistantBubble.querySelector('.message-meta-badge')) {
        const metaBadge = document.createElement('div');
        metaBadge.className = 'message-meta-badge';
        const isOffline = this.settings.mode === 'offline';
        metaBadge.innerHTML = `
          <span class="meta-icon">${isOffline ? '⚡' : '🌐'}</span>
          <span>${durationSec}с</span>
          <span>•</span>
          <span>${this.escapeHtml(engineName || '')}</span>
        `;
        this.activeAssistantBubble.querySelector('.assistant-body')?.appendChild(metaBadge);
      }
    }
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

          ${durationSec ? `
            <div class="message-meta-badge">
              <span class="meta-icon">⚡</span>
              <span>${durationSec}с</span>
            </div>
          ` : ''}
        </div>
      `;

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

  private renderMarkdown(raw: string): string {
    const rawHtml = marked.parse(raw) as string;
    const cleanHtml = this.sanitizeMarkdownHtml(rawHtml);
    const temp = document.createElement('div');
    temp.innerHTML = cleanHtml;

    // Подсветка и обёртка блоков кода
    temp.querySelectorAll('pre code').forEach((codeBlock) => {
      const codeElement = codeBlock as HTMLElement;
      hljs.highlightElement(codeElement);

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
      btn.addEventListener('click', (e) => {
        const target = e.currentTarget as HTMLElement;
        const codeWrapper = target.closest('.code-block-wrapper');
        const code = codeWrapper?.querySelector('pre code')?.textContent || '';
        navigator.clipboard.writeText(code);
        target.textContent = '✓ Скопировано!';
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
      StorageService.downloadFileToDevice('argent_project_export.txt', combined);
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

    manualPathInput?.addEventListener('input', () => {
      const val = manualPathInput.value.trim();
      if (val) {
        this.settings.offline.model = val;
        this.settings.offline.localFileName = val.split('/').pop() || val;
        this.switchMode('offline');
        if (optCustomLocal) {
          optCustomLocal.style.display = 'block';
          optCustomLocal.textContent = `⚡ ${this.settings.offline.localFileName}`;
          optCustomLocal.value = val;
        }
        if (selectOfflineModel) selectOfflineModel.value = val;
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
      const modelName = select.value;
      const banner = document.getElementById('model-download-banner');
      const bannerPercent = document.getElementById('banner-percent-text');
      const bannerFill = document.getElementById('banner-progress-fill');
      const bannerText = document.getElementById('banner-status-text');

      banner?.classList.remove('hidden');
      try {
        await AIEngine.preloadOfflineModel(modelName, (p) => {
          if (bannerPercent && bannerFill && bannerText) {
            bannerPercent.textContent = `${p.progress}%`;
            bannerFill.style.width = `${p.progress}%`;
            bannerText.textContent = p.text;
          }
        });
        alert('Модель успешно загружена в кэш устройства и готова к оффлайн работе!');
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
        this.currentSession = s;
        StorageService.setCurrentSessionId(s.id);
        this.renderCurrentSession();
        document.getElementById('menu-drawer')?.classList.add('hidden');
      });

      item.querySelector('.icon-btn-delete')?.addEventListener('click', (e) => {
        e.stopPropagation();
        const filtered = sessions.filter(x => x.id !== s.id);
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
      StorageService.downloadFileToDevice(file.name, content.value);
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

    const manualInput = document.getElementById('setting-manual-model-path') as HTMLInputElement;
    if (manualInput) {
      manualInput.value = this.settings.offline.localFileName || this.settings.offline.model || '';
    }

    if (this.settings.offline.localFileName) {
      if (optCustomLocal) {
        optCustomLocal.style.display = 'block';
        optCustomLocal.textContent = `⚡ ${this.settings.offline.localFileName} (${this.settings.offline.localFileSize || 'llama.cpp'})`;
        optCustomLocal.value = this.settings.offline.model;
      }
      if (offlineModel) offlineModel.value = this.settings.offline.model;
      if (localFileInfo) {
        localFileInfo.style.display = 'block';
        localFileInfo.style.color = '#10b981';
        localFileInfo.innerHTML = `✅ Выбран файл: <b>${this.escapeHtml(this.settings.offline.localFileName)}</b> (${this.settings.offline.localFileSize || ''}) для нативного llama.cpp`;
      }
    } else {
      if (offlineModel) offlineModel.value = this.settings.offline.model;
    }

    // Проверка разрешений на файлы и автосканирование моделей
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

    // Автозаполнение известных URL при смене провайдера
    provider?.addEventListener('change', () => {
      const presets: Record<string, { endpoint: string; model: string }> = {
        deepseek: { endpoint: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
        openrouter: { endpoint: 'https://openrouter.ai/api/v1', model: 'deepseek/deepseek-chat' },
        groq: { endpoint: 'https://api.groq.com/openai/v1', model: 'llama-3.3-70b-versatile' },
        openai: { endpoint: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
        ollama: { endpoint: 'http://192.168.1.50:11434/v1', model: 'nanbeige4.2:3b' }
      };
      if (presets[provider.value]) {
        endpoint.value = presets[provider.value].endpoint;
        model.value = presets[provider.value].model;
      }
    });
  }

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
          let opt = selectOfflineModel.querySelector(`option[value="${m.path}"]`) as HTMLOptionElement;
          if (!opt) {
            opt = document.createElement('option');
            opt.value = m.path;
            selectOfflineModel.appendChild(opt);
          }
          opt.textContent = `⚡ ${m.name} (${m.sizeFormatted}, Найдено)`;
        }

        if (userTriggered || !this.settings.offline.model.endsWith('.gguf')) {
          const first = res.models[0];
          this.settings.offline.localFileName = first.name;
          this.settings.offline.localFileSize = first.sizeFormatted;
          this.settings.offline.model = first.path;
          selectOfflineModel.value = first.path;
          if (manualInput) manualInput.value = first.path;
          this.switchMode('offline');
          StorageService.saveSettings(this.settings);
        }

        if (localFileInfo) {
          localFileInfo.style.display = 'block';
          localFileInfo.style.color = '#10b981';
          localFileInfo.innerHTML = `✅ Найдено моделей на телефоне: <b>${res.models.length}</b>.<br>` +
            res.models.map(m => `• <b>${this.escapeHtml(m.name)}</b> (${m.sizeFormatted})`).join('<br>');
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
    this.settings.online.endpoint = endpoint || 'https://api.deepseek.com/v1';
    this.settings.online.apiKey = apiKey || '';
    this.settings.online.model = model || 'deepseek-chat';

    if (manualPath && (manualPath.endsWith('.gguf') || manualPath.endsWith('.bin'))) {
      this.settings.offline.model = manualPath;
      this.settings.offline.localFileName = manualPath.split('/').pop() || manualPath;
    } else {
      const customId = customOfflineInput?.value?.trim();
      if (customId) {
        this.settings.offline.customModelId = customId;
        if (!this.settings.offline.localFileName) {
          this.settings.offline.model = customId;
        }
      } else if (offlineModel) {
        this.settings.offline.model = offlineModel;
        if (offlineModel.endsWith('.gguf') || offlineModel.endsWith('.bin')) {
          this.settings.offline.localFileName = offlineModel.split('/').pop() || offlineModel;
        }
      }
    }

    this.settings.temperature = temp;
    StorageService.saveSettings(this.settings);
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

  // Регистрация Service Worker для полной автономности приложения без ПК
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/sw.js')
        .then((reg) => console.log('Argent Mobile Service Worker зарегистрирован:', reg.scope))
        .catch((err) => console.warn('Ошибка Service Worker:', err));
    });
  }
});
