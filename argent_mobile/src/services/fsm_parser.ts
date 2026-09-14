/**
 * Детерминированный конечный автомат (FSM) для разбора потока токенов
 * с разделением на основной ответ, размышления (<think>) и перепроверку (<rethink>).
 * Защищен от разрыва токенов между чанками, ложных срабатываний математических символов '<'
 * и поддерживает прямой ввод reasoning_content (DeepSeek Reasoner).
 */

export const STATE_CONTENT = 0;
export const STATE_THINKING = 1;
export const STATE_RETHINK = 2;

export class StreamTagParser {
  private state: number = STATE_CONTENT;
  private buffer: string = '';
  public fullContent: string = '';
  public fullThinking: string = '';
  public fullRethink: string = '';

  private readonly TAG_OPEN_THINK = '<think>';
  private readonly TAG_CLOSE_THINK = '</think>';
  private readonly TAG_OPEN_RETHINK = '<rethink>';
  private readonly TAG_CLOSE_RETHINK = '</rethink>';
  private readonly MAX_LOOKAHEAD = 12;

  private onContent: (delta: string) => void;
  private onThinking: (delta: string) => void;
  private onRethink: (delta: string) => void;

  constructor(
    onContent: (delta: string) => void,
    onThinking: (delta: string) => void,
    onRethink: (delta: string) => void
  ) {
    this.onContent = onContent;
    this.onThinking = onThinking;
    this.onRethink = onRethink;
  }

  /**
   * Прямое добавление токенов рассуждений (для DeepSeek Reasoner reasoning_content)
   */
  public appendReasoning(delta: string): void {
    if (!delta) return;
    this.fullThinking += delta;
    this.onThinking(delta);
  }

  public feed(chunk: string): void {
    this.buffer += chunk;

    while (this.buffer.length > 0) {
      const lower = this.buffer.toLowerCase();

      if (this.state === STATE_CONTENT) {
        const idxThink = lower.indexOf(this.TAG_OPEN_THINK);
        const idxRethink = lower.indexOf(this.TAG_OPEN_RETHINK);

        let targetIdx = -1;
        let isRethink = false;

        if (idxThink !== -1 && idxRethink !== -1) {
          if (idxThink < idxRethink) {
            targetIdx = idxThink;
          } else {
            targetIdx = idxRethink;
            isRethink = true;
          }
        } else if (idxThink !== -1) {
          targetIdx = idxThink;
        } else if (idxRethink !== -1) {
          targetIdx = idxRethink;
          isRethink = true;
        }

        if (targetIdx !== -1) {
          // Выводим весь контент до начала тега
          if (targetIdx > 0) {
            const text = this.buffer.slice(0, targetIdx);
            this.fullContent += text;
            this.onContent(text);
          }
          const tagLen = isRethink ? this.TAG_OPEN_RETHINK.length : this.TAG_OPEN_THINK.length;
          this.buffer = this.buffer.slice(targetIdx + tagLen);
          this.state = isRethink ? STATE_RETHINK : STATE_THINKING;
          continue;
        }

        // Проверяем возможность разрыва тега на границе буфера (<th... или <reth...)
        const lastLt = lower.lastIndexOf('<');
        if (lastLt !== -1 && lower.length - lastLt < this.MAX_LOOKAHEAD) {
          const prefix = lower.slice(lastLt);
          if (
            this.TAG_OPEN_THINK.startsWith(prefix) ||
            this.TAG_OPEN_RETHINK.startsWith(prefix)
          ) {
            // Сбрасываем всё до '<'
            if (lastLt > 0) {
              const text = this.buffer.slice(0, lastLt);
              this.fullContent += text;
              this.onContent(text);
              this.buffer = this.buffer.slice(lastLt);
            }
            break;
          }
        }

        // Обычный текст без тегов
        this.fullContent += this.buffer;
        this.onContent(this.buffer);
        this.buffer = '';

      } else if (this.state === STATE_THINKING) {
        const closeIdx = lower.indexOf(this.TAG_CLOSE_THINK);
        if (closeIdx !== -1) {
          if (closeIdx > 0) {
            const thinkText = this.buffer.slice(0, closeIdx);
            this.fullThinking += thinkText;
            this.onThinking(thinkText);
          }
          this.buffer = this.buffer.slice(closeIdx + this.TAG_CLOSE_THINK.length);
          this.state = STATE_CONTENT;
          continue;
        }

        const lastLt = lower.lastIndexOf('<');
        if (lastLt !== -1 && lower.length - lastLt < this.MAX_LOOKAHEAD) {
          const prefix = lower.slice(lastLt);
          if (this.TAG_CLOSE_THINK.startsWith(prefix)) {
            if (lastLt > 0) {
              const thinkText = this.buffer.slice(0, lastLt);
              this.fullThinking += thinkText;
              this.onThinking(thinkText);
              this.buffer = this.buffer.slice(lastLt);
            }
            break;
          }
        }

        this.fullThinking += this.buffer;
        this.onThinking(this.buffer);
        this.buffer = '';

      } else if (this.state === STATE_RETHINK) {
        const closeIdx = lower.indexOf(this.TAG_CLOSE_RETHINK);
        if (closeIdx !== -1) {
          if (closeIdx > 0) {
            const rethinkText = this.buffer.slice(0, closeIdx);
            this.fullRethink += rethinkText;
            this.onRethink(rethinkText);
          }
          this.buffer = this.buffer.slice(closeIdx + this.TAG_CLOSE_RETHINK.length);
          this.state = STATE_CONTENT;
          continue;
        }

        const lastLt = lower.lastIndexOf('<');
        if (lastLt !== -1 && lower.length - lastLt < this.MAX_LOOKAHEAD) {
          const prefix = lower.slice(lastLt);
          if (this.TAG_CLOSE_RETHINK.startsWith(prefix)) {
            if (lastLt > 0) {
              const rethinkText = this.buffer.slice(0, lastLt);
              this.fullRethink += rethinkText;
              this.onRethink(rethinkText);
              this.buffer = this.buffer.slice(lastLt);
            }
            break;
          }
        }

        this.fullRethink += this.buffer;
        this.onRethink(this.buffer);
        this.buffer = '';
      }
    }
  }

  public flush(): void {
    if (this.buffer.length > 0) {
      if (this.state === STATE_THINKING) {
        this.fullThinking += this.buffer;
        this.onThinking(this.buffer);
      } else if (this.state === STATE_RETHINK) {
        this.fullRethink += this.buffer;
        this.onRethink(this.buffer);
      } else {
        this.fullContent += this.buffer;
        this.onContent(this.buffer);
      }
      this.buffer = '';
    }
    this.state = STATE_CONTENT;
  }
}
