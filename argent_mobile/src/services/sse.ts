/**
 * Разбор потока Server-Sent Events от OpenAI-совместимых API (DeepSeek, OpenRouter, Groq, Ollama, LM Studio...).
 *
 * Вынесено из ai_engine, чтобы проверяться тестами без сети. Что здесь учтено:
 * - строка может прийти разрезанной между чанками: незаконченный хвост остаётся в буфере до следующего чанка
 *   и дочитывается в finish(), когда поток закрылся без завершающего перевода строки;
 * - `data: [DONE]` и `data:[DONE]` (без пробела) — оба означают конец;
 * - ошибка посреди потока приходит обычным JSON-событием `{"error": {...}}` (так делает OpenRouter, когда
 *   падает модель у апстрима) — раньше она молча пропускалась и пользователь видел «0 токенов»;
 * - рассуждения приходят в `reasoning_content` (DeepSeek) или в `reasoning` (OpenRouter, Ollama).
 */

export interface SseDelta {
  content?: string;
  reasoning?: string;
}

export class SseStreamError extends Error {}

export class SseParser {
  private buffer = '';
  public done = false;

  /** Добавить кусок текста потока; возвращает разобранные дельты. Бросает SseStreamError на событии-ошибке. */
  public push(text: string): SseDelta[] {
    if (this.done) return [];
    this.buffer += text;
    const lines = this.buffer.split('\n');
    this.buffer = lines.pop() ?? '';
    return this.parseLines(lines);
  }

  /** Поток закрыт: разобрать то, что осталось в буфере без перевода строки. */
  public finish(): SseDelta[] {
    if (this.done || !this.buffer.trim()) return [];
    const rest = this.buffer;
    this.buffer = '';
    return this.parseLines([rest]);
  }

  private parseLines(lines: string[]): SseDelta[] {
    const out: SseDelta[] = [];
    for (const line of lines) {
      if (this.done) break;
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;           // комментарии ": ping", пустые строки, event:/id:
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        this.done = true;
        break;
      }
      let json: any;
      try {
        json = JSON.parse(payload);
      } catch {
        continue;                                           // не JSON — не наше событие
      }
      const delta = parseEvent(json);
      if (delta) out.push(delta);
    }
    return out;
  }
}

/** Одна JSON-полезная нагрузка события -> дельта (или null). Бросает SseStreamError, если это ошибка. */
export function parseEvent(json: any): SseDelta | null {
  if (json && json.error) {
    const err = json.error;
    const message = typeof err === 'string' ? err : (err.message || JSON.stringify(err));
    const upstream = err?.metadata?.provider_name ? ` (апстрим: ${err.metadata.provider_name})` : '';
    throw new SseStreamError(`Сервер прервал ответ: ${message}${upstream}`);
  }
  const choice = json?.choices?.[0];
  const delta = choice?.delta ?? choice?.message;           // некоторые серверы шлют message вместо delta
  if (!delta) return null;
  const reasoning = delta.reasoning_content ?? delta.reasoning;
  const result: SseDelta = {};
  if (typeof reasoning === 'string' && reasoning) result.reasoning = reasoning;
  if (typeof delta.content === 'string' && delta.content) result.content = delta.content;
  return result.content || result.reasoning ? result : null;
}
