/**
 * Сколько истории диалога помещается в окно модели.
 *
 * Вся история уходила в каждый запрос: длинный диалог рано или поздно переполнял окно — облачный сервер
 * отвечал ошибкой, а нативный llama.cpp на телефоне (окно 2048 токенов) падал на первом же шаге декодирования
 * с сообщением «повреждён файл модели». Здесь история подрезается со старого конца: системный промпт и
 * последнее сообщение пользователя остаются всегда, старые реплики уходят первыми.
 */

export interface PromptMessage {
  role: string;
  content: string;
}

/** Грубая, но осторожная оценка токенов: кириллица у токенизаторов Qwen/Llama дороже латиницы. */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  let cyrillic = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c >= 0x0400 && c <= 0x04ff) cyrillic++;
  }
  const other = text.length - cyrillic;
  // + накладные расходы шаблона чата на сообщение (метки роли, разделители)
  return Math.ceil(cyrillic / 2.2 + other / 3.5) + 4;
}

export interface FitResult {
  messages: PromptMessage[];
  dropped: number;      // сколько старых сообщений не вошло
  truncated: boolean;   // последнее сообщение пришлось укоротить
}

/**
 * system + как можно больше последних сообщений истории в пределах `budgetTokens`.
 * Последнее сообщение (текущий запрос) остаётся всегда; если оно одно больше бюджета — у него вырезается
 * середина, чтобы запрос всё же ушёл, а не упал.
 */
export function fitToBudget(system: PromptMessage, history: PromptMessage[], budgetTokens: number): FitResult {
  const room = Math.max(0, budgetTokens - estimateTokens(system.content));
  if (history.length === 0) return { messages: [system], dropped: 0, truncated: false };

  const last = history[history.length - 1];
  let truncated = false;
  let lastMsg = last;
  if (estimateTokens(last.content) > room) {
    lastMsg = { role: last.role, content: shrinkMiddle(last.content, room) };
    truncated = true;
  }

  const kept: PromptMessage[] = [lastMsg];
  let used = estimateTokens(lastMsg.content);
  let i = history.length - 2;
  for (; i >= 0; i--) {
    const cost = estimateTokens(history[i].content);
    if (used + cost > room) break;
    kept.unshift(history[i]);
    used += cost;
  }
  // История не должна начинаться с ответа ассистента без вопроса к нему
  while (kept.length > 1 && kept[0].role === 'assistant') {
    kept.shift();
    i++;
  }
  return { messages: [system, ...kept], dropped: i + 1, truncated };
}

/**
 * Подряд идущие сообщения одной роли — одним сообщением. Так бывает, когда вопрос остался без ответа
 * (ошибка, «Стоп» до первого токена), а шаблоны чата ряда моделей (Gemma, Mistral) требуют строгого
 * чередования user/assistant и отказываются собирать такой промпт.
 */
export function mergeConsecutive(messages: PromptMessage[]): PromptMessage[] {
  const out: PromptMessage[] = [];
  for (const m of messages) {
    const prev = out[out.length - 1];
    if (prev && prev.role === m.role) {
      out[out.length - 1] = { role: m.role, content: `${prev.content}\n\n${m.content}` };
    } else {
      out.push({ role: m.role, content: m.content });
    }
  }
  return out;
}

/** Текст, укороченный до ~`tokens` токенов: начало и конец остаются, середина заменяется пометкой. */
export function shrinkMiddle(text: string, tokens: number): string {
  if (estimateTokens(text) <= tokens) return text;
  const marker = '\n…[середина сообщения не поместилась в окно модели]…\n';
  // символов на токен с запасом под кириллицу
  const keepChars = Math.max(0, Math.floor((tokens - estimateTokens(marker)) * 2.2));
  if (keepChars <= 0) return marker.trim();
  const head = Math.ceil(keepChars * 0.6);
  const tail = keepChars - head;
  return text.slice(0, head) + marker + (tail > 0 ? text.slice(-tail) : '');
}
