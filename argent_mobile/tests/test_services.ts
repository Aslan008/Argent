// Тесты чистой логики: разбор SSE, подгонка истории под окно, слияние настроек, выбор движка.
import assert from 'node:assert';
import { SseParser, SseStreamError, parseEvent } from '../src/services/sse';
import { estimateTokens, fitToBudget, mergeConsecutive, shrinkMiddle } from '../src/services/context';
import { mergeSettings } from '../src/services/storage';
import { isNativeModel, nativeReplyTokens, chatmlPrompt, explainFetchFailure } from '../src/services/ai_engine';
import type { AppSettings } from '../src/types';

// ── SSE ─────────────────────────────────────────────────────────────────────

{
  // строка, разрезанная между чанками, и [DONE] без пробела
  const p = new SseParser();
  const first = p.push('data: {"choices":[{"delta":{"content":"При"}}]}\n\ndata: {"choices":[{"del');
  assert.deepStrictEqual(first, [{ content: 'При' }]);
  const second = p.push('ta":{"content":"вет"}}]}\n\ndata:[DONE]\n\ndata: {"choices":[{"delta":{"content":"лишнее"}}]}\n');
  assert.deepStrictEqual(second, [{ content: 'вет' }]);
  assert.strictEqual(p.done, true);
  assert.deepStrictEqual(p.push('data: {"choices":[{"delta":{"content":"после"}}]}\n'), []);
}

{
  // последнее событие без перевода строки в конце потока не теряется
  const p = new SseParser();
  assert.deepStrictEqual(p.push('data: {"choices":[{"delta":{"content":"хвост"}}]}'), []);
  assert.deepStrictEqual(p.finish(), [{ content: 'хвост' }]);
}

{
  // рассуждения: reasoning_content (DeepSeek) и reasoning (OpenRouter, Ollama)
  assert.deepStrictEqual(parseEvent({ choices: [{ delta: { reasoning_content: 'думаю' } }] }), { reasoning: 'думаю' });
  assert.deepStrictEqual(parseEvent({ choices: [{ delta: { reasoning: 'думаю', content: 'ответ' } }] }),
    { reasoning: 'думаю', content: 'ответ' });
  assert.strictEqual(parseEvent({ choices: [{ delta: { role: 'assistant' } }] }), null);
}

{
  // ошибка посреди потока — исключение с текстом, а не тишина и «0 токенов»
  const p = new SseParser();
  assert.throws(
    () => p.push('data: {"error":{"message":"Provider returned error","metadata":{"provider_name":"Nvidia"}}}\n'),
    (e: any) => e instanceof SseStreamError && /Provider returned error/.test(e.message) && /Nvidia/.test(e.message)
  );
}

{
  // комментарии и служебные строки пропускаются
  const p = new SseParser();
  assert.deepStrictEqual(p.push(': OPENROUTER PROCESSING\n\nevent: ping\ndata: not-json\n\n'), []);
}

// ── окно контекста ──────────────────────────────────────────────────────────

{
  const sys = { role: 'system', content: 'Ты помощник.' };
  const history = Array.from({ length: 40 }, (_, i) => ({
    role: i % 2 === 0 ? 'user' : 'assistant',
    content: `Сообщение номер ${i}: ` + 'текст '.repeat(60),
  }));
  history.push({ role: 'user', content: 'Последний вопрос?' });
  const budget = 1500;
  const fit = fitToBudget(sys, history, budget);
  const used = fit.messages.reduce((n, m) => n + estimateTokens(m.content), 0);
  assert.ok(used <= budget, `использовано ${used} > ${budget}`);
  assert.strictEqual(fit.messages[0], sys);
  assert.strictEqual(fit.messages[fit.messages.length - 1].content, 'Последний вопрос?');
  assert.ok(fit.dropped > 0 && !fit.truncated);
  assert.strictEqual(fit.messages[1].role, 'user', 'история не начинается с ответа ассистента');
  assert.strictEqual(fit.dropped + fit.messages.length - 1, history.length);
}

{
  // всё помещается — ничего не выброшено
  const sys = { role: 'system', content: 's' };
  const history = [{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }, { role: 'user', content: 'c' }];
  const fit = fitToBudget(sys, history, 10_000);
  assert.strictEqual(fit.dropped, 0);
  assert.strictEqual(fit.messages.length, 4);
}

{
  // одно огромное сообщение укорачивается посередине, запрос всё равно уходит
  const sys = { role: 'system', content: 's' };
  const huge = 'НАЧАЛО ' + 'x'.repeat(50_000) + ' КОНЕЦ';
  const fit = fitToBudget(sys, [{ role: 'user', content: huge }], 800);
  assert.ok(fit.truncated);
  const last = fit.messages[1].content;
  assert.ok(last.startsWith('НАЧАЛО') && last.endsWith('КОНЕЦ'));
  assert.ok(estimateTokens(last) <= 800);
  assert.strictEqual(shrinkMiddle('коротко', 100), 'коротко');
}

{
  // вопрос без ответа + следующий вопрос -> одно сообщение пользователя (строгие шаблоны чата)
  const merged = mergeConsecutive([
    { role: 'user', content: 'а' }, { role: 'user', content: 'б' },
    { role: 'assistant', content: 'в' }, { role: 'user', content: 'г' },
  ]);
  assert.deepStrictEqual(merged.map(m => m.role), ['user', 'assistant', 'user']);
  assert.strictEqual(merged[0].content, 'а\n\nб');
}

{
  // кириллица считается дороже латиницы
  assert.ok(estimateTokens('привет'.repeat(100)) > estimateTokens('hello!'.repeat(100)));
}

// ── настройки ───────────────────────────────────────────────────────────────

{
  const defaults: AppSettings = {
    mode: 'online',
    online: { provider: 'deepseek', endpoint: 'https://api.deepseek.com/v1', apiKey: '', model: 'deepseek-chat' },
    offline: { model: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC', threads: 2, context: 2048 },
    temperature: 0.6,
    enableRethink: true,
  };
  // сохранённый раздел offline из старой версии не стирает новые поля по умолчанию
  const merged = mergeSettings(defaults, { offline: { model: '/sdcard/Download/m.gguf' }, temperature: 0.2 });
  assert.strictEqual(merged.offline.model, '/sdcard/Download/m.gguf');
  assert.strictEqual(merged.offline.context, 2048);
  assert.strictEqual(merged.temperature, 0.2);
  assert.strictEqual(merged.online.endpoint, 'https://api.deepseek.com/v1');
  // испорченные значения не принимаются, и defaults не меняются
  const broken = mergeSettings(defaults, { mode: 'turbo', temperature: 'hot', online: 'x' });
  assert.strictEqual(broken.mode, 'online');
  assert.strictEqual(broken.temperature, 0.6);
  assert.strictEqual(broken.online.model, 'deepseek-chat');
  merged.online.model = 'changed';
  assert.strictEqual(defaults.online.model, 'deepseek-chat');
  assert.deepStrictEqual(mergeSettings(defaults, null), { ...defaults, online: { ...defaults.online }, offline: { ...defaults.offline } });
}

// ── движок ──────────────────────────────────────────────────────────────────

{
  assert.strictEqual(isNativeModel('Qwen2.5-1.5B-Instruct-q4f16_1-MLC'), false);
  assert.strictEqual(isNativeModel('Qwen/Qwen2.5-1.5B-Instruct-q4f16_1-MLC'), false);
  assert.strictEqual(isNativeModel('/storage/emulated/0/Download/Nanbeige4.2-3B-Q4_K_M.gguf'), true);
  assert.strictEqual(isNativeModel('Llama-3.2-3B-Instruct-Q4_K_M'), true, 'имя файла без расширения — тоже GGUF');
  assert.strictEqual(isNativeModel('Qwen2.5-1.5B-Instruct-q4f16_1-MLC', 'model.gguf'), true);
  assert.strictEqual(isNativeModel(''), false);
  assert.strictEqual(nativeReplyTokens(2048), 1024);
  assert.strictEqual(nativeReplyTokens(1024), 512);
  assert.strictEqual(nativeReplyTokens(8192), 1024);
  assert.strictEqual(
    chatmlPrompt([{ role: 'system', content: 's' }, { role: 'user', content: 'q' }]),
    '<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n'
  );
  assert.match(explainFetchFailure('http://192.168.1.50:11434/v1/chat/completions', new TypeError('Failed to fetch')),
    /OLLAMA_HOST=0\.0\.0\.0/);
  assert.match(explainFetchFailure('https://api.deepseek.com/v1/chat/completions', new TypeError('Failed to fetch')),
    /интернет/);
}

console.log('  services: все проверки пройдены');
