import assert from 'node:assert';
import { StreamTagParser } from '../src/services/fsm_parser.ts';

// Тест 1: Разбор обычного текста без тегов
{
  let content = '';
  let think = '';
  let rethink = '';

  const parser = new StreamTagParser(
    (d) => content += d,
    (d) => think += d,
    (d) => rethink += d
  );

  parser.feed('Привет! Это простой ответ.');
  parser.flush();

  assert.strictEqual(content, 'Привет! Это простой ответ.');
  assert.strictEqual(think, '');
  assert.strictEqual(rethink, '');
}

// Тест 2: Разделение размышлений <think> и перепроверки <rethink>
{
  let content = '';
  let think = '';
  let rethink = '';

  const parser = new StreamTagParser(
    (d) => content += d,
    (d) => think += d,
    (d) => rethink += d
  );

  parser.feed('<think>Изучаю код задачи</think>');
  parser.feed('<rethink>Проверяю граничные случаи</rethink>');
  parser.feed('Вот готовый результат!');
  parser.flush();

  assert.strictEqual(think, 'Изучаю код задачи');
  assert.strictEqual(rethink, 'Проверяю граничные случаи');
  assert.strictEqual(content, 'Вот готовый результат!');
}

// Тест 3: Разрыв тегов между чанками (<reth + ink>)
{
  let content = '';
  let think = '';
  let rethink = '';

  const parser = new StreamTagParser(
    (d) => content += d,
    (d) => think += d,
    (d) => rethink += d
  );

  parser.feed('Начало. <reth');
  parser.feed('ink>Сложная проверка</rethink> Конец.');
  parser.flush();

  assert.strictEqual(content, 'Начало.  Конец.');
  assert.strictEqual(rethink, 'Сложная проверка');
}

// Тест 4: Безопасность математических знаков <
{
  let content = '';
  const parser = new StreamTagParser(
    (d) => content += d,
    () => {},
    () => {}
  );

  parser.feed('Условие: x < y и a <= b.');
  parser.flush();

  assert.strictEqual(content, 'Условие: x < y и a <= b.');
}

// Тест 5: Прямой ввод reasoning_content (DeepSeek Reasoner)
{
  let content = '';
  let think = '';
  let rethink = '';

  const parser = new StreamTagParser(
    (d) => content += d,
    (d) => think += d,
    (d) => rethink += d
  );

  parser.appendReasoning('Токен 1. ');
  parser.appendReasoning('Токен 2.');
  parser.feed('Основной ответ после рассуждений.');
  parser.flush();

  assert.strictEqual(think, 'Токен 1. Токен 2.');
  assert.strictEqual(parser.fullThinking, 'Токен 1. Токен 2.');
  assert.strictEqual(content, 'Основной ответ после рассуждений.');
  assert.strictEqual(parser.fullContent, 'Основной ответ после рассуждений.');
}

// Тест 6: Регистронезависимые теги (<Think> и <THINK>)
{
  let content = '';
  let think = '';

  const parser = new StreamTagParser(
    (d) => content += d,
    (d) => think += d,
    () => {}
  );

  parser.feed('<Think>Рассуждение с большой буквы</Think>');
  parser.feed('<THINK>Второй блок</THINK>');
  parser.feed('Финальный текст.');
  parser.flush();

  assert.strictEqual(think, 'Рассуждение с большой буквыВторой блок');
  assert.strictEqual(content, 'Финальный текст.');
}

// Тест 7: Обрыв потока на незакрытом теге <think>
{
  let content = '';
  let think = '';

  const parser = new StreamTagParser(
    (d) => content += d,
    (d) => think += d,
    () => {}
  );

  parser.feed('<think>Начало мыслей, но поток оборван лимитом токенов');
  parser.flush();

  assert.strictEqual(think, 'Начало мыслей, но поток оборван лимитом токенов');
  assert.strictEqual(parser.fullThinking, 'Начало мыслей, но поток оборван лимитом токенов');
}

console.log('✓ Все 7 тестов FSM парсера успешно пройдены!');

