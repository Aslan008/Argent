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

console.log('✓ Все 4 теста FSM парсера успешно пройдены!');
