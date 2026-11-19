#!/usr/bin/env python3
"""
Точные правки больших файлов (C#, HLSL, .shader, .uxml, YAML) скриптом.

Зачем: в файлах на тысячи строк правка «найти фрагмент → заменить» надёжнее, чем
редактирование по номерам строк, а проверка «фрагмент найден ровно N раз» не даёт
молча промахнуться или заменить не то вхождение. Кодировка (BOM) и переводы строк
(CRLF/LF) файла сохраняются — Unity и git не видят «лишних» изменений.

Пример (скрипт правки кладут во временную папку, не в проект):
    import sys; sys.path.insert(0, r'<skill>/scripts'); from patcher import Patcher
    P = Patcher(r'Assets/Scripts/Player.cs')
    P.rep('''    public float speed = 5f;''', '''    public float speed = 5f;
    public float sprint = 9f;''')
    P.rep('old()', 'new()', cnt=3)       # ровно три вхождения
    P.save()                             # пишет только если всё совпало

Запуск правок одним скриптом атомарен по смыслу: если любое rep не нашло фрагмент,
исключение выбрасывается ДО save() — файл остаётся нетронутым.
"""
import codecs


class PatchError(Exception):
    pass


class Patcher:
    def __init__(self, path):
        self.path = path
        raw = open(path, 'rb').read()
        self.bom = raw.startswith(codecs.BOM_UTF8)
        text = raw.decode('utf-8-sig')
        self.crlf = '\r\n' in text
        self.text = text.replace('\r\n', '\n')
        self.changes = 0

    def count(self, s):
        return self.text.count(s.replace('\r\n', '\n'))

    def rep(self, old, new, cnt=1):
        old = old.replace('\r\n', '\n'); new = new.replace('\r\n', '\n')
        n = self.text.count(old)
        if n != cnt:
            head = old.strip().splitlines()[0][:100] if old.strip() else repr(old)
            raise PatchError(f'{self.path}: ожидалось {cnt} вхожд., найдено {n}: «{head}»')
        self.text = self.text.replace(old, new)
        self.changes += 1
        return self

    def save(self):
        out = self.text.replace('\n', '\r\n') if self.crlf else self.text
        with open(self.path, 'wb') as f:
            f.write((codecs.BOM_UTF8 if self.bom else b'') + out.encode('utf-8'))
        print(f'patched {self.path} ({self.changes} правок)')
