#!/usr/bin/env python3
"""
Запуск Unity в batchmode (без окна) и разбор лога: компиляция C#, импорт шейдеров,
выполнение своего editor-метода (тесты, бенчмарки, снимки, правка сцен через API).

  python unity_batch.py                                   # полная перекомпиляция + импорт
  python unity_batch.py --method MyTest_TEMP.Run          # -executeMethod
  python unity_batch.py --method X.Run --grep MYTAG       # показать строки лога с маркером
  python unity_batch.py --graphics --method Bench.Run     # с видеокартой (рендер, compute, снимки)
  python unity_batch.py --wait 900                        # ждать, пока проект освободится
  python unity_batch.py --method X.Run -- -buildTarget Android   # доп. аргументы Unity после --

Ключевое:
  • Проект должен быть ЗАКРЫТ в редакторе (Temp/UnityLockfile). Скрипт проверяет блокировку
    и либо ждёт (--wait), либо выходит с кодом 3 — не запускайте второй Unity на тот же проект.
  • По умолчанию -nographics: быстрее, но без GPU (compute-ядра «not found», рендер пустой).
    Для рендера, compute, снимков экрана и замеров FPS — --graphics.
  • Unity.exe ищется: --unity, переменная UNITY_EXE, Unity Hub (версия из ProjectVersion.txt).

Коды выхода: 0 — ок; 1 — ошибки компиляции/исключения/код Unity ≠ 0; 2 — Unity не найден;
3 — проект открыт другим экземпляром; 4 — таймаут.
"""
import argparse, json, os, re, subprocess, sys, tempfile, time


def find_project(start):
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, 'ProjectSettings')) and os.path.isdir(os.path.join(d, 'Assets')):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            sys.exit('Не найден корень Unity-проекта (ProjectSettings/ + Assets/). Укажите --project.')
        d = nd


def editor_version(root):
    with open(os.path.join(root, 'ProjectSettings', 'ProjectVersion.txt'), encoding='utf-8') as f:
        m = re.search(r'm_EditorVersion:\s*(\S+)', f.read())
    return m.group(1) if m else None


def find_unity(root, explicit):
    if explicit:
        return explicit
    if os.environ.get('UNITY_EXE'):
        return os.environ['UNITY_EXE']
    ver = editor_version(root)
    cands = []
    bases = []
    if sys.platform == 'win32':
        bases += [r'C:\Program Files\Unity\Hub\Editor', r'C:\Program Files\Unity\Editor']
        sec = os.path.join(os.environ.get('APPDATA', ''), 'UnityHub', 'secondaryInstallPath.json')
        if os.path.isfile(sec):
            try:
                p = json.load(open(sec, encoding='utf-8'))
                if p:
                    bases.insert(0, p)
            except Exception:
                pass
        exe = os.path.join('Editor', 'Unity.exe')
    elif sys.platform == 'darwin':
        bases += ['/Applications/Unity/Hub/Editor']
        exe = os.path.join('Unity.app', 'Contents', 'MacOS', 'Unity')
    else:
        bases += [os.path.expanduser('~/Unity/Hub/Editor')]
        exe = os.path.join('Editor', 'Unity')
    for b in bases:
        if ver:
            cands.append(os.path.join(b, ver, exe))
    for c in cands:
        if os.path.isfile(c):
            return c
    sys.exit(f'Unity {ver} не найден (искал: {cands}). Укажите --unity или UNITY_EXE.')


def project_locked(root):
    lock = os.path.join(root, 'Temp', 'UnityLockfile')
    if not os.path.exists(lock):
        return False
    if sys.platform == 'win32':
        try:
            with open(lock, 'a'):
                return False   # файл остался от упавшего Unity, но не занят
        except PermissionError:
            return True
    try:
        out = subprocess.run(['pgrep', '-f', root], capture_output=True, text=True).stdout
        return bool(out.strip())
    except FileNotFoundError:
        return True


PATTERNS = [
    ('compile', re.compile(r'error CS\d+')),
    ('compile', re.compile(r'Scripts have compiler errors|Compilation failed')),
    ('shader', re.compile(r'Shader error in|shader compiler.*error', re.I)),
    ('exception', re.compile(r'^\s*\w*(Exception|Error): ', re.M)),
    ('abort', re.compile(r'Aborting batchmode due to failure|executeMethod method .* threw exception|Couldn.t find method')),
    ('license', re.compile(r'No valid Unity Editor license|License is not active|Entitlement')),
    ('lock', re.compile(r'another Unity instance is running|Multiple Unity instances cannot open the same project')),
]
NOISE = re.compile(r'Exception occured while accepting client connection|Microsoft\.Extensions\.Hosting|'
                   r'Curl error|UnityConnect|Licensing::|\[Package Manager\]|Unable to connect|'
                   r'Error while (getting|sending)|TrimDiskCacheJob')


def summarize(log, grep):
    try:
        with open(log, encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return {}, []
    found, marks = {}, []
    for i, l in enumerate(lines):
        if grep and any(g in l for g in grep):
            marks.append(l.strip())
        if NOISE.search(l):
            continue
        for kind, rx in PATTERNS:
            if rx.search(l):
                s = l.strip()
                if kind == 'exception' and i + 1 < len(lines):
                    s += '  ⟶ ' + lines[i + 1].strip()[:200]
                lst = found.setdefault(kind, [])
                if s not in lst and len(lst) < 40:
                    lst.append(s)
                break
    return found, marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project')
    ap.add_argument('--unity')
    ap.add_argument('--method', help='Класс.Метод для -executeMethod (static, в папке Editor)')
    ap.add_argument('--graphics', action='store_true', help='с видеокартой (без -nographics)')
    ap.add_argument('--no-quit', action='store_true', help='не передавать -quit (метод сам вызывает EditorApplication.Exit)')
    ap.add_argument('--log', help='путь лога (по умолчанию — временный)')
    ap.add_argument('--grep', nargs='*', default=[], help='маркеры: показать строки лога, где они есть')
    ap.add_argument('--wait', type=int, default=0, help='сколько секунд ждать освобождения проекта')
    ap.add_argument('--timeout', type=int, default=1800)
    ap.add_argument('rest', nargs=argparse.REMAINDER, help='после -- : доп. аргументы Unity')
    a = ap.parse_args()

    root = find_project(a.project or os.getcwd())
    unity = find_unity(root, a.unity)
    if not os.path.isfile(unity):
        print(f'Нет файла {unity}'); sys.exit(2)

    t0 = time.time()
    while project_locked(root):
        if time.time() - t0 >= a.wait:
            print('[unity_batch] Проект открыт в редакторе или другом batchmode (Temp/UnityLockfile занят). '
                  'Закройте Unity или используйте --wait.')
            sys.exit(3)
        time.sleep(10)

    log = a.log or os.path.join(tempfile.gettempdir(), f'unity_batch_{int(time.time())}.log')
    cmd = [unity, '-batchmode', '-projectPath', root, '-logFile', log]
    if not a.graphics:
        cmd.append('-nographics')
    if not a.no_quit:
        cmd.append('-quit')
    if a.method:
        cmd += ['-executeMethod', a.method]
    rest = a.rest[1:] if a.rest[:1] == ['--'] else a.rest
    cmd += rest
    print('[unity_batch]', ' '.join(f'"{c}"' if ' ' in c else c for c in cmd), flush=True)
    t1 = time.time()
    try:
        code = subprocess.run(cmd, timeout=a.timeout).returncode
    except subprocess.TimeoutExpired:
        print(f'[unity_batch] Таймаут {a.timeout} с. Лог: {log}'); sys.exit(4)
    dt = time.time() - t1

    found, marks = summarize(log, a.grep)
    for m in marks:
        print('MARK ', m)
    for kind, items in found.items():
        for s in items:
            print(f'{kind.upper():9}', s)
    bad = code != 0 or any(k in found for k in ('compile', 'shader', 'abort', 'lock', 'license'))
    print(f'[unity_batch] код Unity {code}, {dt:.0f} с, лог: {log}')
    print('[unity_batch] ' + ('ЕСТЬ ПРОБЛЕМЫ' if bad else 'OK (исключения выше, если есть, — проверьте, ваши ли они)'))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
