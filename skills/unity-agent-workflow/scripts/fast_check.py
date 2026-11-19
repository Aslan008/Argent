#!/usr/bin/env python3
"""
Быстрая проверка компиляции C# Unity-проекта без запуска Unity (5–30 с вместо 1–3 мин).

Берёт .csproj, которые Unity сгенерировал в корне проекта (для IDE), копирует их во
временную папку и ЧИНИТ главную их беду — устаревший список файлов: Unity обновляет
.csproj только при синхронизации с IDE, поэтому новые .cs в них отсутствуют, а удалённые
остаются. Скрипт заново раскладывает все .cs из Assets/ и Packages/ по сборкам
(asmdef → своя сборка, папка Editor → Assembly-CSharp-Editor, остальное → Assembly-CSharp),
затем запускает `dotnet build` и печатает только ошибки.

Проект НЕ трогается: копии и obj/bin — во временной папке.

Использование:
  python fast_check.py                       # сборки, где есть изменённые по git .cs
  python fast_check.py --path Assets/X/A.cs  # сборка(и), куда входит файл/папка
  python fast_check.py --csproj GameCore     # конкретная сборка (с зависимостями)
  python fast_check.py --all                 # все сборки проекта (долго)
  --project <корень>   по умолчанию — ближайший вверх каталог с ProjectSettings/
  --warnings           показать и предупреждения по изменённым файлам

Ограничения (тогда нужна проверка в Unity — unity_batch.py):
  • шейдеры, .uxml, ассеты, сериализация не проверяются — только C#;
  • .asmref, versionDefines и defineConstraints asmdef не учитываются;
  • генераторы кода Unity (FishNet codegen, IL post-processing) не запускаются;
  • если .csproj нет вовсе (проект ни разу не открывали с IDE) — сначала
    сгенерировать: unity_batch.py --method UnityEditor.SyncVS.SyncSolution  (или открыть IDE).
"""
import argparse, hashlib, os, re, subprocess, sys, tempfile

SKIP_DIR = re.compile(r'(^|/)(\.[^/]*|[^/]*~)(/|$)')  # Unity игнорирует .папки и папки~


def find_project(start):
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, 'ProjectSettings')) and os.path.isdir(os.path.join(d, 'Assets')):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            sys.exit('Не найден корень Unity-проекта (папка с ProjectSettings/ и Assets/). Укажите --project.')
        d = nd


def norm(p):
    return os.path.normcase(os.path.abspath(p)).replace('\\', '/')


def rel(root, p):
    return os.path.relpath(p, root).replace('\\', '/')


def read(p):
    with open(p, encoding='utf-8-sig') as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project')
    ap.add_argument('--path', nargs='*', default=[])
    ap.add_argument('--csproj', nargs='*', default=[])
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--warnings', action='store_true')
    a = ap.parse_args()

    root = find_project(a.project or os.getcwd())
    csprojs = sorted(f for f in os.listdir(root) if f.endswith('.csproj'))
    if not csprojs:
        sys.exit('В корне проекта нет .csproj. Сгенерируйте их: откройте проект в IDE '
                 'или unity_batch.py --method UnityEditor.SyncVS.SyncSolution')
    names = {c[:-7] for c in csprojs}
    texts = {c: read(os.path.join(root, c)) for c in csprojs}

    # --- корни asmdef → сборка
    asm_roots = {}
    for c, t in texts.items():
        for m in re.finditer(r'<None Include="([^"]+\.asmdef)"', t):
            p = m.group(1)
            p = p if os.path.isabs(p) else os.path.join(root, p)
            asm_roots[norm(os.path.dirname(p))] = c[:-7]
    roots_sorted = sorted(asm_roots, key=len, reverse=True)

    def owner(path_abs):
        n = norm(path_abs)
        for r in roots_sorted:
            if n.startswith(r + '/'):
                return asm_roots[r]
        rp = rel(root, path_abs)
        if not rp.startswith('Assets/'):
            return None  # пакеты без asmdef не компилируются
        first = re.match(r'Assets/(Plugins|Standard Assets|Pro Standard Assets)/', rp) is not None
        editor = '/Editor/' in '/' + rp
        name = 'Assembly-CSharp' + ('-Editor' if editor else '') + ('-firstpass' if first else '')
        return name if name in names else None

    # --- все .cs проекта
    owned = {n: [] for n in names}
    for base in ('Assets', 'Packages'):
        for dp, dns, fns in os.walk(os.path.join(root, base)):
            rdp = rel(root, dp)
            if SKIP_DIR.search(rdp + '/'):
                dns[:] = []
                continue
            for fn in fns:
                if fn.endswith('.cs'):
                    p = os.path.join(dp, fn)
                    o = owner(p)
                    if o:
                        owned[o].append(p)

    # --- что собирать
    targets = set(a.csproj)
    for c in targets:
        if c not in names:
            sys.exit(f'Нет сборки {c}. Есть: {", ".join(sorted(names))}')
    paths = list(a.path)
    if not a.all and not targets and not paths:
        try:
            out = subprocess.run(['git', 'status', '--porcelain', '-uall'], cwd=root, capture_output=True,
                                 text=True, encoding='utf-8', errors='replace').stdout
            for line in out.splitlines():
                f = line[3:].strip().strip('"')
                if ' -> ' in f:
                    f = f.split(' -> ')[1]
                if f.endswith('.cs'):
                    paths.append(f)
        except FileNotFoundError:
            pass
        if not paths:
            sys.exit('Изменённых .cs по git нет. Укажите --path, --csproj или --all.')
    changed = set()
    for p in paths:
        pa = os.path.join(root, p) if not os.path.isabs(p) else p
        if os.path.isdir(pa):
            for n, files in owned.items():
                if any(norm(f).startswith(norm(pa) + '/') for f in files):
                    targets.add(n)
        else:
            changed.add(norm(pa))
            o = owner(pa)
            if o:
                targets.add(o)
    if a.all:
        targets = set(names)
    if not targets:
        sys.exit('Не удалось определить сборку для указанных файлов (вне Assets/ и asmdef?).')

    # --- копии .csproj с актуальными файлами и абсолютными путями
    work = os.path.join(tempfile.gettempdir(), 'unity_fastcheck', hashlib.md5(root.encode()).hexdigest()[:8])
    os.makedirs(work, exist_ok=True)

    def absify(m):
        tag, attr, p = m.group(1), m.group(2), m.group(3)
        if tag == 'ProjectReference' or os.path.isabs(p) or p.startswith('$('):
            return m.group(0)
        return f'<{tag} {attr}="{os.path.join(root, p)}"'

    for c, t in texts.items():
        n = c[:-7]
        # убрать Compile из проекта, кроме файлов вне Assets/ и Packages/ (PackageCache и т. п.)
        def keep_compile(m):
            p = m.group(1)
            pa = p if os.path.isabs(p) else os.path.join(root, p)
            r = rel(root, pa)
            return m.group(0) if not (r.startswith('Assets/') or r.startswith('Packages/')) else ''
        t = re.sub(r'\s*<Compile Include="([^"]+)"\s*/>', keep_compile, t)
        inc = ''.join(f'\n    <Compile Include="{f}" />' for f in sorted(owned.get(n, [])))
        t = t.replace('<ItemGroup>', '<ItemGroup>' + inc, 1) if inc else t
        t = re.sub(r'<(Compile|None|Analyzer|Reference|ProjectReference|AdditionalFiles) (Include)="([^"]+)"', absify, t)
        t = re.sub(r'<HintPath>(?!/|[A-Za-z]:)([^<]+)</HintPath>', lambda m: f'<HintPath>{os.path.join(root, m.group(1))}</HintPath>', t)
        with open(os.path.join(work, c), 'w', encoding='utf-8') as f:
            f.write(t)

    # --- сборка
    errors, warns, failed = [], [], False
    for n in sorted(targets):
        print(f'[fast_check] dotnet build {n} ...', flush=True)
        r = subprocess.run(['dotnet', 'build', os.path.join(work, n + '.csproj'), '-nologo', '-v', 'q',
                            '-p:TreatWarningsAsErrors=false'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=work)
        failed |= r.returncode != 0
        for line in (r.stdout + r.stderr).splitlines():
            s = re.sub(r'\s*\[[^\]]+\.csproj\]\s*$', '', line.strip())
            if ': error ' in s and s not in errors:
                errors.append(s)
            elif ': warning ' in s and s not in warns:
                if any(norm(s.split('(')[0]) == c for c in changed):
                    warns.append(s)
        if r.returncode != 0 and not any(': error ' in l for l in r.stdout.splitlines()):
            print(r.stdout[-3000:], r.stderr[-2000:])

    for e in errors:
        print('ERROR', e)
    if a.warnings:
        for w in warns:
            print('WARN ', w)
    print(f'[fast_check] сборок: {len(targets)} ({", ".join(sorted(targets))}), ошибок: {len(errors)}'
          + (f', предупреждений в изменённых файлах: {len(warns)}' if a.warnings else ''))
    sys.exit(1 if failed or errors else 0)


if __name__ == '__main__':
    main()
