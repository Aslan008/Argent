#!/usr/bin/env python3
"""
«Посмотреть» Unity-проект без Unity — по текстовым YAML-файлам и .meta.

  python unity_find.py guid  <путь ассета>        # GUID из .meta
  python unity_find.py path  <guid>               # какой ассет у GUID
  python unity_find.py refs  <путь ассета|guid>   # кто ссылается (сцены, префабы, материалы, SO…)
  python unity_find.py scene <файл .unity|.prefab> [--filter текст]
        # иерархия: объекты, активность, компоненты (скрипты — по именам файлов),
        # префаб-инстансы с источником
  python unity_find.py script <ИмяКласса>          # где файл скрипта и его GUID

Требует Asset Serialization = Force Text (по умолчанию в новых проектах).
Ссылки из кода (Resources.Load, Addressables по ключу, строки) так не найти — ищите grep по коду.
"""
import os, re, sys

TEXT_EXT = ('.unity', '.prefab', '.mat', '.asset', '.controller', '.overrideController', '.anim',
            '.playable', '.mask', '.spriteatlas', '.spriteatlasv2', '.lighting', '.preset', '.terrainlayer',
            '.shadergraph', '.shadersubgraph', '.vfx', '.signal', '.guiskin', '.uss', '.uxml', '.tss',
            '.asmdef', '.asmref', '.inputactions', '.renderTexture', '.cubemap', '.flare', '.physicMaterial',
            '.physicsMaterial2D', '.mixer', '.brush', '.fontsettings')


def find_project(start):
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, 'ProjectSettings')) and os.path.isdir(os.path.join(d, 'Assets')):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            sys.exit('Не найден корень Unity-проекта.')
        d = nd


ROOT = find_project(os.getcwd())


def walk(exts=None, metas=False):
    # .meta ищем и в кэше пакетов реестра — чтобы опознать скрипты TextMeshPro, UI, URP…
    bases = ('Assets', 'Packages', os.path.join('Library', 'PackageCache')) if metas else ('Assets', 'Packages')
    for base in bases:
        for dp, dns, fns in os.walk(os.path.join(ROOT, base)):
            dns[:] = [d for d in dns if not d.startswith('.') and not d.endswith('~')]
            for fn in fns:
                if metas and fn.endswith('.meta'):
                    yield os.path.join(dp, fn)
                elif not metas and (exts is None or fn.endswith(exts)):
                    yield os.path.join(dp, fn)


def rel(p):
    return os.path.relpath(p, ROOT).replace('\\', '/')


def guid_of(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    try:
        m = re.search(r'^guid:\s*([0-9a-f]{32})', open(p + '.meta', encoding='utf-8', errors='replace').read(), re.M)
        return m.group(1) if m else None
    except FileNotFoundError:
        return None


_guid_map = None


def guid_map():
    global _guid_map
    if _guid_map is None:
        _guid_map = {}
        for m in walk(metas=True):
            try:
                with open(m, encoding='utf-8', errors='replace') as f:
                    g = re.search(r'^guid:\s*([0-9a-f]{32})', f.read(400), re.M)
                if g:
                    _guid_map[g.group(1)] = rel(m[:-5])
            except OSError:
                pass
    return _guid_map


def cmd_refs(arg):
    g = arg if re.fullmatch(r'[0-9a-f]{32}', arg) else guid_of(arg)
    if not g:
        sys.exit('GUID не найден (нет .meta?)')
    print(f'GUID {g} = {guid_map().get(g, "?")}')
    n = 0
    for f in walk(TEXT_EXT):
        try:
            with open(f, encoding='utf-8', errors='replace') as fh:
                t = fh.read()
        except OSError:
            continue
        c = t.count(g)
        if c:
            n += 1
            print(f'  {rel(f)}  ({c})')
    print(f'Файлов со ссылкой: {n}')


def cmd_scene(path, flt=None):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    text = open(p, encoding='utf-8', errors='replace').read()
    docs = re.split(r'^--- !u!(\d+) &(-?\d+)( stripped)?\s*$', text, flags=re.M)
    objs = {}
    for i in range(1, len(docs) - 1, 4):
        cls, fid, body = docs[i], docs[i + 1], docs[i + 3]
        typ = body.strip().split(':', 1)[0]
        objs[fid] = (cls, typ, body)
    gm = guid_map()

    def field(body, name):
        m = re.search(rf'^\s*{name}:\s*(.*)$', body, re.M)
        return m.group(1).strip() if m else ''

    def fileid(s):
        m = re.search(r'fileID:\s*(-?\d+)', s)
        return m.group(1) if m else '0'

    go_comps, tr_of_go, parent_of_tr, children = {}, {}, {}, {}
    for fid, (cls, typ, body) in objs.items():
        if typ == 'GameObject':
            go_comps[fid] = re.findall(r'component:\s*\{fileID:\s*(-?\d+)\}', body)
        if typ in ('Transform', 'RectTransform'):
            go = fileid(field(body, 'm_GameObject'))
            tr_of_go[go] = fid
            parent_of_tr[fid] = fileid(field(body, 'm_Father'))
            children[fid] = re.findall(r'-\s*\{fileID:\s*(-?\d+)\}', body.split('m_Children:')[1].split('m_Father')[0]) if 'm_Children:' in body else []

    def comp_name(cid):
        if cid not in objs:
            return '?'
        cls, typ, body = objs[cid]
        if typ == 'MonoBehaviour':
            m = re.search(r'm_Script:\s*\{[^}]*guid:\s*([0-9a-f]{32})', body)
            if m:
                sp = gm.get(m.group(1))
                return os.path.splitext(os.path.basename(sp))[0] if sp else f'Script({m.group(1)[:8]}… не найден — Missing?)'
            return 'MonoBehaviour(без скрипта — Missing!)'
        return typ

    go_of_tr = {v: k for k, v in tr_of_go.items()}

    def show(tr, depth):
        go = go_of_tr.get(tr)
        if go is None:
            return
        body = objs[go][2]
        name = field(body, 'm_Name')
        active = field(body, 'm_IsActive') != '0'
        comps = [comp_name(c) for c in go_comps.get(go, [])]
        line = f'{"  " * depth}{"" if active else "[off] "}{name}  — {", ".join(comps)}'
        if not flt or flt.lower() in line.lower():
            print(line)
        for ch in children.get(tr, []):
            show(ch, depth + 1)

    roots = [t for t, par in parent_of_tr.items() if par == '0']
    for r in roots:
        show(r, 0)
    for fid, (cls, typ, body) in objs.items():
        if typ == 'PrefabInstance':
            m = re.search(r'm_SourcePrefab:\s*\{[^}]*guid:\s*([0-9a-f]{32})', body)
            src = gm.get(m.group(1), m.group(1)) if m else '?'
            nm = re.search(r'propertyPath: m_Name\s*\n\s*value:\s*(.*)', body)
            line = f'PrefabInstance «{nm.group(1).strip() if nm else "?"}» ← {src}'
            if not flt or flt.lower() in line.lower():
                print(line)


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    c, arg = sys.argv[1], sys.argv[2]
    if c == 'guid':
        print(guid_of(arg) or 'нет .meta')
    elif c == 'path':
        print(guid_map().get(arg, 'не найден в Assets/Packages (может быть в Library/PackageCache)'))
    elif c == 'refs':
        cmd_refs(arg)
    elif c == 'scene':
        flt = sys.argv[sys.argv.index('--filter') + 1] if '--filter' in sys.argv else None
        cmd_scene(arg, flt)
    elif c == 'script':
        for f in walk(('.cs',)):
            with open(f, encoding='utf-8', errors='replace') as fh:
                if re.search(rf'\b(class|struct|interface|enum)\s+{re.escape(arg)}\b', fh.read()):
                    print(rel(f), guid_of(f))
    else:
        print(__doc__); sys.exit(1)


if __name__ == '__main__':
    main()
