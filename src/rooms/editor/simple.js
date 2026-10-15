// Простой режим редактора комнат: блок объясняется парой полей, а не десятью.
//
// Всё, что умеет движок, остаётся: «Все настройки блока» раскрывают прежнюю форму, а переключатель
// «Простой / Полный» в шапке показывает полные формы сразу. Здесь только то, что нужно, чтобы собрать
// комнату без знания {{task}}, state.x, writes, JSON-ответа и sets:
//   * значения вставляются кнопками-«пилюлями» (что известно на этом шаге), а не набираются;
//   * ответ модели сам становится значением шага — его можно назвать и вставить дальше;
//   * блок Git: действие словами, поля на выходе — готовые.
// Загружается до app.js и обращается к его функциям только во время работы.

const MODE_KEY = "rooms.ui.mode";
const SAVE_AS_TYPES = ["tool", "agent", "mcp", "plugin", "leeth", "skill", "git", "http"];
const GIT_WRITES = ["add", "commit", "branch", "checkout", "stash", "pull", "push"];
const READ_ONLY_TOOLS = ["read_file", "grep_search", "search_files", "list_directory", "get_file_outline", "read_git_diff"];
const EFFORT = {quick: [3, 10000, "Быстро"], normal: [6, 20000, "Обычно"], deep: [12, 40000, "Тщательно"]};

function uiMode() {
  try { return localStorage.getItem(MODE_KEY) === "advanced" ? "advanced" : "simple"; } catch (e) { return "simple"; }
}
function isSimple() { return uiMode() === "simple"; }
function setUiMode(mode) {
  try { localStorage.setItem(MODE_KEY, mode); } catch (e) { /* без хранилища режим живёт до перезагрузки */ }
  syncModeButton();
  if (typeof panel === "function") panel();
}
function syncModeButton() {
  const b = document.getElementById("btnMode");
  if (!b) return;
  const simple = isSimple();
  b.textContent = simple ? "Режим: простой" : "Режим: полный";
  b.setAttribute("aria-pressed", simple ? "false" : "true");
  b.title = simple ? "Показаны основные поля. Нажмите, чтобы видеть все настройки каждого блока"
    : "Показаны все настройки. Нажмите, чтобы вернуться к простым формам";
}
(function initMode() {
  const b = document.getElementById("btnMode");
  if (b) { b.addEventListener("click", () => setUiMode(isSimple() ? "advanced" : "simple")); syncModeButton(); }
})();

// ── человеческие названия значений ─────────────────────────────────────────────────
function pillLabel(name) {
  if (name === "task") return "Задача";
  const dot = name.lastIndexOf(".");
  if (dot > 0) {
    const id = name.slice(0, dot), key = name.slice(dot + 1);
    const node = typeof nodeById === "function" ? nodeById(id) : null;
    if (key === "output") return node && node.type === "agent" ? `Ответ «${id}»` : `Вывод «${id}»`;
    if (node && node.type === "git") {
      const what = (GIT_FIELDS[node.mode || "status"] || {})[key];
      if (what) return `«${id}»: ${what}`;
    }
  }
  return name;
}

// какие шаги стоят раньше выбранного: их результаты показываем первыми
function upstreamIds(id) {
  const before = new Set();
  if (!id || typeof room === "undefined" || !room) return before;
  const stack = [id];
  while (stack.length) {
    const cur = stack.pop();
    for (const e of room.edges) if (e.to === cur && !before.has(e.from) && e.from !== id) { before.add(e.from); stack.push(e.from); }
  }
  return before;
}
function fieldOwners() {
  const own = new Map();
  const put = (k, id) => { if (k && !own.has(k)) own.set(k, id); };
  for (const n of room.nodes) {
    for (const k of Object.keys(n.sets || {})) put(k, n.id);
    for (const k of n.writes || []) put(k, n.id);
    for (const k of Object.keys(n.values || {})) put(k, n.id);
    if (n.save_as) put(n.save_as, n.id);
    if (n.into) { put(n.into, n.id); put(n.into + "_approved", n.id); }
    if (n.item) put(n.item, n.id);
  }
  return own;
}

// ── кнопки вместо набора {{полей}} ─────────────────────────────────────────────────
function pillInserter(control) {
  const all = [...knownFields().entries()];
  const owners = fieldOwners();
  const here = sel && sel.kind === "node" ? sel.id : null;
  const before = upstreamIds(here);
  const params = new Set(Object.keys(room.defaults || {}));
  const rank = (k) => {
    if (k === "task") return 0;
    if (params.has(k)) return 1;
    const dot = k.indexOf(".");
    const owner = owners.get(k) || (dot > 0 ? k.slice(0, dot) : null);
    if (owner && owner === here) return 9;           // то, что этот шаг сам пишет, ему ещё не доступно
    if (owner && before.has(owner)) return k.endsWith(".output") ? 2 : dot > 0 ? 4 : 3;   // ответы и названные результаты — раньше полей git
    return 5;
  };
  const sorted = all.map((kv, i) => [kv, i]).sort((a, b) => rank(a[0][0]) - rank(b[0][0]) || a[1] - b[1]).map((x) => x[0]);
  const useful = sorted.filter(([k]) => rank(k) < 5);
  const shown = useful.slice(0, 9);
  const insert = (name) => {
    const ins = "{{" + name + "}}";
    const t = control;
    const a = t.selectionStart ?? t.value.length, b = t.selectionEnd ?? a;
    t.value = t.value.slice(0, a) + ins + t.value.slice(b);
    t.focus(); t.setSelectionRange(a + ins.length, a + ins.length);
    t.dispatchEvent(new Event("input"));
  };
  const bar = el("div", {class: "pills", role: "group", "aria-label": "Вставить значение"});
  if (shown.length) bar.append(el("span", {class: "pillnote"}, "Вставить:"));
  for (const [k, why] of shown) {
    bar.append(el("button", {class: "pill", type: "button", title: "{{" + k + "}} — " + why, onclick: () => insert(k)}, pillLabel(k)));
  }
  const rest = sorted.filter((kv) => !shown.includes(kv));
  if (rest.length) {
    const s = el("select", {class: "ins", "aria-label": "Вставить другое значение"}, el("option", {value: ""}, "другое…"));
    for (const [k, why] of rest) s.append(el("option", {value: k, title: why}, pillLabel(k) === k ? k + " — " + why : pillLabel(k)));
    s.addEventListener("change", () => { if (s.value) { insert(s.value); s.value = ""; } });
    bar.append(s);
  }
  return el("div", {}, control, bar);
}

// ── «Назвать результат» ────────────────────────────────────────────────────────────
function saveAsField(side, n, upd) {
  side.dataset.saveAs = "1";
  const note = el("div", {class: "hint"}, "Результат шага получит это имя; дальше его можно вставить как {{имя}} или проверить в условии.");
  const i = inp(n.save_as || "", (v) => {
    const name = v.trim();
    if (name && !/^[A-Za-z_]\w*$/.test(name)) { note.textContent = "Имя: латиница, цифры и «_», не с цифры."; return; }
    note.textContent = "Результат шага получит это имя; дальше его можно вставить как {{имя}} или проверить в условии.";
    n.save_as = name; upd("saveas" + n.id);
  }, {ph: n.type === "agent" ? "например: план" : n.type === "git" ? "например: отчёт_git" : "например: вывод"});
  const f = el("div", {class: "field"}, el("label", {}, "Назвать результат"), i, note);
  side.append(f);
}
function afterNodeForm(side, n, upd) {
  if (SAVE_AS_TYPES.includes(n.type) && !side.dataset.saveAs) saveAsField(side, n, upd);
  delete side.dataset.saveAs;
}

// ── сворачиваемые «Все настройки блока» ────────────────────────────────────────────
function allSettings(side, build) {
  const body = el("div", {class: "body"});
  side.append(el("details", {class: "adv"}, el("summary", {}, "Все настройки блока"), body));
  build(body);
}

// ── входная точка для nodePanel ────────────────────────────────────────────────────
function simpleNodeForm(side, n, upd) {
  if (n.type === "git") { gitNodeForm(side, n, upd); return true; }
  if (n.type === "http") { httpNodeForm(side, n, upd); return true; }
  if (!isSimple()) return false;
  if (n.type === "mcp" && typeof profileNodeForm === "function" && profileNodeForm(side, n, upd)) return true;
  if (n.type === "agent") { simpleAgentForm(side, n, upd); return true; }
  if (n.type === "tool" && n.tool === "run_command" && typeof toolNodeForm === "function") { simpleCommandForm(side, n, upd); return true; }
  return false;
}

// ── агент ──────────────────────────────────────────────────────────────────────────
function abilityOf(n) {
  const known = (typeof META !== "undefined" && META.tools) || [];
  const readSet = READ_ONLY_TOOLS.filter((t) => known.includes(t));
  if (n.tools != null) {
    if (n.tools.length === 0) return "none";            // ни файлов, ни команд (например, «советник» с веб-ИИ)
    return n.tools.every((t) => readSet.includes(t)) ? "read" : "custom";
  }
  return "write";
}
function effortOf(n) {
  const b = n.budget || {};
  for (const [key, [it, tok]] of Object.entries(EFFORT)) if (b.max_iterations === it && b.max_tokens === tok) return key;
  return "custom";
}
function simpleAgentForm(side, n, upd) {
  const ins = area(n.instruction, (v) => { n.instruction = v; upd("inst" + n.id); }, 6);
  side.append(field("Что должна сделать модель", withInserter(ins),
    "Пишите как человеку. Нужные значения (задача, ответы прошлых шагов) вставляйте кнопками."));
  side.append(field("Что модель может", select([
    ["read", "Только читать и искать (ничего не меняет)"],
    ["write", "Читать, менять файлы и запускать команды"],
    ...(abilityOf(n) === "custom" ? [["custom", "Свой набор (в «Всех настройках»)"]] : []),
    ...(abilityOf(n) === "none" ? [["none", "Ничего не меняет и не читает — только советуется"]] : []),
  ], abilityOf(n), (v) => {
    const known = (typeof META !== "undefined" && META.tools) || [];
    if (v === "read") { n.role = "Researcher"; n.tools = READ_ONLY_TOOLS.filter((t) => known.includes(t)); }
    if (v === "write") { n.role = "Coder"; delete n.tools; }
    upd("ability" + n.id); panel();
  })));
  const effort = effortOf(n);
  side.append(field("Насколько тщательно", select([
    ...Object.entries(EFFORT).map(([k, v]) => [k, v[2]]),
    ...(effort === "custom" ? [["custom", "Свои лимиты (в «Всех настройках»)"]] : []),
  ], effort, (v) => {
    if (EFFORT[v]) n.budget = {...(n.budget || {}), max_iterations: EFFORT[v][0], max_tokens: EFFORT[v][1]};
    upd("effort" + n.id);
  }), "Чем тщательнее, тем больше шагов и токенов модель вправе потратить."));
  if (typeof profileAgentCard === "function") profileAgentCard(side, n, upd);
  saveAsField(side, n, upd);
  if ((n.writes || []).length) {
    side.append(el("div", {class: "notice note"}, "Этот блок просит у модели ответ полями: " + n.writes.join(", ") +
      ". Так задумано в расширенных настройках; в простом режиме достаточно «Назвать результат»."));
  }
  allSettings(side, (body) => agentFormLegacy(body, n, upd));
}

// ── команда ────────────────────────────────────────────────────────────────────────
function simpleCommandForm(side, n, upd) {
  n.args = n.args || {};
  const cmd = area(n.args.command || "", (v) => { n.args.command = v; upd("cmd" + n.id); }, 2, true);
  side.append(field("Команда", withInserter(cmd), "Выполнится как есть, без модели."));
  const verdict = Object.keys(n.sets || {}).find((k) => n.sets[k] === "ok") || "";
  side.append(field("Запомнить, прошла ли команда, как", inp(verdict, (v) => {
    const name = v.trim();
    const rest = {};
    for (const [k, x] of Object.entries(n.sets || {})) if (x !== "ok") rest[k] = x;
    n.sets = name && /^[A-Za-z_]\w*$/.test(name) ? {[name]: "ok", ...rest} : rest;
    upd("verdict" + n.id);
  }, {ph: "например: тесты_прошли"}), "Прошла = код возврата 0. Потом: стрелка «если <имя>»."));
  side.append(check("Спросить вас перед запуском", n.confirm, (v) => { n.confirm = v; upd("confirm"); }, "Нет «да» — команда не выполнится."));
  saveAsField(side, n, upd);
  allSettings(side, (body) => toolNodeForm(body, n, upd));
}

// ── Git ────────────────────────────────────────────────────────────────────────────
function gitNodeForm(side, n, upd) {
  n.args = n.args || {};
  const a = n.args;
  const mode = n.mode || "status";
  const clear = (key) => { if (a[key] === "" || a[key] === false || a[key] == null) delete a[key]; };
  const setArg = (key, v) => { a[key] = v; clear(key); upd("g" + key + n.id); };
  side.append(field("Что сделать", select(Object.entries(GIT_LABEL), mode, (v) => {
    const repo = a.repo;
    n.mode = v; n.args = repo ? {repo} : {};
    n.confirm = GIT_WRITES.includes(v) && v !== "branch";
    upd("gmode"); panel();
  }), "Git запускается сам, без командной строки и без модели. Принудительных (force) действий здесь нет."));

  if (mode === "diff") {
    const kind = a.staged ? "staged" : a.against ? "other" : "head";
    side.append(field("Сравнить с", select([["head", "последним коммитом"], ["staged", "тем, что уже добавлено в коммит"], ["other", "другой веткой или коммитом"]], kind, (v) => {
      delete a.staged; delete a.against;
      if (v === "staged") a.staged = true;
      if (v === "other") a.against = "main";
      upd("gdiffkind"); panel();
    })));
    if (kind === "other") side.append(field("Ветка или коммит", inp(a.against || "", (v) => setArg("against", v.trim()), {ph: "main"})));
    side.append(check("Показать всю разницу, а не только список файлов", a.full, (v) => setArg("full", v)));
  }
  if (mode === "log") side.append(field("Сколько коммитов", inp(a.count ?? 10, (v) => setArg("count", v || 10), {type: "number", min: 1})));
  if (mode === "add") side.append(field("Какие файлы", inp(Array.isArray(a.files) ? a.files.join(", ") : (a.files || ""), (v) => setArg("files", v.trim()), {ph: "пусто — все"}),
    "Через запятую. Пусто — все изменённые и новые."));
  if (mode === "commit") {
    side.append(field("Сообщение коммита", withInserter(area(n.message || "", (v) => { n.message = v; upd("gmsg" + n.id); }, 3)),
      "Коротко: что сделано. Сообщение не исполняется как команда."));
    side.append(field("Какие файлы", inp(Array.isArray(a.files) ? a.files.join(", ") : (a.files || ""), (v) => setArg("files", v.trim()), {ph: "пусто — все изменённые"})));
    side.append(check("Только то, что уже добавлено в коммит", a.only_staged, (v) => setArg("only_staged", v)));
  }
  if (mode === "branch") side.append(field("Имя ветки", inp(a.name || "", (v) => setArg("name", v.trim()), {ph: "пусто — только показать"}),
    "Есть такая ветка — переключится, нет — создаст и переключится."));
  if (mode === "checkout") {
    side.append(field("Ветка", inp(a.branch || "", (v) => setArg("branch", v.trim()), {ph: "main"})));
    side.append(check("Создать, если её нет", a.create, (v) => setArg("create", v)));
  }
  if (mode === "stash") {
    side.append(field("Действие", select([["save", "отложить изменения"], ["pop", "вернуть отложенное"], ["list", "показать отложенное"]], a.action || "save", (v) => {
      a.action = v; if (v === "save") delete a.action; upd("gstash"); panel();
    })));
    if (!a.action || a.action === "save") side.append(field("Подпись", inp(n.message || "", (v) => { n.message = v; upd("gstmsg"); }, {ph: "необязательно"})));
  }
  if (mode === "pull" || mode === "push") {
    side.append(el("div", {class: "row"},
      field("Куда", inp(a.remote || "", (v) => setArg("remote", v.trim()), {ph: "origin"})),
      field("Ветка", inp(a.branch || "", (v) => setArg("branch", v.trim()), {ph: "текущая"}))));
    if (mode === "pull") side.append(el("p", {class: "hint"}, "Только простое обновление: если ветки разошлись, блок сообщит об ошибке, а не склеит их сам."));
  }
  side.append(check("Спросить вас перед запуском", n.confirm, (v) => { n.confirm = v; upd("confirm"); },
    GIT_WRITES.includes(mode) ? "Меняет репозиторий. Нет «да» — шаг не выполнится." : "Только смотрит; вопрос не нужен."));

  const fields = GIT_FIELDS[mode] || {};
  const outs = el("div", {class: "pills static"});
  for (const [k, why] of Object.entries(fields)) outs.append(el("span", {class: "pill ghost", title: "{{" + n.id + "." + k + "}}"}, why));
  side.append(el("div", {class: "field"}, el("label", {}, "Что блок оставляет"), outs,
    el("div", {class: "hint"}, "Эти значения можно вставить дальше или проверить в условии стрелки: «" + n.id + "." + Object.keys(fields)[0] + "».")));
  saveAsField(side, n, upd);
  side.append(el("details", {class: "adv"}, el("summary", {}, "Дополнительно"), el("div", {class: "body"},
    field("Папка репозитория", inp(a.repo || "", (v) => setArg("repo", v.trim()), {ph: "пусто — папка проекта"})),
    field("Запомнить из отчёта", kvEditor(n.sets, (v) => { n.sets = v; upd("sets" + n.id); },
      {keyPh: "поле, напр. ok_git", keyList: "fieldsList", valueChoices: META.derived.map((d) => [d, DERIVED_LABEL[d] || d]), addLabel: "Запомнить поле"})))));
}

// ── палитра: частые действия и Git ─────────────────────────────────────────────────
function simplePaletteEntries(add, C) {
  C.common = "Частые действия";
  C.git = "Git";
  const known = (typeof META !== "undefined" && META.tools) || [];
  const read = READ_ONLY_TOOLS.filter((t) => known.includes(t));
  add(C.common, "agent", "Спросить модель", "Читает и отвечает, ничего не меняет", {role: "Researcher", tools: read, instruction: "{{task}}"}, "llm модель вопрос ответ спросить agent");
  add(C.common, "agent", "Поручить модели правку", "Читает, меняет файлы, запускает команды", {role: "Coder", tools: null, instruction: "{{task}}"}, "правка код coder исправить сделать");
  add(C.common, "tool", "Запустить команду", "Любая команда, без модели", {tool: "run_command", args: {command: ""}}, "команда shell запустить run cmd");
  add(C.common, "tool", "Запустить тесты", "Прошли ли — запомнит само", {tool: "run_command", args: {command: "python -m pytest -q"}, sets: {tests_passed: "ok", tests_tail: "tail"}}, "тесты pytest test проверить");
  add(C.common, "git", "Что изменилось в проекте", "Ветка, список файлов, чисто ли", {mode: "status"}, "git изменения status изменилось");
  add(C.common, "git", "Закоммитить изменения", "Сначала спросит вас", {mode: "commit", message: "Обновление проекта", confirm: true}, "git commit коммит сохранить");
  add(C.common, "human_pause", "Спросить вас", "Остановиться и дождаться ответа", {}, "вопрос спросить человек подтвердить");
  add(C.common, "human_pause", "Спросить с вариантами", "Кнопки вместо ввода: выбор определяет путь",
    {prompt: "Как поступить?", options: ["Продолжить", "Остановиться"], into: "decision"}, "вопрос варианты кнопки выбор меню развилка");
  add(C.common, "notify", "Сообщить", "В консоль, окном ПК или в Telegram", {message: "Готово"}, "уведомление сообщить telegram");
  C.net = "Сеть и интернет";
  add(C.net, "http", "Получить по адресу (GET)", "Страница или ответ API в поля", {mode: "get", args: {url: "https://"}}, "http запрос get api сайт получить страница адрес url");
  add(C.net, "http", "Отправить данные (POST JSON)", "Сначала спросит вас", {mode: "post", args: {url: "https://", json: {}}, confirm: true}, "http post отправить api json webhook вебхук");
  add(C.net, "http", "Проверить, что сайт отвечает", "Потом: стрелка «если шаг.ok»; код — «шаг.status»", {mode: "head", args: {url: "https://"}}, "http head проверить сайт жив отвечает доступен");
  const MORE = {
    status: "Ветка, список файлов, чисто ли", diff: "Что именно поменялось, сколько строк", log: "Последние коммиты списком",
    add: "Подготовить файлы к коммиту", commit: "Записать изменения одним коммитом", branch: "Показать, создать или переключить ветку",
    checkout: "Перейти на другую ветку", stash: "Убрать изменения в сторону и вернуть", pull: "Получить новое с сервера",
    push: "Отправить коммиты на сервер",
  };
  for (const [m, label] of Object.entries(GIT_LABEL)) {
    const preset = {mode: m};
    if (m === "commit") preset.message = "Обновление проекта";
    if (m === "checkout") preset.args = {branch: "main"};
    if (GIT_WRITES.includes(m) && m !== "branch") preset.confirm = true;
    add(C.git, "git", label, MORE[m] || "", preset, "git гит " + m + " " + label);
  }
}

// ── вопрос с вариантами ────────────────────────────────────────────────────────────
function optionsEditor(side, n, upd) {
  const list = n.options || (n.options = []);
  const box = el("div", {class: "kv"});
  const draw = () => {
    box.replaceChildren();
    list.forEach((o, i) => box.append(el("div", {class: "kvrow"},
      inp(o, (v) => { list[i] = v; upd("opt" + n.id); }, {ph: "вариант ответа"}),
      el("button", {class: "btn icon ghost sm", title: "Убрать", "aria-label": "Убрать вариант", onclick: () => { list.splice(i, 1); upd("opt" + n.id); draw(); }}, icon("i-close", "sm")))));
    box.append(el("button", {class: "btn sm", onclick: () => { list.push(""); upd("opt" + n.id); draw(); }}, icon("i-plus", "sm"), "Добавить вариант"));
  };
  draw();
  side.append(field("Варианты ответа", box,
    "Если заданы, вам предложат кнопки (в консоли — меню) вместо ввода текста. Выбранное лежит в «" + n.id + ".choice»."));
  const ready = list.map((o) => String(o).trim()).filter(Boolean);
  if (ready.length >= 2 && n.into) {
    side.append(el("button", {class: "btn sm", title: "Создаст проверку поля «" + n.into + "» и по стрелке на каждый вариант",
      onclick: () => { SWITCH_SEED = {field: n.into, values: ready}; sel = {kind: "node", id: n.id}; inspectMode = "switch"; panel(); }},
      "Развести по вариантам (создать стрелки)"));
  } else if (ready.length >= 2) {
    side.append(el("p", {class: "hint"}, "Укажите «Сохранить ответ в поле» — тогда можно создать по стрелке на каждый вариант."));
  }
}

// ── запрос в сеть ──────────────────────────────────────────────────────────────────
const HTTP_LABEL = {
  get: "Получить (GET)", post: "Отправить данные (POST)", put: "Заменить (PUT)", patch: "Изменить (PATCH)",
  delete: "Удалить (DELETE)", head: "Только проверить, что отвечает (HEAD)",
};
const HTTP_READS = ["get", "head"];
const HTTP_FIELDS = {
  status: "код ответа (200, 404…)", ok: "успешно ли (код 2xx–3xx)", body: "текст ответа",
  json: "ответ как данные (если это JSON)", type: "тип содержимого", seconds: "сколько секунд ждали",
};
function httpNodeForm(side, n, upd) {
  n.args = n.args || {};
  const a = n.args;
  const mode = (n.mode || "get").toLowerCase();
  const clear = (key) => { const v = a[key]; if (v === "" || v == null || (typeof v === "object" && !Object.keys(v).length)) delete a[key]; };
  const setArg = (key, v) => { a[key] = v; clear(key); upd("h" + key + n.id); };
  side.append(field("Что сделать", select(Object.entries(HTTP_LABEL), mode, (v) => {
    n.mode = v;
    n.confirm = !HTTP_READS.includes(v);
    if (HTTP_READS.includes(v)) { delete a.json; delete a.form; delete a.body; }
    upd("hmode"); panel();
  }), "Получение ничего не меняет и идёт без вопроса. Всё остальное что-то меняет на той стороне — по умолчанию спросит вас."));
  side.append(field("Адрес", withInserter(inp(a.url || "", (v) => setArg("url", v.trim()), {ph: "https://example.com/api/status"})),
    "Начинается с http:// или https://. Нужные значения вставляйте кнопками."));
  side.append(field("Параметры адреса", kvEditor(a.query || {}, (v) => setArg("query", v), {keyPh: "имя", valPh: "значение или {{поле}}", addLabel: "Добавить параметр"}),
    "Допишутся к адресу как ?имя=значение."));

  if (!HTTP_READS.includes(mode)) {
    const kind = a.json != null && a.json !== "" ? "json" : a.form ? "form" : a.body ? "text" : "none";
    side.append(field("Что отправить", select([["none", "ничего"], ["json", "данные JSON"], ["form", "поля формы"], ["text", "просто текст"]], kind, (v) => {
      delete a.json; delete a.form; delete a.body;
      if (v === "json") a.json = {};
      if (v === "form") a.form = {name: "value"};
      if (v === "text") a.body = "";
      upd("hbodykind"); panel();
    })));
    if (kind === "json") side.append(field("Данные (JSON)", jsonArea(a.json, (v) => { a.json = v; upd("hjson" + n.id); }, 6), "Значения могут содержать {{поле}}."));
    if (kind === "form") side.append(field("Поля формы", kvEditor(a.form, (v) => { a.form = v; upd("hform" + n.id); }, {keyPh: "поле", valPh: "значение или {{поле}}", addLabel: "Добавить поле"})));
    if (kind === "text") side.append(field("Текст", withInserter(area(a.body || "", (v) => { a.body = v; upd("hbody" + n.id); }, 4))));
  }

  side.append(field("Заголовки", kvEditor(a.headers || {}, (v) => setArg("headers", v), {keyPh: "Authorization", valPh: "Bearer ${ENV:API_TOKEN}", addLabel: "Добавить заголовок"}),
    "Секрет пишите как ${ENV:ИМЯ}: значение берётся из переменной окружения при запросе и не сохраняется в комнате."));

  const outs = el("div", {class: "pills static"});
  for (const [k, why] of Object.entries(HTTP_FIELDS)) outs.append(el("span", {class: "pill ghost", title: "{{" + n.id + "." + k + "}}"}, why));
  side.append(el("div", {class: "field"}, el("label", {}, "Что блок оставляет"), outs,
    el("div", {class: "hint"}, "Поля доступны дальше и в условиях стрелок: «" + n.id + ".ok», «" + n.id + ".status». Вставляются кнопками.")));
  side.append(field("Запомнить из ответа", setsEditor(n.sets, (v) => { n.sets = v; upd("sets" + n.id); }),
    "Достаньте нужное: по пути внутри JSON, по шаблону, число, список строк."));
  saveAsField(side, n, upd);
  side.append(check("Спросить вас перед запросом", !!n.confirm, (v) => { n.confirm = v; upd("confirm"); },
    HTTP_READS.includes(mode) ? "Запрос ничего не меняет; вопрос нужен, только если адрес или данные в нём личные." : "Запрос что-то меняет на другой стороне."));
  side.append(el("details", {class: "adv"}, el("summary", {}, "Дополнительно"), el("div", {class: "body"},
    field("Ждать не дольше, секунд", inp(a.timeout ?? "", (v) => setArg("timeout", v === "" ? "" : Math.max(1, Math.min(120, Number(v)))), {type: "number", ph: "30"})),
    field("Повторов при сбое", inp(n.retries || 0, (v) => { n.retries = v || 0; upd("ret"); }, {type: "number"}),
      "Только если сервер не ответил совсем; код 404 или 500 — это ответ, а не сбой."))));
}
