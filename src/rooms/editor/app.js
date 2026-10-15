"use strict";
/* Редактор комнат Argent. Страница ничем не владеет: правду о комнате говорит сервер (/api/validate),
   здесь только удобный способ её нарисовать и поправить. Файл без внешних зависимостей. */

// ── состояние ─────────────────────────────────────────────────────────────────────
const params = new URLSearchParams(location.search);
let TOKEN = params.get("t") || "";
try { if (TOKEN) sessionStorage.setItem("rooms_token", TOKEN); else TOKEN = sessionStorage.getItem("rooms_token") || ""; } catch (e) { /* private mode */ }

const NW = 208, NH = 62, XW = 156, XH = 38;

let META = {tools: [], tool_info: [], tool_groups: {}, not_in_rooms: [], roles: [], channels: [], derived: [], node_types: [], rooms: [],
  leeth: {available: true, modes: []}, vocab: {}};
let EXAMPLES = [];
let room = null;              // рабочая копия (JSON комнаты в виде, который хранит сервер)
let layout = {};              // id узла | "EXIT:имя" -> [x, y]; "_exits" -> имена выходов без стрелок
let source = "new";           // "user" | "starter" | "new"
let sel = null;               // {kind: "node"|"edge"|"exit", id|index}
let inspectMode = null;       // null | "new" — форма создания комнаты в правой панели
let saveAs = null;            // {name} — идёт «сохранить копией»
let connectFrom = null, connectMode = false;
let errors = [], friendly = [];
let lastFlow = "", lastMermaid = "";
let dirty = false;
let view = "graph";
let validateTimer = null;
let helpOpen = false;
let zoom = 1;                 // масштаб схемы
let multi = new Set();        // выделенные группой шаги (кроме основного sel)
let runMap = {};              // id шага -> "active" | "done" | "failed" | "waiting" | "skipped" (подсветка прогона)
let clipPack = null;          // что скопировано внутри редактора

// отмена/возврат: снимки {room, layout} строкой
let baseline = "", lastKey = "", lastAt = 0;
const undoStack = [], redoStack = [];

// ── справочники ───────────────────────────────────────────────────────────────────
const TYPES = {
  tool: {icon: "i-tool", name: "Команда", short: "Запустить команду или инструмент",
    hint: "Без модели: делает ровно то, что написано. Здесь рождается честный вердикт «тесты зелёные»."},
  agent: {icon: "i-agent", name: "Агент", short: "Поручить работу модели",
    hint: "Модель с урезанным набором инструментов и лимитами. Ответ — в заранее оговорённых полях."},
  human_pause: {icon: "i-human_pause", name: "Вопрос вам", short: "Остановиться и спросить",
    hint: "Показывает факты и ждёт ответа; по ответу комната идёт дальше."},
  condition: {icon: "i-condition", name: "Развилка", short: "Выбрать путь по условию",
    hint: "Просто перекрёсток: куда идти, решают условия на стрелках."},
  set: {icon: "i-set", name: "Значение", short: "Записать флаг или текст",
    hint: "Кладёт значения в состояние. Без модели и без затрат."},
  notify: {icon: "i-notify", name: "Уведомление", short: "Сообщить результат",
    hint: "В консоль, всплывающим окном или в Telegram."},
  room: {icon: "i-room", name: "Подкомната", short: "Запустить другую комнату",
    hint: "Готовый рецепт как один шаг, как вызов функции."},
  mcp: {icon: "i-mcp", name: "MCP", short: "Вызвать инструмент MCP-сервера",
    hint: "Внешний сервер с инструментами (Unity, Slack, базы…). Вызывается как обычный шаг, с вопросом и ограничениями."},
  skill: {icon: "i-skill", name: "Скилл", short: "Прочитать инструкции скилла",
    hint: "Кладёт текст установленного скилла в состояние; можно скрыть его разделы."},
  plugin: {icon: "i-plugin", name: "Плагин", short: "Команда плагина или событие",
    hint: "Вызывает команду вашего плагина или запускает событие-хук."},
  leeth: {icon: "i-leeth", name: "Leeth", short: "Поручить быстрой локальной модели",
    hint: "Leeth за миллисекунды отбирает, сортирует и разбирает списки, страницы и формы, без токенов API."},
  foreach: {icon: "i-foreach", name: "Для каждого", short: "Пройти по списку по одному",
    hint: "На каждом заходе берёт следующий элемент списка; когда элементы кончились, отпускает вперёд."},
  wait: {icon: "i-wait", name: "Пауза", short: "Подождать N секунд", hint: "Просто ждёт: пока страница загрузится, пока сервер ответит."},
  loop: {icon: "i-loop", name: "Цикл", short: "for, while, do-while, repeat-until, for-in",
    hint: "Заголовок цикла: каждый заход решает «ещё один круг или хватит». Всегда ограничен."},
  break: {icon: "i-break", name: "Break", short: "Выйти из цикла", hint: "Прерывает цикл: при следующем заходе он закончится."},
  continue: {icon: "i-continue", name: "Continue", short: "К следующему кругу", hint: "Пропускает остаток тела и возвращается в заголовок цикла."},
  power: {icon: "i-power", name: "Питание ПК", short: "Выключить, перезагрузить, сон…",
    hint: "Действие над компьютером. Выключение и перезагрузка идут с задержкой, чтобы их можно было отменить."},
  hint: {icon: "i-hint", name: "Подсказка", short: "Совет модели по ходу работы",
    hint: "Текст, который агент увидит перед работой: не забыть цель, выбраться из зацикливания, учесть то, чего он не знает."},
  forget: {icon: "i-forget", name: "Забыть", short: "Очистить память агентов",
    hint: "Агенты комнаты начинают с чистого листа: память и подсказки стираются, по желанию и поля состояния."},
  assert: {icon: "i-assert", name: "Проверка", short: "Остановиться, если неверно",
    hint: "Подтверждает, что условие выполнено. Иначе шаг падает — дальше стрелка «если шаг упал» или остановка."},
  parallel: {icon: "i-parallel", name: "Параллельно", short: "Запустить несколько веток одновременно",
    hint: "Каждая исходящая стрелка — отдельная ветка. Все ветки идут одновременно и сходятся в блоке «Слияние»."},
  join: {icon: "i-join", name: "Слияние", short: "Собрать ветки обратно в одну",
    hint: "Ждёт ветки своего блока «Параллельно» и объединяет то, что они записали."},
  // Системный блок: есть только в комнате triage, в палитру не попадает.
  spawn_room: {icon: "i-spawn_room", name: "Новая комната", short: "Системный блок: ИИ предлагает комнату",
    hint: "Только для triage: модель предлагает новую комнату как данные, а вы её утверждаете."},
};
const PALETTE_ORDER = ["tool", "agent", "human_pause", "condition", "set", "notify", "room"];
const DERIVED_LABEL = {ok: "прошла ли команда (код 0)", exit_code: "код возврата", output: "весь вывод",
  body: "вывод без служебной строки", tail: "последние строки при провале",
  lines: "строки (список)", count: "сколько строк", first_line: "первая строка", int: "первое число (целое)",
  float: "первое число", bool: "«да/нет» из текста", json: "разобранный JSON"};
const ROLE_LABEL = {Coder: "Coder — правит код", Researcher: "Researcher — читает и ищет", Reviewer: "Reviewer — ревью",
  DocWriter: "DocWriter — тексты и документация", BrowserWorker: "BrowserWorker — работает в браузере", Critic: "Critic — критикует и проверяет"};
const CHANNEL_LABEL = {console: "в консоль", desktop: "всплывающим окном ПК", telegram: "в Telegram"};
const EXIT_PRESETS = [
  ["done", "Готово", "Успешное завершение"],
  ["failed", "Провал", "Не получилось, но это ожидаемый исход"],
  ["escalate", "Человеку", "Передать на разбор (комната triage или вы)"],
];
const TARGET_OPTIONS = [["END", "закончить на этом"], ["TRIAGE", "передать на разбор (triage)"]];

// ── утилиты ───────────────────────────────────────────────────────────────────────
const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return n;
};
const NS = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; };
const icon = (name, cls = "") => {
  const s = svgEl("svg", {class: "ic " + cls}); const u = svgEl("use", {href: "#" + name}); s.append(u); return s;
};
const clone = (o) => JSON.parse(JSON.stringify(o));
const short = (s, n) => { s = String(s ?? ""); return s.length > n ? s.slice(0, n - 1) + "…" : s; };
const typeLabel = (t) => (TYPES[t] || {name: t}).name;
const plural = (n, one, few, many) => { const a = Math.abs(n) % 100, b = a % 10; return (a > 10 && a < 20) ? many : b > 1 && b < 5 ? few : b === 1 ? one : many; };

async function api(path, body) {
  const opt = {headers: {"X-Rooms-Token": TOKEN}};
  if (body !== undefined) { opt.method = "POST"; opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  let data = {};
  try { data = await r.json(); } catch (e) { /* not JSON */ }
  return {status: r.status, data};
}

function say(text, kind, items) {
  const s = $("#status");
  s.className = kind || "";
  const line = el("div", {class: "line"}, kind === "ok" ? icon("i-check", "sm") : kind === "bad" ? icon("i-alert", "sm") : null, el("span", {}, text));
  s.replaceChildren(line);
  if (items && items.length) {
    const ul = el("ul");
    for (const it of items) {
      const li = el("li", {title: it.raw, onclick: () => focusFromError(it.raw)}, it.text);
      if (it.fix) li.append(el("small", {}, it.fix));
      ul.append(li);
    }
    s.append(ul);
  }
}

// ── модель ────────────────────────────────────────────────────────────────────────
function blankRoom(name) {
  return {room: name, description: "", tags: [], budget: {max_iterations: 10, max_tokens: 40000}, entry: "start",
    defaults: {}, next: {done: "END"}, nodes: [{id: "start", type: "agent", role: "Coder", budget: {max_iterations: 6, max_tokens: 20000},
      writes: ["result"], instruction: "Задача: {{task}}\nСделай работу. В конце ответь JSON-объектом: result=<что сделано>."}],
    edges: [{from: "start", to: "EXIT:done"}]};
}
function nodeById(id) { return room.nodes.find((n) => n.id === id); }
function exitNames() {
  const s = new Set(layout._exits || []);
  for (const e of room.edges) if (String(e.to).startsWith("EXIT:")) s.add(e.to.slice(5));
  return [...s].sort();
}
function uniqueId(base) { let i = 1, id = base; while (nodeById(id)) id = base + "_" + (++i); return id; }

function defaultNode(type) {
  const id = uniqueId({tool: "command", agent: "agent", condition: "branch", human_pause: "ask", set: "set_values", notify: "notify", room: "sub_room",
    mcp: "mcp_call", skill: "read_skill", plugin: "plugin_cmd", leeth: "leeth", foreach: "each", wait: "pause", loop: "loop", break: "break_out",
    continue: "next_round", power: "pc_power", hint: "hint", forget: "forget", assert: "check", parallel: "fork", join: "merge_point"}[type] || type);
  const b = {id, type};
  if (type === "mcp") Object.assign(b, {server: Object.keys((typeof CAT !== "undefined" && CAT.mcpServers) || {})[0] || "", tool: "", args: {}, confirm: true});
  if (type === "skill") Object.assign(b, {skill: ((typeof CAT !== "undefined" && CAT.skills) || [])[0] ? CAT.skills[0].name : ""});
  if (type === "plugin") Object.assign(b, {command: "", args: {}, confirm: true});
  if (type === "leeth") Object.assign(b, {mode: "delegate", instruction: "", confirm: true});
  if (type === "foreach") Object.assign(b, {input_from: "", item: "item", max_items: 100});
  if (type === "wait") b.seconds = 5;
  if (type === "loop") Object.assign(b, {mode: "for", start: 0, stop: 5, step: 1, item: "i"});
  if (type === "break" || type === "continue") b.loop = (room.nodes.find((x) => x.type === "loop" || x.type === "foreach") || {}).id || "";
  if (type === "power") Object.assign(b, {mode: "shutdown", seconds: 60});
  if (type === "hint") Object.assign(b, {mode: "next", message: ""});
  if (type === "forget") b.mode = "all";
  if (type === "assert") Object.assign(b, {condition: "state.ok", message: ""});
  if (type === "parallel") b.join = (room.nodes.find((x) => x.type === "join" && !room.nodes.some((p) => p.type === "parallel" && p.join === x.id)) || {}).id || "";
  if (type === "join") Object.assign(b, {merge: "error", wait_for: "all", on_branch_fail: "fail"});
  if (type === "tool") Object.assign(b, {tool: "run_command", args: {command: ""}});
  if (type === "agent") Object.assign(b, {role: "Researcher", tools: ["read_file", "grep_search"],
    budget: {max_iterations: 6, max_tokens: 20000}, writes: ["result"], instruction: ""});
  if (type === "condition") b.condition = "state.ok";
  if (type === "human_pause") Object.assign(b, {prompt: "Продолжать?", into: "answer"});
  if (type === "set") b.values = {flag: "value"};
  if (type === "notify") Object.assign(b, {message: "Готово", channel: "console"});
  if (type === "room") Object.assign(b, {room: (META.rooms.find((r) => r.name !== room.room) || {}).name || "", inputs: {}, outputs: []});
  return b;
}

// какие поля умеет нести узел; всё остальное вычищается при смене типа и перед отправкой
const COMMON = ["disabled", "note"];
const OWN = {
  tool: ["tool", "args", "input_from", "instruction", "idempotent", "confirm", "retries", "sets"],
  agent: ["role", "tools", "budget", "writes", "context", "instruction", "retries", "skills", "mcp", "ignore", "memory", "hint", "sampling"],
  condition: ["condition"],
  human_pause: ["prompt", "into"],
  set: ["values"],
  notify: ["message", "title", "channel", "compose"],
  room: ["room", "inputs", "outputs", "into", "retries"],
  mcp: ["server", "mode", "tool", "args", "input_from", "sets", "confirm", "retries", "ignore", "idempotent"],
  skill: ["skill", "ignore", "sets", "retries"],
  plugin: ["plugin", "mode", "command", "args", "sets", "confirm", "retries", "idempotent"],
  leeth: ["mode", "instruction", "input_from", "args", "sets", "confirm", "retries", "idempotent"],
  foreach: ["input_from", "item", "max_items"],
  wait: ["seconds"],
  loop: ["mode", "start", "stop", "step", "condition", "input_from", "item", "item_value", "max_items"],
  break: ["loop"],
  continue: ["loop"],
  power: ["mode", "seconds", "message", "args", "confirm", "idempotent"],
  hint: ["mode", "message", "compose", "targets"],
  forget: ["mode", "clear"],
  assert: ["condition", "message"],
  parallel: ["join", "workers"],
  join: ["merge", "wait_for", "on_branch_fail"],
};
function cleaned(obj) {
  const r = clone(obj);
  for (const n of r.nodes) {
    const own = new Set(["id", "type", ...COMMON, ...(OWN[n.type] || [])]);
    for (const k of Object.keys(n)) {
      if (!own.has(k)) delete n[k];
      else if (n[k] === "" || n[k] === null || n[k] === undefined || (Array.isArray(n[k]) && !n[k].length && k !== "tools" && k !== "mcp") ||
               (n[k] && typeof n[k] === "object" && !Array.isArray(n[k]) && !Object.keys(n[k]).length && k !== "args")) delete n[k];
    }
    if (n.type === "tool" && n.idempotent === true) delete n.idempotent;
    if (["mcp", "plugin", "leeth", "power"].includes(n.type) && n.idempotent === false) delete n.idempotent;
    for (const k of ["confirm", "disabled", "compose"]) if (n[k] === false) delete n[k];
    if (!n.retries) delete n.retries;
    if (n.args && !Object.keys(n.args).length) delete n.args;
    if (n.type === "mcp" && n.mode === "call") delete n.mode;
    if (n.type === "plugin" && n.mode === "command") delete n.mode;
    if (n.type === "leeth" && n.mode === "delegate") delete n.mode;
    if (n.type === "loop" && n.mode === "for") delete n.mode;
    if (n.type === "hint" && n.mode === "next") delete n.mode;
    if (n.type === "forget" && n.mode === "all") delete n.mode;
    if (n.args) for (const [k, v] of Object.entries(n.args)) if (v === "" || v === false || v === null) delete n.args[k];
    if (n.args && !Object.keys(n.args).length) delete n.args;
  }
  if (r.rules) {
    for (const [k, v] of Object.entries(r.rules)) {
      if (v === false || v === "" || v == null || (Array.isArray(v) && !v.length) || (v && typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length) ||
          (k === "memory" && v === "isolated")) delete r.rules[k];
    }
    if (!Object.keys(r.rules).length) delete r.rules;
  }
  for (const e of r.edges) {
    for (const k of ["if", "while", "max", "id"]) if (e[k] === "" || e[k] == null) delete e[k];
    if (!e.else) delete e.else;
    if (!e.on_error) delete e.on_error;
  }
  if (!(r.tags || []).length) delete r.tags;
  for (const k of ["defaults", "next"]) if (!Object.keys(r[k] || {}).length) delete r[k];
  if (!r.report) delete r.report;
  if (!r.description) r.description = r.room;
  return r;
}

// ── поля состояния: что уже есть к моменту, когда вы пишете условие или подстановку ──
function knownFields() {
  const f = new Map();
  const add = (k, why) => { if (k && !f.has(k)) f.set(k, why); };
  add("task", "задача, с которой запустили комнату");
  for (const k of Object.keys(room.defaults || {})) add(k, "параметр комнаты");
  for (const n of room.nodes) {
    if (["tool", "mcp", "plugin", "leeth", "skill"].includes(n.type)) for (const [k, v] of Object.entries(n.sets || {})) add(k, `шаг «${n.id}»: ${DERIVED_LABEL[v] || v}`);
    if (["tool", "mcp", "plugin", "leeth", "skill", "power", "hint", "assert"].includes(n.type)) add(n.id + ".output", `вывод шага «${n.id}»`);
    if (n.type === "foreach") { add(n.item || "item", `текущий элемент («${n.id}»)`); add(n.id + "_index", `номер элемента («${n.id}»)`); add(n.id + "_count", `сколько элементов («${n.id}»)`); add(n.id + "_done", `список закончился («${n.id}»)`); }
    if (n.type === "loop") {
      add(n.item || (n.mode === "for_in" ? "key" : "i"), `счётчик цикла «${n.id}»`);
      if (n.item_value) add(n.item_value, `значение в цикле «${n.id}»`);
      add(n.id + "_index", `номер круга («${n.id}»)`); add(n.id + "_done", `цикл закончился («${n.id}»)`);
      add(n.id + "_broken", `прерван через break («${n.id}»)`); add(n.id + "_capped", `остановлен лимитом кругов («${n.id}»)`);
    }
    if (n.type === "notify") add(n.id + ".delivered", `дошло ли уведомление «${n.id}»`);
    if (n.type === "join") { add(n.id + "_arrived", `какие ветки дошли до «${n.id}» (список)`); add(n.id + "_failed", `какие ветки не дошли до «${n.id}»`); add(n.id + ".output", `итог слияния «${n.id}»`); }
    add(n.id + "_visits", `сколько раз начинался шаг «${n.id}» (для условия «зациклилось»)`);
    if (n.type === "agent") for (const k of n.writes || []) add(k, `шаг «${n.id}»: пишет агент`);
    if (n.type === "human_pause" && n.into) { add(n.into, `ответ на вопрос «${n.id}»`); add(n.into + "_approved", `был ли ответ «да» («${n.id}»)`); }
    if (n.type === "set") for (const k of Object.keys(n.values || {})) add(k, `шаг «${n.id}»: записанное значение`);
    if (n.type === "room") { for (const k of n.outputs || []) add(k, `вернула подкомната «${n.room}»`); add(n.into || n.id + "_exit", `чем закончилась подкомната «${n.id}»`); }
  }
  if (room.edges.some((e) => e.on_error)) { add("error", "текст ошибки упавшего шага"); add("error_node", "какой шаг упал"); }
  return f;
}
function refreshFieldList() {
  const dl = $("#fieldsList"); dl.replaceChildren();
  if (!room) return;
  for (const k of knownFields().keys()) dl.append(el("option", {value: k}));
}

// ── простыми словами ──────────────────────────────────────────────────────────────
function describeNode(n) {
  const q = (s) => "«" + short(s, 70) + "»";
  if (n.type === "tool") {
    let s = n.tool === "run_command" ? (n.args && n.args.command ? `Запускает команду ${q(n.args.command)}.` : "Запускает команду (она пока не задана).") : `Вызывает инструмент ${n.tool}.`;
    const sets = Object.entries(n.sets || {});
    if (sets.length) s += " Запомнит: " + sets.map(([k, v]) => `${k} — ${DERIVED_LABEL[v] || v}`).join("; ") + ".";
    if (n.confirm) s += " Сначала спросит вас.";
    return s;
  }
  if (n.type === "agent") {
    let s = `Поручает работу модели (${ROLE_LABEL[n.role || "Coder"] || n.role}).`;
    if (n.tools != null) s += n.tools.length ? ` Может пользоваться: ${n.tools.join(", ")}.` : " Без инструментов.";
    if ((n.writes || []).length) s += ` Вернёт: ${n.writes.join(", ")}.`;
    if (n.memory === "shared") s += " Помнит, что делали прошлые шаги.";
    if (Object.keys(n.sampling || {}).length) s += " Модель: " + Object.entries(n.sampling).map(([k, v]) => `${k}=${v}`).join(", ") + ".";
    if ((n.skills || []).length) s += " Скиллы: " + n.skills.join(", ") + ".";
    if (n.hint) s += " Есть своя подсказка.";
    return s;
  }
  if (n.type === "mcp") {
    if ((n.mode || "call") === "list") return `Спрашивает у MCP-сервера «${n.server || "?"}», какие у него инструменты.`;
    let s = `Вызывает инструмент «${n.tool || "?"}» MCP-сервера «${n.server || "?"}».`;
    if (Object.keys(n.args || {}).length) s += " Аргументы: " + Object.entries(n.args).map(([k, v]) => `${k} = ${short(JSON.stringify(v), 30)}`).join(", ") + ".";
    if (Object.keys(n.sets || {}).length) s += " Запомнит: " + Object.entries(n.sets).map(([k, v]) => `${k} — ${DERIVED_LABEL[v] || v}`).join("; ") + ".";
    if ((n.ignore || []).length) s += " Игнорирует: " + n.ignore.join(", ") + ".";
    return s + (n.confirm ? " Сначала спросит вас." : "");
  }
  if (n.type === "skill") return `Читает инструкции скилла «${n.skill || "?"}»` + ((n.ignore || []).length ? `, пропуская: ${n.ignore.join(", ")}` : "") + ".";
  if (n.type === "plugin") return `${(n.mode || "command") === "hook" ? "Запускает событие" : "Выполняет команду плагина"} «${n.command || "?"}».` + (n.confirm ? " Сначала спросит вас." : "");
  if (n.type === "leeth") {
    const m = n.mode || "delegate", q = n.instruction ? ` «${short(n.instruction, 60)}»` : "";
    const base = {delegate: `Поручает Leeth задачу${q}.`, filter: `Leeth отвечает «да/нет» про каждый элемент списка «${n.input_from || "?"}»:${q}.`,
      classify: `Leeth раскладывает «${n.input_from || "?"}» по категориям: ${((n.args || {}).categories || []).join(", ") || "?"}.`,
      scan: `Leeth оценивает открытую страницу (${(n.args || {}).include || "links"}):${q}.`, fill: `Leeth заполняет форму на странице:${q}.`,
      click: `Leeth нажимает то, что нашла оценка страницы («${n.input_from || "?"}»)${(n.args || {}).button ? ", кнопка по шаблону «" + n.args.button + "»" : ""}.`}[m];
    return base + (n.confirm ? " Сначала спросит вас." : "");
  }
  if (n.type === "foreach") return `Идёт по списку «${n.input_from || "?"}», по одному: элемент в «${n.item || "item"}». Когда кончится — «${n.id}_done» станет истиной.`;
  if (n.type === "wait") return `Ждёт ${n.seconds ?? "?"} с.`;
  if (n.type === "loop") {
    const m = n.mode || "for";
    return ({for: `Цикл: ${n.item || "i"} от ${n.start ?? 0} до ${n.stop ?? "?"} с шагом ${n.step ?? 1}.`, times: `Повторяет ${n.stop ?? "?"} раз.`,
      while: `Повторяет, пока ${condWords(n.condition)}.`, do_while: `Делает круг, затем повторяет, пока ${condWords(n.condition)}.`,
      until: `Делает круг и повторяет, пока не ${condWords(n.condition)}.`, for_in: `Идёт по словарю или списку «${n.input_from || "?"}»: ключ в «${n.item || "key"}»${n.item_value ? ", значение в «" + n.item_value + "»" : ""}.`}[m]) +
      ` Не больше ${n.max_items || (META.vocab || {}).loop_default_max || 100} кругов.`;
  }
  if (n.type === "break") return `Прерывает цикл «${n.loop || "?"}»: при следующем заходе он закончится.`;
  if (n.type === "continue") return `Переходит к следующему кругу цикла «${n.loop || "?"}».`;
  if (n.type === "power") {
    const w = {shutdown: "Выключит компьютер", restart: "Перезагрузит компьютер", sleep: "Переведёт компьютер в спящий режим", hibernate: "Отправит компьютер в гибернацию",
      lock: "Заблокирует экран", logout: "Завершит сессию пользователя", cancel: "Отменит запланированное выключение"}[n.mode] || "Действие над ПК";
    return w + ((n.mode === "shutdown" || n.mode === "restart") ? ` через ${n.seconds ?? 60} с (можно отменить)` : "") + "." + (n.confirm ? " Сначала спросит вас." : " Без вопроса.");
  }
  if (n.type === "hint") {
    const m = n.mode || "next";
    if (m === "clear") return "Снимает накопленные подсказки" + ((n.targets || []).length ? " для: " + n.targets.join(", ") : "") + ".";
    return `${m === "all" ? "Подсказка всем агентам до отмены" : "Подсказка следующему агенту"}${(n.targets || []).length ? " (" + n.targets.join(", ") + ")" : ""}: ` +
      (n.compose ? `модель напишет её по инструкции «${short(n.message, 60)}».` : `«${short(n.message, 80)}».`);
  }
  if (n.type === "forget") return ({all: "Агенты забывают память и подсказки", memory: "Агенты забывают, что делали прошлые шаги", hints: "Стирает подсказки"}[n.mode || "all"]) +
    ((n.clear || []).length ? "; удаляет поля: " + n.clear.join(", ") : "") + ".";
  if (n.type === "assert") return `Проверяет, что ${condWords(n.condition)}${n.message ? " («" + short(n.message, 50) + "»)" : ""}; иначе шаг упадёт.`;
  if (n.type === "parallel") return `Запускает ${room.edges.filter((e) => e.from === n.id).length} ветки одновременно${n.workers ? " (не больше " + n.workers + " разом)" : ""}; они сойдутся в «${n.join || "?"}».`;
  if (n.type === "join") return `Собирает ветки: ${({all: "ждёт все", any: "идёт дальше, когда придёт первая"})[n.wait_for || "all"]}; если ветки записали разное в одно поле — ${({error: "ошибка", first: "берёт значение первой ветки", last: "берёт значение последней ветки", list: "кладёт все значения списком"})[n.merge || "error"]}.`;
  if (n.type === "condition") return `Развилка${n.condition ? " (" + n.condition + ")" : ""}: путь выбирают условия на стрелках.`;
  if (n.type === "human_pause") return `Спрашивает вас: ${q(n.prompt || "")}.` + (n.into ? ` Ответ запишет в «${n.into}».` : "");
  if (n.type === "set") return "Записывает: " + Object.entries(n.values || {}).map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join(", ") + ".";
  if (n.type === "notify") return n.compose ? `Сообщает ${CHANNEL_LABEL[n.channel] || n.channel}; текст пишет модель по инструкции ${q(n.message || "")}.`
    : `Сообщает ${CHANNEL_LABEL[n.channel] || n.channel}: ${q(n.message || "")}.`;
  if (n.type === "room") {
    let s = `Запускает комнату «${n.room || "?"}» как один шаг.`;
    if ((n.outputs || []).length) s += ` Заберёт: ${n.outputs.join(", ")}.`;
    return s + ` Чем она закончилась — в поле «${n.into || n.id + "_exit"}».`;
  }
  if (n.type === "spawn_room") return "Системный шаг: модель предлагает новую комнату как данные. Она пройдёт проверку и ваше утверждение, прежде чем появится.";
  return "";
}
function parseCond(expr) {
  const s = String(expr || "").trim();
  let m = s.match(/^state\.(\w+)$/);
  if (m) return {field: m[1], op: "true"};
  m = s.match(/^not\s+state\.(\w+)$/);
  if (m) return {field: m[1], op: "false"};
  m = s.match(/^state\.(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)$/);
  if (m) {
    const opMap = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"};
    let v = m[3].trim();
    const q = v.match(/^"(.*)"$|^'(.*)'$/);
    if (q) v = q[1] ?? q[2];
    else if (!/^-?\d+(\.\d+)?$|^(True|False|None)$/.test(v)) return null;
    return {field: m[1], op: opMap[m[2]], value: v, quoted: !!q};
  }
  return null;
}
const OPS = [["true", "выполнено (истинно)"], ["false", "не выполнено (ложно)"], ["eq", "равно"], ["ne", "не равно"],
  ["gt", "больше"], ["ge", "не меньше"], ["lt", "меньше"], ["le", "не больше"], ["custom", "своё выражение…"]];
const OPSYM = {eq: "==", ne: "!=", gt: ">", ge: ">=", lt: "<", le: "<="};
function buildCond(p) {
  if (p.op === "true") return `state.${p.field}`;
  if (p.op === "false") return `not state.${p.field}`;
  const raw = /^-?\d+(\.\d+)?$|^(True|False|None)$/.test(String(p.value)) && !p.forceString;
  return `state.${p.field} ${OPSYM[p.op]} ${raw ? p.value : JSON.stringify(String(p.value ?? ""))}`;
}
function condWords(expr) {
  const p = parseCond(expr);
  if (!p) return expr;
  if (p.op === "true") return `«${p.field}» выполнено`;
  if (p.op === "false") return `«${p.field}» не выполнено`;
  const w = {eq: "равно", ne: "не равно", gt: "больше", ge: "не меньше", lt: "меньше", le: "не больше"}[p.op];
  return `«${p.field}» ${w} ${p.quoted ? "«" + p.value + "»" : p.value}`;
}
function edgeKind(e) { return e.on_error ? "error" : e.while != null ? "while" : e.if != null ? "if" : e.else ? "else" : "uncond"; }
function edgeLabel(e) {
  const k = edgeKind(e);
  if (k === "if") return "если " + short(condWords(e.if), 26);
  if (k === "while") return "пока " + short(condWords(e.while), 22) + " ≤" + e.max;
  if (k === "else") return "иначе";
  if (k === "error") return "при сбое";
  return "";
}
function edgeWords(e) {
  const k = edgeKind(e);
  if (k === "if") return "если " + condWords(e.if);
  if (k === "while") return `пока ${condWords(e.while)} (не более ${e.max} раз)`;
  if (k === "else") return "иначе";
  if (k === "error") return "если шаг упал";
  return "всегда";
}
function targetWords(to) {
  if (String(to).startsWith("EXIT:")) {
    const name = to.slice(5), nx = (room.next || {})[name];
    return `выход «${name}»` + (nx ? ` (дальше: ${nx === "END" ? "конец" : nx === "TRIAGE" ? "на разбор" : "комната «" + nx + "»"})` : "");
  }
  return `шаг «${to}»`;
}
function bfsOrder() {
  const out = {}, order = [], seen = new Set();
  for (const e of room.edges) (out[e.from] = out[e.from] || []).push(e.to);
  const q = [room.entry || (room.nodes[0] && room.nodes[0].id)];
  while (q.length) {
    const id = q.shift();
    if (!id || seen.has(id) || !nodeById(id)) continue;
    seen.add(id); order.push(id);
    for (const t of out[id] || []) q.push(t);
  }
  for (const n of room.nodes) if (!seen.has(n.id)) order.push(n.id);
  return order;
}

// ── история ───────────────────────────────────────────────────────────────────────
function snap() { return JSON.stringify({room, layout}); }
function resetState(obj, lay, src, isDirty) {
  room = obj; layout = lay || {}; source = src; sel = null; dirty = !!isDirty; errors = []; friendly = [];
  multi.clear(); runMap = {};
  room.tags = room.tags || []; room.defaults = room.defaults || {}; room.next = room.next || {};
  room.budget = room.budget || {max_iterations: 20, max_tokens: 80000};
  room.nodes.forEach((n) => { if (n.type === "tool") n.args = n.args || {}; });
  inspectMode = null; saveAs = null; connectMode = false; connectFrom = null;
  if (!Object.keys(layout).length) autoLayout();
  baseline = snap(); lastKey = ""; undoStack.length = 0; redoStack.length = 0;
  fullRender(); validate();
}
function touch(key, rebuildPanel) {
  if (!room) return;
  const s = snap();
  if (s === baseline) { if (rebuildPanel) panel(); return; }
  const now = Date.now();
  if (!(key && key === lastKey && now - lastAt < 1500)) { undoStack.push(baseline); if (undoStack.length > 100) undoStack.shift(); }
  redoStack.length = 0;
  baseline = s; lastKey = key || ""; lastAt = now;
  dirty = true;
  scheduleValidate(); draw(); chrome(); refreshFieldList();
  if (rebuildPanel) panel();
  if (view === "words") renderWords();
}
function restore(s) {
  const o = JSON.parse(s);
  room = o.room; layout = o.layout;
  if (sel && sel.kind === "node" && !nodeById(sel.id)) sel = null;
  if (sel && sel.kind === "edge" && !room.edges[sel.index]) sel = null;
  baseline = s; lastKey = ""; dirty = true;
  fullRender(); validate();
}
function undo() { if (!undoStack.length) return; redoStack.push(baseline); restore(undoStack.pop()); }
function redo() { if (!redoStack.length) return; undoStack.push(baseline); restore(redoStack.pop()); }

// ── раскладка ─────────────────────────────────────────────────────────────────────
function autoLayout() {
  const depth = {};
  const out = {};
  for (const e of room.edges) (out[e.from] = out[e.from] || []).push(e.to);
  const start = room.entry || (room.nodes[0] && room.nodes[0].id);
  const q = [start]; depth[start] = 0;
  while (q.length) {
    const id = q.shift();
    for (const t of out[id] || []) if (!(t in depth) && !String(t).startsWith("EXIT:") && nodeById(t)) { depth[t] = depth[id] + 1; q.push(t); }
  }
  let maxD = Math.max(0, ...Object.values(depth));
  for (const n of room.nodes) if (!(n.id in depth)) depth[n.id] = ++maxD;
  const cols = {}, keep = layout._exits;
  layout = {};
  for (const n of room.nodes) {
    const d = depth[n.id]; cols[d] = cols[d] || 0;
    layout[n.id] = [40 + cols[d] * (NW + 56), 56 + d * (NH + 70)];
    cols[d]++;
  }
  const lastY = 56 + (maxD + 1) * (NH + 70);
  exitNames().forEach((x, i) => { layout["EXIT:" + x] = [40 + i * (XW + 40), lastY]; });
  if (keep) layout._exits = keep;
}
function ensureLayout() {
  if (room.nodes.some((n) => !layout[n.id])) {
    const have = {...layout};
    autoLayout();
    for (const [k, v] of Object.entries(have)) if (k in layout && room.nodes.some((n) => n.id === k)) layout[k] = v;
  }
  let lastY = 56;
  for (const n of room.nodes) lastY = Math.max(lastY, (layout[n.id] || [0, 0])[1] + NH + 70);
  exitNames().forEach((x, i) => { if (!layout["EXIT:" + x]) layout["EXIT:" + x] = [40 + i * (XW + 40), lastY]; });
}
function freeSpot(x, y) {
  const taken = () => Object.entries(layout).some(([k, v]) => k !== "_exits" && Math.abs(v[0] - x) < NW + 16 && Math.abs(v[1] - y) < NH + 16);
  let guard = 0;
  while (taken() && guard++ < 30) x += NW + 40;
  return [x, y];
}

// ── рисование схемы ───────────────────────────────────────────────────────────────
function errorIds() {
  const ids = new Set();
  for (const e of errors) for (const m of String(e).matchAll(/'([^']+)'/g)) ids.add(m[1]);
  return ids;
}
function pos(id) { return layout[id] || [20, 20]; }
function size(id) { return String(id).startsWith("EXIT:") ? [XW, XH] : [NW, NH]; }
function nodeSub(n) {
  if (n.type === "tool") return n.tool === "run_command" ? (n.args && n.args.command ? n.args.command : "команда не задана") : n.tool;
  if (n.type === "agent") return (n.role || "Coder") + (n.tools != null ? " · " + n.tools.length + " инстр." : "");
  if (n.type === "room") return "→ " + (n.room || "?");
  if (n.type === "set") return Object.keys(n.values || {}).join(", ");
  if (n.type === "notify") return (CHANNEL_LABEL[n.channel] || n.channel) + ": " + (n.compose ? "✎ " : "") + (n.message || "");
  if (n.type === "human_pause") return n.prompt || "";
  if (n.type === "condition") return n.condition || "";
  if (n.type === "spawn_room") return "предлагает комнату";
  if (n.type === "mcp") return (n.server || "?") + ((n.mode || "call") === "list" ? " · список" : " · " + (n.tool || "?"));
  if (n.type === "skill") return n.skill || "?";
  if (n.type === "plugin") return n.command || "?";
  if (n.type === "leeth") return ({delegate: "поручить", filter: "отобрать", classify: "разложить", scan: "оценить страницу", fill: "заполнить форму", click: "нажать"})[n.mode || "delegate"];
  if (n.type === "foreach") return "по " + (n.input_from || "?");
  if (n.type === "wait") return (n.seconds ?? "?") + " с";
  if (n.type === "loop") return ({for: `${n.item || "i"}: ${n.start ?? 0}…${n.stop ?? "?"}`, times: `× ${n.stop ?? "?"}`, while: "пока " + (n.condition || ""),
    do_while: "делать, пока " + (n.condition || ""), until: "до " + (n.condition || ""), for_in: "по " + (n.input_from || "?")})[n.mode || "for"];
  if (n.type === "break" || n.type === "continue") return "цикл " + (n.loop || "?");
  if (n.type === "power") return ({shutdown: "выключить", restart: "перезагрузить", sleep: "сон", hibernate: "гибернация", lock: "блокировка", logout: "выйти", cancel: "отмена"})[n.mode] || "";
  if (n.type === "hint") return ({next: "следующему", all: "всем", clear: "снять"})[n.mode || "next"] + (n.message && n.mode !== "clear" ? ": " + n.message : "");
  if (n.type === "forget") return ({all: "память и подсказки", memory: "память", hints: "подсказки"})[n.mode || "all"];
  if (n.type === "assert") return n.condition || "";
  if (n.type === "parallel") return room.edges.filter((e) => e.from === n.id).length + " веток → " + (n.join || "?") + (n.workers ? " · по " + n.workers : "");
  if (n.type === "join") return ({all: "ждать все", any: "первая готовая"})[n.wait_for || "all"] + " · " + ({error: "конфликт — ошибка", first: "первая", last: "последняя", list: "списком"})[n.merge || "error"];
  return "";
}

function draw() {
  const gN = $("#nodes"), gE = $("#edges");
  gN.replaceChildren(); gE.replaceChildren();
  const empty = $("#emptyHint");
  if (!room) {
    empty.hidden = false;
    empty.replaceChildren(el("div", {}, el("b", {}, "Откройте комнату или создайте новую"), el("div", {}, "Список — вверху слева. Справа — быстрый старт.")));
    return;
  }
  empty.hidden = true;
  ensureLayout();
  const bad = errorIds();
  let maxX = 900, maxY = 560;
  room.edges.forEach((e, i) => {
    if (!layout[e.from] || !layout[e.to]) return;
    const from = pos(e.from), to = pos(e.to);
    const [fw, fh] = size(e.from), [tw, th] = size(e.to);
    const sx = from[0] + fw / 2, sy = from[1] + fh, tx = to[0] + tw / 2, ty = to[1];
    let d;
    if (e.from === e.to) d = `M${from[0] + fw},${from[1] + fh / 2} c 56,-14 56,64 0,0`;
    else if (ty > sy + 20) d = `M${sx},${sy} C${sx},${sy + 50} ${tx},${ty - 50} ${tx},${ty}`;
    else { const side = Math.max(from[0] + fw, to[0] + tw) + 56;
           d = `M${from[0] + fw},${from[1] + fh / 2} C${side},${from[1] + fh / 2} ${side},${to[1] + th / 2} ${to[0] + tw},${to[1] + th / 2}`; }
    const isSel = sel && sel.kind === "edge" && sel.index === i;
    const g = svgEl("g", {class: "edge " + (edgeKind(e) === "error" ? "error " : "") + (isSel ? "sel " : "") + (bad.has(e.from) && bad.has(e.to) ? "err" : "")});
    g.append(svgEl("path", {class: "line", d, "marker-end": isSel ? "url(#arrowSel)" : "url(#arrow)"}));
    const hit = svgEl("path", {class: "hit", d});
    hit.addEventListener("click", (ev) => { ev.stopPropagation(); selectEdge(i); });
    g.append(hit);
    const label = edgeLabel(e);
    if (label) {
      const t = svgEl("text", {x: (sx + tx) / 2, y: (sy + ty) / 2 - 5, "text-anchor": "middle"});
      t.textContent = label; g.append(t);
    }
    gE.append(g);
  });
  const targets = [...room.nodes.map((n) => n.id), ...exitNames().map((x) => "EXIT:" + x)];
  for (const id of targets) {
    const isExit = id.startsWith("EXIT:");
    const n = isExit ? null : nodeById(id);
    const [x, y] = pos(id), [w, h] = size(id);
    maxX = Math.max(maxX, x + w + 260); maxY = Math.max(maxY, y + h + 200);
    const isSel = sel && (sel.kind === "node" || sel.kind === "exit") && sel.id === id;
    const t = isExit ? "exit" : n.type;
    const g = svgEl("g", {class: "node " + (isSel ? "sel " : "") + (multi.has(id) ? "multi " : "") + (!isExit && n.disabled ? "off " : "") +
      (runMap[id] ? "run-" + runMap[id] + " " : "") + (bad.has(id) ? "err" : ""), transform: `translate(${x},${y})`, "data-id": id});
    g.append(svgEl("rect", {class: "card", width: w, height: h, rx: isExit ? 19 : 11, fill: `var(--c-${t}, var(--c-room))`}));
    const ic = icon("i-" + (isExit ? "exit" : n.type)); ic.setAttribute("x", 11); ic.setAttribute("y", isExit ? 10 : 12);
    ic.setAttribute("width", 18); ic.setAttribute("height", 18); ic.style.color = `var(--k-${t}, var(--k-room))`;
    g.append(ic);
    const name = svgEl("text", {class: "name", x: 38, y: isExit ? 24 : 28});
    name.textContent = short(isExit ? id.slice(5) : n.id, isExit ? 15 : 22); g.append(name);
    if (!isExit) {
      const sub = svgEl("text", {class: "sub", x: 38, y: 47});
      sub.textContent = short(nodeSub(n) || typeLabel(n.type), 28); g.append(sub);
      if (id === room.entry) {
        const b = svgEl("g", {class: "badge", transform: "translate(10,-9)"});
        b.append(svgEl("rect", {width: 46, height: 17, rx: 8.5}));
        const bt = svgEl("text", {x: 9, y: 12.5}); bt.textContent = "старт"; b.append(bt); g.append(b);
      }
    }
    if (!isExit) {
      const port = svgEl("circle", {class: "port", cx: w / 2, cy: h, r: 6});
      const pt2 = svgEl("title"); pt2.textContent = "Потяните к другому шагу или выходу, чтобы провести стрелку"; port.append(pt2);
      port.addEventListener("pointerdown", (ev) => startPortDrag(ev, id));
      g.append(port);
      g.append(svgEl("circle", {class: "port in", cx: w / 2, cy: 0, r: 4, "pointer-events": "none"}));
      if (n.note) {
        const nm = svgEl("circle", {class: "notemark", cx: w - 9, cy: 9, r: 4});
        const nt = svgEl("title"); nt.textContent = n.note; nm.append(nt); g.append(nm);
      }
    }
    if (isSel && !isExit && !connectMode) {
      const hnd = svgEl("g", {class: "handle", transform: `translate(${w},${h / 2})`});
      hnd.append(svgEl("circle", {r: 11}));
      const pt = svgEl("text", {x: 0, y: 5, "text-anchor": "middle"}); pt.textContent = "→"; hnd.append(pt);
      const ti = svgEl("title"); ti.textContent = "Куда ведёт этот шаг: нажмите, затем на цель"; hnd.append(ti);
      hnd.addEventListener("pointerdown", (ev) => { ev.stopPropagation(); startConnect(id); });
      g.append(hnd);
    }
    g.addEventListener("pointerdown", (ev) => nodeDown(ev, id, isExit));
    gN.append(g);
  }
  const svg = $("#svg"); svg.setAttribute("width", maxX * zoom); svg.setAttribute("height", maxY * zoom);
  $("#vp").setAttribute("transform", `scale(${zoom})`);
  $("#btnZoomReset").textContent = Math.round(zoom * 100) + "%";
  $("#btnConnect").classList.toggle("on", connectMode);
  const cb = $("#connectBanner");
  cb.hidden = !connectMode;
  if (connectMode) cb.textContent = connectFrom ? `Откуда: «${connectFrom}». Нажмите на шаг или выход, куда вести. Esc — отмена.` : "Нажмите на шаг, из которого ведёт стрелка. Esc — отмена.";
}

let drag = null;
function nodeDown(ev, id, isExit) {
  ev.stopPropagation();
  if (connectMode) {
    if (!connectFrom) { if (isExit) return; connectFrom = id; draw(); return; }
    finishConnect(id);
    return;
  }
  if (!isExit && (ev.shiftKey || ev.ctrlKey || ev.metaKey)) {            // выделение группы
    if (sel && sel.kind === "node" && !multi.size) multi.add(sel.id);
    multi.has(id) ? multi.delete(id) : multi.add(id);
    sel = multi.size ? {kind: "node", id: [...multi][multi.size - 1]} : null;
    inspectMode = null; panel(); draw();
    return;
  }
  if (!(multi.size > 1 && multi.has(id))) multi.clear();
  sel = {kind: isExit ? "exit" : "node", id}; inspectMode = null; panel(); draw();
  const ids = multi.size > 1 && multi.has(id) ? [...multi] : [id];
  drag = {ids, sx: ev.clientX, sy: ev.clientY, from: Object.fromEntries(ids.map((i) => [i, [...pos(i)]])), moved: false};
  window.addEventListener("pointermove", nodeMove);
  window.addEventListener("pointerup", nodeUp, {once: true});
}
function nodeMove(ev) {
  if (!drag) return;
  const dx = (ev.clientX - drag.sx) / zoom, dy = (ev.clientY - drag.sy) / zoom;
  if (!drag.moved && Math.abs(dx) + Math.abs(dy) < 3 / zoom) return;
  drag.moved = true;
  for (const id of drag.ids) layout[id] = [Math.max(0, drag.from[id][0] + dx), Math.max(0, drag.from[id][1] + dy)];
  draw();
}
function nodeUp() {
  window.removeEventListener("pointermove", nodeMove);
  if (drag && drag.moved) touch("drag");
  drag = null;
}
function selectEdge(i) { sel = {kind: "edge", index: i}; inspectMode = null; panel(); draw(); }

function startConnect(from) { connectMode = true; connectFrom = from || null; draw(); }
function guessCondition(fromId) {
  const n = nodeById(fromId);
  if (n && n.type === "tool") { const k = Object.entries(n.sets || {}).find(([, v]) => v === "ok"); if (k) return "state." + k[0]; }
  if (n && n.type === "human_pause" && n.into) return "state." + n.into + "_approved";
  const k = [...knownFields().keys()].find((x) => x !== "task");
  return "state." + (k || "ok");
}
function finishConnect(to) {
  const from = connectFrom;
  connectFrom = null; connectMode = false;
  if (!from) { draw(); return; }
  const e = {from, to};
  const src = nodeById(from);
  const mine = room.edges.filter((x) => x.from === from);
  let loopRule = false;
  if (src && (src.type === "loop" || src.type === "foreach")) {         // заголовок цикла: тело, затем «иначе»
    if (!mine.some((x) => edgeKind(x) === "if")) { e.if = `not state.${from}_done`; loopRule = true; }
    else if (!mine.some((x) => edgeKind(x) === "else")) { e.else = true; loopRule = true; }
  }
  const hasFallback = !loopRule && mine.some((x) => ["uncond", "else"].includes(edgeKind(x)));
  if (hasFallback) { e.if = guessCondition(from); }     // второго «всегда» быть не должно: первый выиграл бы молча
  room.edges.push(e);
  sel = {kind: "edge", index: room.edges.length - 1};
  touch("connect", true);
  if (hasFallback) say("У шага уже есть переход «всегда», поэтому новая стрелка получилась условной. Проверьте условие справа.", "");
}


// ── порты: стрелка перетаскиванием ───────────────────────────────────────────────
let portDrag = null;
function canvasPoint(ev) {
  const w = $("#canvasWrap"), r = w.getBoundingClientRect();
  return [(ev.clientX - r.left + w.scrollLeft) / zoom, (ev.clientY - r.top + w.scrollTop) / zoom];
}
function startPortDrag(ev, id) {
  ev.stopPropagation(); ev.preventDefault();
  portDrag = {from: id, moved: false, sx: ev.clientX, sy: ev.clientY};
  window.addEventListener("pointermove", portMove);
  window.addEventListener("pointerup", portUp, {once: true});
}
function dropTarget(ev) {
  const t = document.elementFromPoint(ev.clientX, ev.clientY);
  const g = t && t.closest ? t.closest("[data-id]") : null;
  return g && g.dataset.id !== portDrag.from ? g : null;
}
function portMove(ev) {
  if (!portDrag) return;
  if (!portDrag.moved && Math.abs(ev.clientX - portDrag.sx) + Math.abs(ev.clientY - portDrag.sy) < 4) return;
  portDrag.moved = true;
  const [fx, fy] = pos(portDrag.from), [fw, fh] = size(portDrag.from);
  const [cx, cy] = canvasPoint(ev);
  const sx = fx + fw / 2, sy = fy + fh;
  const p = $("#tmpEdge");
  p.removeAttribute("hidden");
  p.setAttribute("d", `M${sx},${sy} C${sx},${sy + 50} ${cx},${cy - 50} ${cx},${cy}`);
  document.querySelectorAll(".node.drop").forEach((n) => n.classList.remove("drop"));
  const g = dropTarget(ev);
  if (g) g.classList.add("drop");
}
function portUp(ev) {
  window.removeEventListener("pointermove", portMove);
  $("#tmpEdge").setAttribute("hidden", "");
  document.querySelectorAll(".node.drop").forEach((n) => n.classList.remove("drop"));
  const d = portDrag; portDrag = null;
  if (!d) return;
  portDrag = d;
  const g = dropTarget(ev);
  portDrag = null;
  if (g) { connectFrom = d.from; finishConnect(g.dataset.id); }
  else if (!d.moved) startConnect(d.from);                 // просто нажали на порт: прежний режим «выберите цель»
}

// ── масштаб ──────────────────────────────────────────────────────────────────────
function setZoom(z, cx, cy) {
  const w = $("#canvasWrap");
  z = Math.max(0.3, Math.min(2, Math.round(z * 100) / 100));
  if (z === zoom) return;
  const rect = w.getBoundingClientRect();
  const ax = cx ?? rect.width / 2, ay = cy ?? rect.height / 2;
  const px = (w.scrollLeft + ax) / zoom, py = (w.scrollTop + ay) / zoom;
  zoom = z;
  draw();
  w.scrollLeft = px * zoom - ax; w.scrollTop = py * zoom - ay;
}
function fitView() {
  if (!room) return;
  ensureLayout();
  const ids = [...room.nodes.map((n) => n.id), ...exitNames().map((x) => "EXIT:" + x)].filter((id) => layout[id]);
  if (!ids.length) return;
  let x0 = 1e9, y0 = 1e9, x1 = 0, y1 = 0;
  for (const id of ids) { const [x, y] = pos(id), [w, h] = size(id); x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + w); y1 = Math.max(y1, y + h); }
  const w = $("#canvasWrap");
  const z = Math.max(0.3, Math.min(1.2, (w.clientWidth - 60) / (x1 - x0), (w.clientHeight - 60) / (y1 - y0)));
  zoom = Math.round(z * 100) / 100;
  draw();
  w.scrollLeft = Math.max(0, x0 * zoom - 30); w.scrollTop = Math.max(0, y0 * zoom - 30);
}

// ── фон схемы: сдвиг, рамка выделения, колесо ────────────────────────────────────
let bgDrag = null;
let spaceDown = false;                 // пробел зажат: левая кнопка двигает схему, а не выделяет
window.addEventListener("keydown", (ev) => {
  if (ev.code !== "Space" || isTyping() || !room || view !== "graph") return;
  spaceDown = true; $("#canvasWrap").classList.add("spacepan"); ev.preventDefault();
});
window.addEventListener("keyup", (ev) => {
  if (ev.code === "Space") { spaceDown = false; $("#canvasWrap").classList.remove("spacepan"); }
});
window.addEventListener("blur", () => { spaceDown = false; $("#canvasWrap").classList.remove("spacepan"); });
$("#canvasWrap").addEventListener("contextmenu", (ev) => ev.preventDefault());   // правая кнопка — тоже сдвиг схемы
// Левая кнопка на пустом месте — рамка выделения (Shift/Ctrl добавляют к выделенному).
// Двигать схему: колесо, пробел + левая, средняя или правая кнопка, Alt + левая.
$("#canvasWrap").addEventListener("pointerdown", (ev) => {
  if (connectMode || ev.button > 2) return;
  const w = $("#canvasWrap");
  const pan = ev.button !== 0 || spaceDown || ev.altKey;
  if (pan) {
    if (ev.button !== 0) ev.preventDefault();
    bgDrag = {mode: "pan", sx: ev.clientX, sy: ev.clientY, sl: w.scrollLeft, st: w.scrollTop, moved: false};
  } else {
    const [x, y] = canvasPoint(ev);
    const additive = ev.shiftKey || ev.ctrlKey || ev.metaKey;
    bgDrag = {mode: "rubber", x, y, x2: x, y2: y, sx: ev.clientX, sy: ev.clientY, moved: false, additive,
      base: additive ? new Set(multi.size ? multi : (sel && sel.kind === "node" ? [sel.id] : [])) : new Set()};
  }
  window.addEventListener("pointermove", bgMove);
  window.addEventListener("pointerup", bgUp, {once: true});
});
function bgMove(ev) {
  if (!bgDrag) return;
  const w = $("#canvasWrap");
  if (bgDrag.mode === "pan") {
    const dx = ev.clientX - bgDrag.sx, dy = ev.clientY - bgDrag.sy;
    if (!bgDrag.moved && Math.abs(dx) + Math.abs(dy) < 4) return;
    bgDrag.moved = true; w.classList.add("panning");
    w.scrollLeft = bgDrag.sl - dx; w.scrollTop = bgDrag.st - dy;
  } else {
    if (!bgDrag.moved && Math.abs(ev.clientX - bgDrag.sx) + Math.abs(ev.clientY - bgDrag.sy) < 5) return;   // это ещё клик
    const [x, y] = canvasPoint(ev);
    const r = $("#rubber");
    if (!bgDrag.moved) {
      r.removeAttribute("hidden");
      bgDrag.moved = true;
    }
    bgDrag.x2 = x; bgDrag.y2 = y;
    r.setAttribute("x", Math.min(bgDrag.x, x) * zoom); r.setAttribute("y", Math.min(bgDrag.y, y) * zoom);
    r.setAttribute("width", Math.abs(x - bgDrag.x) * zoom); r.setAttribute("height", Math.abs(y - bgDrag.y) * zoom);
  }
}
function bgUp() {
  window.removeEventListener("pointermove", bgMove);
  const d = bgDrag; bgDrag = null;
  $("#canvasWrap").classList.remove("panning");
  $("#rubber").setAttribute("hidden", "");
  if (!d) return;
  if (d.mode === "rubber" && d.moved && room) {
    const x0 = Math.min(d.x, d.x2), x1 = Math.max(d.x, d.x2), y0 = Math.min(d.y, d.y2), y1 = Math.max(d.y, d.y2);
    multi = new Set([...d.base, ...room.nodes.filter((n) => { const [x, y] = pos(n.id), [w, h] = size(n.id); return x < x1 && x + w > x0 && y < y1 && y + h > y0; }).map((n) => n.id)]);
    sel = multi.size === 1 ? {kind: "node", id: [...multi][0]} : null;
    if (multi.size === 1) multi.clear();
    inspectMode = null; panel(); draw();
    return;
  }
  if (!d.moved && (sel || inspectMode || multi.size)) { sel = null; inspectMode = null; multi.clear(); panel(); draw(); }
}
$("#canvasWrap").addEventListener("wheel", (ev) => {
  if (!(ev.ctrlKey || ev.metaKey)) return;
  ev.preventDefault();
  const r = $("#canvasWrap").getBoundingClientRect();
  setZoom(zoom * (ev.deltaY < 0 ? 1.1 : 1 / 1.1), ev.clientX - r.left, ev.clientY - r.top);
}, {passive: false});

// ── копировать / вставить / вырезать шаги ───────────────────────────────────────
const isTyping = () => /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
window.addEventListener("copy", (ev) => {
  if (isTyping() || !room || view !== "graph") return;
  const ids = groupIds();
  if (!ids.length) return;
  ev.preventDefault();
  clipPack = packNodes(ids);
  ev.clipboardData.setData("text/plain", JSON.stringify({argent_room_nodes: clipPack}));
  say(`Скопировано шагов: ${ids.length}. Ctrl+V вставит их, в том числе в другую комнату.`, "");
});
window.addEventListener("cut", (ev) => {
  if (isTyping() || !room || view !== "graph") return;
  const ids = groupIds();
  if (!ids.length) return;
  ev.preventDefault();
  clipPack = packNodes(ids);
  ev.clipboardData.setData("text/plain", JSON.stringify({argent_room_nodes: clipPack}));
  removeSelected();
});
window.addEventListener("paste", (ev) => {
  if (isTyping() || !room || view !== "graph") return;
  let pack = null;
  try { const o = JSON.parse(ev.clipboardData.getData("text/plain") || "null"); pack = o && o.argent_room_nodes; } catch (e) { /* не наши данные */ }
  pack = pack || clipPack;
  if (!pack) return;
  ev.preventDefault();
  if (pasteNodes(pack)) say(`Вставлено шагов: ${pack.nodes.length}. Соедините их стрелками.`, "");
});

// ── панель нескольких шагов ──────────────────────────────────────────────────────
function multiPanel(side) {
  const ids = [...multi].filter((id) => nodeById(id));
  side.append(ihead("room", `Выбрано шагов: ${ids.length}`, "Shift или Ctrl + щелчок — добавить/убрать; рамка по фону — выделить ещё"));
  side.append(el("p", {class: "says"}, ids.join(", ")));
  side.append(el("div", {class: "footer"},
    el("button", {class: "btn sm", onclick: () => { clipPack = packNodes(ids); say("Скопировано. Ctrl+V вставит.", ""); }}, icon("i-copy", "sm"), "Копировать"),
    el("button", {class: "btn sm", onclick: duplicateSelected}, icon("i-copy", "sm"), "Дублировать"),
    el("button", {class: "btn sm", onclick: () => { const off = !ids.every((id) => nodeById(id).disabled); ids.forEach((id) => { nodeById(id).disabled = off; }); touch("disgrp", true); }},
      ids.every((id) => nodeById(id).disabled) ? "Включить все" : "Отключить все"),
    el("button", {class: "btn sm danger", onclick: removeSelected}, icon("i-trash", "sm"), "Удалить")));
}

// ── добавление блоков ─────────────────────────────────────────────────────────────
function addNode(type, preset) {
  if (!room) { say("Сначала откройте или создайте комнату.", "bad"); return; }
  ensureLayout();
  const n = defaultNode(type);
  if (preset) {
    const keep = n.id;
    Object.assign(n, clone(preset));
    n.id = keep;
    if (n.loop === undefined && (type === "break" || type === "continue")) n.loop = "";
  }
  const S = sel && sel.kind === "node" ? nodeById(sel.id) : null;
  room.nodes.push(n);
  let note = "";
  if (S) {
    const [sx, sy] = pos(S.id);
    layout[n.id] = freeSpot(sx, sy + NH + 70);
    const uncond = room.edges.find((e) => e.from === S.id && edgeKind(e) === "uncond");
    if (uncond) {                       // вставка в цепочку: S → n → прежняя цель
      room.edges.push({from: n.id, to: uncond.to});
      uncond.to = n.id;
    } else if (!room.edges.some((e) => e.from === S.id && edgeKind(e) === "else")) {
      room.edges.push({from: S.id, to: n.id});
    } else note = "У выбранного шага уже есть переход «иначе», соедините новый шаг вручную.";
  } else {
    let right = 0;
    for (const m of room.nodes) if (m !== n) right = Math.max(right, (layout[m.id] || [0, 0])[0]);
    layout[n.id] = [right + NW + 56, 56];
  }
  sel = {kind: "node", id: n.id}; inspectMode = null;
  touch("add", true);
  revealNode(n.id);
  if (note) say(note, "");
}
function addExit(name) {
  if (!room) return;
  name = String(name || "").trim();
  if (!/^[A-Za-z0-9_\-]+$/.test(name)) { say("Имя выхода — латиница, цифры, «_» и «-».", "bad"); return; }
  ensureLayout();
  layout._exits = [...new Set([...(layout._exits || []), name])];
  const S = sel && sel.kind === "node" ? nodeById(sel.id) : null;
  if (!layout["EXIT:" + name]) {
    if (S) { const [sx, sy] = pos(S.id); layout["EXIT:" + name] = freeSpot(sx, sy + NH + 70); }
    else layout["EXIT:" + name] = freeSpot(40, 56 + room.nodes.length * (NH + 70));
  }
  if (S && !room.edges.some((e) => e.from === S.id && ["uncond", "else"].includes(edgeKind(e)))) {
    room.edges.push({from: S.id, to: "EXIT:" + name});
  }
  sel = {kind: "exit", id: "EXIT:" + name}; inspectMode = null;
  touch("exit", true);
  revealNode("EXIT:" + name);
}
function revealNode(id) {
  const w = $("#canvasWrap"), [x, y] = pos(id), [bw, bh] = size(id);
  if (x < w.scrollLeft || x + bw > w.scrollLeft + w.clientWidth) w.scrollLeft = Math.max(0, x - 80);
  if (y < w.scrollTop || y + bh > w.scrollTop + w.clientHeight) w.scrollTop = Math.max(0, y - 80);
}
function renameNode(oldId, newId) {
  newId = newId.trim();
  if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(newId) || newId === oldId || nodeById(newId)) return false;
  const n = nodeById(oldId); n.id = newId;
  for (const e of room.edges) { if (e.from === oldId) e.from = newId; if (e.to === oldId) e.to = newId; }
  if (room.entry === oldId) room.entry = newId;
  if (layout[oldId]) { layout[newId] = layout[oldId]; delete layout[oldId]; }
  sel = {kind: "node", id: newId};
  return true;
}
function renameExit(oldName, newName) {
  newName = newName.trim();
  if (!/^[A-Za-z0-9_\-]+$/.test(newName) || newName === oldName || exitNames().includes(newName)) return false;
  for (const e of room.edges) if (e.to === "EXIT:" + oldName) e.to = "EXIT:" + newName;
  if (room.next && oldName in room.next) { room.next[newName] = room.next[oldName]; delete room.next[oldName]; }
  layout._exits = (layout._exits || []).map((x) => x === oldName ? newName : x);
  if (layout["EXIT:" + oldName]) { layout["EXIT:" + newName] = layout["EXIT:" + oldName]; delete layout["EXIT:" + oldName]; }
  sel = {kind: "exit", id: "EXIT:" + newName};
  return true;
}
function groupIds() {
  if (multi.size > 1) return [...multi];
  return sel && sel.kind === "node" ? [sel.id] : [];
}
function removeSelected() {
  if (!room) return;
  if (multi.size > 1) {
    const ids = new Set(multi);
    if (room.nodes.length - ids.size < 1) { say("Нельзя удалить все шаги.", "bad"); return; }
    room.nodes = room.nodes.filter((n) => !ids.has(n.id));
    room.edges = room.edges.filter((e) => !ids.has(e.from) && !ids.has(e.to));
    for (const id of ids) delete layout[id];
    if (ids.has(room.entry)) room.entry = room.nodes[0].id;
    multi.clear(); sel = null; touch("remove", true);
    return;
  }
  if (!sel) return;
  if (sel.kind === "edge") room.edges.splice(sel.index, 1);
  else if (sel.kind === "node") {
    if (room.nodes.length <= 1) { say("Нельзя удалить последний шаг.", "bad"); return; }
    room.nodes = room.nodes.filter((n) => n.id !== sel.id);
    room.edges = room.edges.filter((e) => e.from !== sel.id && e.to !== sel.id);
    delete layout[sel.id];
    if (room.entry === sel.id) room.entry = room.nodes[0].id;
  } else if (sel.kind === "exit") {
    const name = sel.id.slice(5);
    room.edges = room.edges.filter((e) => e.to !== sel.id);
    layout._exits = (layout._exits || []).filter((x) => x !== name); delete layout[sel.id];
    if (room.next) delete room.next[name];
  }
  sel = null; touch("remove", true);
}
function packNodes(ids) {
  const set = new Set(ids);
  return {nodes: clone(room.nodes.filter((n) => set.has(n.id))),
    edges: clone(room.edges.filter((e) => set.has(e.from) && (set.has(e.to) || String(e.to).startsWith("EXIT:")))),
    layout: Object.fromEntries(ids.map((i) => [i, pos(i)]))};
}
function pasteNodes(pack, offset) {
  if (!room || !pack || !Array.isArray(pack.nodes) || !pack.nodes.length) return false;
  ensureLayout();
  const map = {};
  for (const n of pack.nodes) {
    const c = clone(n);
    c.id = uniqueId(String(n.id).replace(/_\d+$/, "") + "_copy");
    map[n.id] = c.id;
    room.nodes.push(c);
  }
  for (const c of room.nodes) {
    if (Object.values(map).includes(c.id)) {
      if (c.loop && map[c.loop]) c.loop = map[c.loop];
      if (c.targets) c.targets = c.targets.map((t) => map[t] || t);
    }
  }
  for (const e of pack.edges || []) {
    const ne = clone(e);
    ne.from = map[e.from];
    ne.to = map[e.to] || e.to;
    if (ne.id) ne.id = ne.id + "_c" + Math.floor(Math.random() * 900 + 100);
    room.edges.push(ne);
  }
  const d = offset || 48;
  for (const [old, nid] of Object.entries(map)) { const p = (pack.layout || {})[old] || [40, 40]; layout[nid] = [p[0] + d, p[1] + d]; }
  multi = new Set(Object.values(map));
  sel = {kind: "node", id: [...multi][0]};
  if (multi.size === 1) multi.clear();
  touch("paste", true);
  return true;
}
function duplicateSelected() {
  if (!room) return;
  if (multi.size > 1) { pasteNodes(packNodes([...multi]), 56); return; }
  if (!sel || sel.kind !== "node") return;
  const src = nodeById(sel.id);
  const c = clone(src); c.id = uniqueId(src.id + "_copy");
  room.nodes.push(c);
  const [x, y] = pos(src.id);
  layout[c.id] = freeSpot(x + NW + 40, y);
  sel = {kind: "node", id: c.id}; touch("dup", true); revealNode(c.id);
}
function moveEdge(i, dir) {
  const from = room.edges[i].from;
  let j = i + dir;
  while (j >= 0 && j < room.edges.length && room.edges[j].from !== from) j += dir;
  if (j < 0 || j >= room.edges.length) return i;
  [room.edges[i], room.edges[j]] = [room.edges[j], room.edges[i]];
  return j;
}

// ── элементы формы ────────────────────────────────────────────────────────────────
function field(label, control, hintText) {
  const f = el("div", {class: "field"});
  if (label) f.append(el("label", {}, label));
  f.append(control);
  if (hintText) f.append(el("div", {class: "hint"}, hintText));
  return f;
}
function inp(value, onChange, opts = {}) {
  const i = el("input", {type: opts.type || "text", value: value ?? "", placeholder: opts.ph || "", list: opts.list || null});
  if (opts.type === "number") i.setAttribute("min", opts.min ?? 0);
  i.addEventListener("input", () => onChange(opts.type === "number" ? (i.value === "" ? null : Number(i.value)) : i.value));
  return i;
}
function area(value, onChange, rows, mono) {
  const t = el("textarea", {rows: rows || 3, spellcheck: "false", class: mono ? "mono" : ""}); t.value = value ?? "";
  t.addEventListener("input", () => onChange(t.value));
  return t;
}
function jsonArea(value, onChange, rows) {
  const note = el("div", {class: "hint"});
  const t = area(JSON.stringify(value ?? {}, null, 1), () => {}, rows || 4, true);
  t.addEventListener("input", () => {
    try { onChange(JSON.parse(t.value || "{}")); note.textContent = ""; }
    catch (e) { note.textContent = "JSON не разобран: " + e.message; }
  });
  return el("div", {}, t, note);
}
function select(options, value, onChange) {
  const s = el("select");
  let found = false;
  for (const o of options) {
    const [v, lbl] = Array.isArray(o) ? o : [o, o];
    const op = el("option", {value: v}, lbl);
    if (v === value) { op.selected = true; found = true; }
    s.append(op);
  }
  if (!found && value != null && value !== "") { const op = el("option", {value}, value + " (нет в списке)"); op.selected = true; s.prepend(op); }
  s.addEventListener("change", () => onChange(s.value));
  return s;
}
function check(label, value, onChange, note) {
  const c = el("input", {type: "checkbox"}); c.checked = !!value;
  c.addEventListener("change", () => onChange(c.checked));
  return el("label", {class: "check"}, c, el("span", {}, label, note ? el("small", {}, note) : null));
}
function listInput(arr, onChange, ph) {
  return inp((arr || []).join(", "), (v) => onChange(v.split(",").map((s) => s.trim()).filter(Boolean)), {ph});
}
function parseScalar(v) {
  if (v === "true") return true;
  if (v === "false") return false;
  if (/^-?\d+(\.\d+)?$/.test(v)) return Number(v);
  return v;
}
// список «ключ — значение» вместо JSON
function kvEditor(obj, onChange, o = {}) {
  const prim = Object.values(obj || {}).every((v) => ["string", "number", "boolean"].includes(typeof v) || v == null);
  if (!prim) return jsonArea(obj, onChange, 4);
  const box = el("div", {class: "kv"});
  const rows = Object.entries(obj || {}).map(([k, v]) => [k, v == null ? "" : String(v)]);
  const commit = () => {
    const out = {};
    for (const [k, v] of rows) { const key = k.trim(); if (key) out[key] = o.strings ? v : parseScalar(v); }
    onChange(out);
  };
  const render = () => {
    box.replaceChildren();
    rows.forEach((r, i) => {
      const keyCtl = o.keyChoices
        ? select(o.keyChoices(r[0]), r[0], (v) => { r[0] = v; commit(); })
        : inp(r[0], (v) => { r[0] = v; commit(); }, {ph: o.keyPh || "имя поля", list: o.keyList || null});
      let valCtl;
      if (o.valueChoices) valCtl = select(o.valueChoices, r[1], (v) => { r[1] = v; commit(); });
      else valCtl = inp(r[1], (v) => { r[1] = v; commit(); }, {ph: o.valPh || "значение"});
      box.append(el("div", {class: "kvrow"}, keyCtl, valCtl,
        el("button", {class: "btn icon ghost sm", title: "Убрать", "aria-label": "Убрать строку", onclick: () => { rows.splice(i, 1); commit(); render(); }}, icon("i-close", "sm"))));
    });
    box.append(el("div", {}, el("button", {class: "btn sm", onclick: () => {
      const free = o.keyChoices ? ((o.keyChoices("") || []).map((x) => Array.isArray(x) ? x[0] : x).find((x) => !rows.some((r) => r[0] === x)) || "") : "";
      rows.push([free, o.valueChoices ? o.valueChoices[0][0] : ""]); commit(); render();
    }}, icon("i-plus", "sm"), o.addLabel || "Добавить")));
  };
  render();
  return box;
}
// вставка {{поля}} в текст одним выбором
function withInserter(control) {
  const names = [...knownFields().entries()];
  const s = el("select", {class: "ins", "aria-label": "Вставить поле"}, el("option", {value: ""}, "Вставить {{поле}}…"));
  for (const [k, why] of names) s.append(el("option", {value: k, title: why}, k + " — " + why));
  s.addEventListener("change", () => {
    if (!s.value) return;
    const t = control, ins = "{{" + s.value + "}}";
    const a = t.selectionStart ?? t.value.length, b = t.selectionEnd ?? a;
    t.value = t.value.slice(0, a) + ins + t.value.slice(b);
    t.focus(); t.setSelectionRange(a + ins.length, a + ins.length);
    s.value = ""; t.dispatchEvent(new Event("input"));
  });
  return el("div", {}, control, s);
}
// условие: поле + оператор + значение, без знания синтаксиса
function condBuilder(value, onChange) {
  const wrap = el("div", {class: "cond"});
  let st = parseCond(value);
  if (!st) st = {field: "", op: "custom", value: ""};
  const render = () => {
    wrap.replaceChildren();
    const current = () => st.op === "custom" ? st.text : buildCond(st);
    const apply = () => {
      if (st.op !== "custom" && !st.field) { prev.querySelector("code").textContent = "выберите поле"; return; }   // не отдаём недописанное
      const c = current(); onChange(c); prev.querySelector("code").textContent = c || "—";
    };
    const opSel = select(OPS, st.op, (v) => {
      if (v === "custom") { st = {op: "custom", text: buildCond({...st, op: st.op === "custom" ? "true" : st.op}) || value || ""}; }
      else if (st.op === "custom") { const p = parseCond(st.text); st = p ? {...p, op: v} : {field: "", op: v, value: ""}; }
      else st = {...st, op: v};
      render(); apply();
    });
    const prev = el("div", {class: "preview"}, "Получится: ", el("code", {}, ""));
    if (st.op === "custom") {
      st.text = st.text ?? value ?? "";
      wrap.append(inp(st.text, (v) => { st.text = v; apply(); }, {ph: 'например: state.ok and not state.dirty', list: null}), opSel);
    } else {
      const fieldIn = inp(st.field, (v) => { st.field = v.trim(); apply(); }, {ph: "поле, например tests_passed", list: "fieldsList"});
      wrap.append(el("div", {class: "row"}, fieldIn, opSel));
      if (!["true", "false"].includes(st.op)) {
        wrap.append(inp(st.value, (v) => { st.value = v; apply(); }, {ph: "значение: ok, 0, True…"}));
      }
    }
    wrap.append(prev);
    prev.querySelector("code").textContent = st.op !== "custom" && !st.field ? "выберите поле" : (current() || "—");
  };
  render();
  return wrap;
}
function helpBlock(h, title) {
  if (!h) return el("span");
  const d = el("details", {class: "help"});
  if (helpOpen) d.setAttribute("open", "");
  d.addEventListener("toggle", () => { helpOpen = d.open; });
  const body = el("div", {class: "body"}, el("p", {}, h.what));
  if (h.tips && h.tips.length) body.append(el("ul", {}, h.tips.map((t) => el("li", {}, t))));
  body.append(el("button", {class: "more", onclick: () => openGuide(h.sec || "g-edges")}, "Подробнее в гайде"));
  d.append(el("summary", {}, title || "Что это"), body);
  return d;
}
const HELP = {
  tool: {what: "Выполняет инструмент по точному описанию, без модели. Чаще всего это команда оболочки. Результат процесса превращается в поля состояния через «Запомнить из результата».",
    tips: ["Нужен вердикт «зелёное/красное»? Запомните «tests_passed» ← «прошла ли команда»: решает код возврата, а не слова модели.",
      "Шаг что-то меняет (коммит, пуш, деплой)? Включите «Спросить вас перед запуском».",
      "Не подставляйте в команду {{task}} и поля от агентов — это инъекция."], sec: "g-nodes"},
  agent: {what: "Рабочий-модель с урезанным набором инструментов и лимитами. Делает то, что описано в «Что сделать», и отвечает JSON-объектом с полями из «Что вернуть». Куда идти дальше, решает не он, а стрелки.",
    tips: ["Читающему агенту оставьте только read_file, grep_search… — тогда он ничего не изменит.",
      "Не давайте агенту писать «вердикты» вроде tests_passed — их выносит процесс.",
      "В инструкции укажите ответ: «Ответь JSON-объектом: diagnosis=…»."], sec: "g-nodes"},
  condition: {what: "Перекрёсток: сам ничего не делает. Куда идти, решают условия на исходящих стрелках («если» и «иначе»). Поле «Условие» — подпись для читающего.",
    tips: ["Часто развилка не нужна: условие можно повесить на стрелку прямо из команды или агента."], sec: "g-nodes"},
  human_pause: {what: "Останавливает комнату и спрашивает вас. Ответ записывается в поле, а «<поле>_approved» говорит, было ли это «да».",
    tips: ["Вставьте в вопрос факты ({{files}}…), по которым вы решаете.", "Ветвитесь по «<поле>_approved»: «если» → продолжить, «иначе» → выход."], sec: "g-nodes"},
  set: {what: "Кладёт значения в состояние без модели и без затрат: флаги, счётчики, готовые строки.", tips: ["Строки поддерживают {{поля}}."], sec: "g-nodes"},
  notify: {what: "Сообщает результат: в консоль, всплывающим окном или в Telegram. Доставка «по возможности» — прогон из-за неё не падает.",
    tips: ["Ставьте в конец длинной комнаты; в сообщение подставляйте итоговые поля."], sec: "g-nodes"},
  room: {what: "Выполняет другую комнату как один шаг (как вызов функции). Чем она закончилась, попадает в поле «<шаг>_exit» — по нему и ветвитесь.",
    tips: ["Саму себя и круг A→B→A вызвать нельзя; глубина до 3."], sec: "g-nodes"},
  mcp: {what: "Вызывает один инструмент MCP-сервера, как это сделал бы агент, но по точному описанию и без модели. Результат превращается в поля через «Запомнить из результата».",
    tips: ["Внешний сервер может менять мир: оставьте «Спросить вас», если шаг не только читает.", "«Игнорировать» запрещает вызов по-настоящему: mcp:unity.delete_* не пройдёт даже у агента."], sec: "g-nodes"},
  skill: {what: "Кладёт инструкции установленного скилла в «<шаг>.output». Ненужные разделы можно скрыть.",
    tips: ["Чтобы агент руководствовался скиллом, проще указать его в самом агенте («Скиллы»)."], sec: "g-nodes"},
  plugin: {what: "Команда вашего плагина (command_*) или событие-хук. Печатает что-то на экран — это попадёт в результат.", tips: ["Плагин — это код: по умолчанию шаг спрашивает вас."], sec: "g-nodes"},
  leeth: {what: "Быстрая локальная модель Leeth: отбирает и раскладывает списки, оценивает страницы, заполняет формы. Миллисекунды на элемент и ни одного токена API.",
    tips: ["Для постов: «Оценить страницу» (посты) → запомнить как JSON → «Нажать найденное» с шаблоном кнопки.", "Дайте 2–3 примера «подходит / не подходит» — точность заметно растёт."], sec: "g-leeth"},
  foreach: {what: "Берёт по одному элементу списка на каждом заходе. Стрелка «если не <шаг>_done» ведёт в тело, «иначе» — дальше.", tips: ["Конец тела соединяется обратно с этим блоком."], sec: "g-loops"},
  wait: {what: "Пауза на заданное время.", tips: ["Не злоупотребляйте: время считается в лимитах комнаты."], sec: "g-nodes"},
  loop: {what: "Заголовок цикла. Режимы: for (счётчик), N раз, while, do-while, repeat-until, for-in. Всегда ограничен «кругами».",
    tips: ["Тело — блоки после стрелки «если не <цикл>_done»; последний блок тела ведёт обратно в цикл.", "Break и Continue — отдельные блоки, они обязаны вести обратно в цикл."], sec: "g-loops"},
  break: {what: "Прерывает цикл. Со стрелкой обратно в цикл — иначе он ничего не остановит.", tips: ["Ставьте его на ветку «нашли — хватит»."], sec: "g-loops"},
  continue: {what: "Следующий круг: пропускает остаток тела.", tips: ["Это просто стрелка в заголовок — блок делает её понятной на схеме."], sec: "g-loops"},
  power: {what: "Действует на ваш компьютер: выключить, перезагрузить, сон, гибернация, блокировка, выход. Выключение и перезагрузка идут с задержкой.",
    tips: ["Пока вы спите: поставьте перед блоком уведомление в Telegram и оставьте задержку.", "Отменить: блок «Отменить выключение» или команда shutdown /a."], sec: "g-pc"},
  hint: {what: "Короткий совет модели. Агент увидит его перед работой, в блоке HINTS.",
    tips: ["Зациклился? На стрелке условие «state.<шаг>_visits >= 3» ведёт сюда.", "«Текст пишет модель» — подсказка по ситуации."], sec: "g-memory"},
  forget: {what: "Очищает память агентов комнаты, подсказки и (по желанию) поля.", tips: ["Нужна общая память: включите её в правилах комнаты."], sec: "g-memory"},
  parallel: {what: "Запускает все исходящие стрелки одновременно: каждая — своя ветка с копией состояния.",
    tips: ["Пишите в разные поля: одно и то же поле из двух веток — конфликт.", "Вопросов человеку внутри ветки быть не может."], sec: "g-parallel"},
  join: {what: "Точка сбора: ждёт ветки и объединяет записанное в общее состояние.",
    tips: ["«Первая готовая» остановит остальные ветки.", "Какие ветки пришли — в поле «<слияние>_arrived»."], sec: "g-parallel"},
  assert: {what: "Проверка условия. Неверно — шаг падает; дальше стрелка «если шаг упал» или остановка.", tips: ["Удобно ставить после шагов, от которых зависит всё остальное."], sec: "g-nodes"},
  edge: {
    uncond: {what: "«Всегда»: переход без условия — запасной путь, если не сработал ни один «если»/«пока».", tips: ["У шага с одним выходом ставьте именно его."]},
    if: {what: "Идёт по этой стрелке, если условие выполнено. Проверяются по порядку сверху вниз.", tips: ["Условие собирается из поля, оператора и значения."]},
    else: {what: "«Иначе»: запасной путь, если ни один «если»/«пока» не подошёл.", tips: ["Не забывайте «иначе» на цикле — иначе после исчерпания лимита будет выход stuck."]},
    while: {what: "Цикл: идёт по стрелке, пока условие выполнено, но не больше N раз. Счётчик растёт с каждым проходом; исчерпан — стрелка пропускается.",
      tips: ["Шаблон: «если ок → дальше», «пока не ок, ≤2 → починить», «иначе → выход escalate».", "Имя счётчика должно быть своим у каждого цикла."]},
    error: {what: "Берётся только если шаг упал (после его повторов). Текст ошибки — в «error», шаг — в «error_node».", tips: ["Не больше одной такой стрелки на шаг.", "Без неё упавший шаг останавливает прогон (его можно продолжить)."]},
  },
  room_: {what: "Свойства всей комнаты. Шаги и стрелки выбираются кликом на схеме.", tips: ["Описание и теги — по ним комнату находит роутер.", "Лимиты — страховка от убежавшей комнаты."], sec: "g-room"},
};

// ── правая панель ─────────────────────────────────────────────────────────────────
function ihead(type, title, sub) {
  const chip = el("span", {class: "chipic"}, icon("i-" + type));
  chip.style.background = `var(--c-${type}, var(--c-room))`; chip.style.color = `var(--k-${type}, var(--k-room))`;
  return el("div", {class: "ihead"}, chip, el("div", {}, el("div", {class: "ttl"}, title), sub ? el("div", {class: "sub"}, sub) : null));
}
function panel() {
  const side = $("#inspector"); side.replaceChildren();
  refreshFieldList();
  if (saveAs) side.append(saveAsBox());
  if (inspectMode === "new") { newRoomForm(side); return; }
  if (!room) { welcomePanel(side); return; }
  if (inspectMode === "switch") { switchPanel(side); return; }
  if (multi.size > 1) { multiPanel(side); return; }
  if (!sel) { roomPanel(side); return; }
  if (sel.kind === "exit") { exitPanel(side); return; }
  if (sel.kind === "edge") { edgePanel(side); return; }
  nodePanel(side);
}

function welcomePanel(side) {
  side.append(el("h2", {}, "С чего начать"));
  side.append(el("p", {}, "Комната — рецепт: цепочка шагов и правил, куда идти после каждого. Движок идёт по схеме сам; модель решает только внутри шагов-агентов."));
  side.append(el("button", {class: "btn primary", onclick: () => { inspectMode = "new"; panel(); }}, icon("i-plus", "sm"), "Создать комнату"));
  side.append(el("p", {class: "hint"}, "Или откройте существующую в списке сверху: комнаты «комплекта» лучше всего показывают, как всё устроено."));
  side.append(el("h2", {}, "Что дальше"));
  side.append(el("button", {class: "btn", onclick: () => openGuide("g-start")}, icon("i-book", "sm"), "Быстрый старт за 5 минут"));
}

function newRoomForm(side) {
  let name = "my_room", desc = "", pick = "";
  side.append(el("h2", {}, "Новая комната"));
  const nameIn = inp(name, (v) => { name = v; });
  side.append(field("Имя", nameIn, "Латиница, цифры, «_». По нему комнату запускают: /rooms run имя"));
  side.append(field("Что она делает", inp(desc, (v) => { desc = v; }, {ph: "одной фразой, как будет звучать задача"})));
  side.append(el("div", {class: "lbl"}, "С чего начать"));
  const list = el("div");
  const options = [{id: "", title: "Пустая комната", summary: "Один шаг-агент и один выход. Остальное добавите сами."}, ...EXAMPLES.map((e) => ({id: e.name, title: e.title, summary: e.summary}))];
  const render = () => {
    list.replaceChildren();
    for (const o of options) {
      const b = el("button", {class: "tpl", "aria-pressed": o.id === pick ? "true" : "false", onclick: () => { pick = o.id; render(); }},
        el("b", {}, o.title), el("span", {}, short(o.summary, 150)));
      list.append(b);
    }
  };
  render();
  side.append(list);
  side.append(el("div", {class: "footer"},
    el("button", {class: "btn primary", onclick: () => {
      const nm = name.trim();
      if (!/^[a-z][a-z0-9_]{1,40}$/.test(nm)) { say("Имя: строчная латиница, цифры и «_», начинается с буквы.", "bad"); return; }
      if (META.rooms.some((r) => r.name === nm)) { say("Комната «" + nm + "» уже есть. Выберите другое имя.", "bad"); return; }
      if (dirty && !confirm("Есть несохранённые изменения. Начать новую комнату без сохранения?")) return;
      const ex = EXAMPLES.find((e) => e.name === pick);
      const r = ex ? clone(ex.room) : blankRoom(nm);
      r.room = nm;
      if (desc.trim()) r.description = desc.trim();
      else if (!ex) r.description = "";
      $("#roomSelect").value = "";
      resetState(r, {}, "new", true);
      say(ex ? "Создано из шаблона «" + ex.title + "». Нажимайте на шаги и читайте справа, что они делают." : "Создана пустая комната. Добавляйте шаги слева.", "");
    }}, "Создать"),
    el("button", {class: "btn", onclick: () => { inspectMode = null; panel(); }}, "Отмена")));
  setTimeout(() => nameIn.select(), 0);
}

function saveAsBox() {
  let name = saveAs.name;
  const i = inp(name, (v) => { name = v; });
  return el("div", {class: "notice warn"},
    el("b", {}, "Комната из комплекта только для чтения."), el("div", {}, "Сохраните копию под своим именем:"), i,
    el("div", {class: "row"},
      el("button", {class: "btn primary sm", onclick: () => { room.room = name.trim(); source = "new"; saveAs = null; touch("name", true); save(false); }}, "Сохранить копию"),
      el("button", {class: "btn sm", onclick: () => { saveAs = null; panel(); }}, "Отмена")));
}

function roomPanel(side) {
  side.append(ihead("room", room.room || "Новая комната", source === "starter" ? "Из комплекта · копия при сохранении" : source === "user" ? "Ваша комната" : "Ещё не сохранена"));
  side.append(helpBlock(HELP.room_, "Что здесь настраивается"));
  const nodesN = room.nodes.length, exitsN = exitNames().length;
  side.append(el("div", {class: "says"}, `${nodesN} ${plural(nodesN, "шаг", "шага", "шагов")}, ${exitsN} ${plural(exitsN, "выход", "выхода", "выходов")}. ` +
    `Начинается с шага «${room.entry}». Подробный пересказ — на вкладке «Словами».`));
  side.append(field("Имя", inp(room.room, (v) => { room.room = v; touch("rname"); }), "Так комнату вызывают: /rooms run " + (room.room || "имя")));
  side.append(field("Описание", area(room.description, (v) => { room.description = v; touch("rdesc"); }, 3),
    "Пишите так, как будет звучать задача: «починить падающие тесты»."));
  side.append(field("Слова для поиска (теги)", listInput(room.tags, (v) => { room.tags = v; touch("rtags"); }, "сломалось, не работает, bug"),
    "Через запятую: русские и английские слова задачи. По ним роутер подбирает комнату."));

  side.append(el("h2", {}, "Лимиты"));
  side.append(el("div", {class: "row"},
    field("Шагов", inp(room.budget.max_iterations, (v) => { room.budget.max_iterations = v; touch("b1"); }, {type: "number", min: 1})),
    field("Токенов", inp(room.budget.max_tokens, (v) => { room.budget.max_tokens = v; touch("b2"); }, {type: "number", min: 1})),
    field("Секунд", inp(room.budget.max_seconds ?? "", (v) => { if (v) room.budget.max_seconds = v; else delete room.budget.max_seconds; touch("b3"); }, {type: "number", min: 1}))));
  side.append(el("p", {class: "hint"}, "Шагов — сколько шагов выполнится за прогон (круг цикла — новый шаг). Токены считают только агенты. Секунды — необязательно. Превышение остановит комнату с выходом «budget»."));

  side.append(el("h2", {}, "Параметры"));
  side.append(kvEditor(room.defaults, (v) => { room.defaults = v; touch("defaults"); }, {strings: true, keyPh: "имя, напр. test_command", valPh: "значение по умолчанию", addLabel: "Добавить параметр"}));
  side.append(el("p", {class: "hint"}, "В тексте шагов параметр вставляется как {{имя}}. При запуске его можно переопределить: test_command=\"pytest -x\"."));

  roomRulesPanel(side);
  roomTriggersPanel(side);

  side.append(el("h2", {}, "Что после выхода"));
  const exits = exitNames();
  if (!exits.length) side.append(el("p", {class: "hint"}, "Выходов пока нет. Добавьте их слева («Завершить комнату»)."));
  else side.append(kvEditor(room.next, (v) => { room.next = v; touch("next"); }, {
    strings: true, addLabel: "Настроить выход",
    keyChoices: (cur) => [...new Set([...exits, ...(cur ? [cur] : [])])],
    valueChoices: [...TARGET_OPTIONS, ...META.rooms.filter((r) => r.name !== room.room).map((r) => [r.name, "запустить комнату «" + r.name + "»"])],
  }));
  side.append(el("p", {class: "hint"}, "Чем комната закончилась, то и решает, что дальше: закончить, передать на разбор или запустить другую комнату."));

  side.append(el("details", {class: "adv"}, el("summary", {}, "Дополнительно"), el("div", {class: "body"},
    field("Первый шаг", select(room.nodes.map((n) => n.id), room.entry, (v) => { room.entry = v; touch("entry", true); })),
    field("Итоговый отчёт", withInserter(area(room.report, (v) => { room.report = v; touch("report"); }, 4)), "Текст по окончании; {{поля}} заполнятся итоговым состоянием."))));

  side.append(el("div", {class: "footer"},
    el("button", {class: "btn", onclick: () => setView("words")}, "Пересказать словами"),
    source === "user" ? el("button", {class: "btn danger", onclick: removeRoom}, icon("i-trash", "sm"), "Удалить комнату") : null));
}

function exitPanel(side) {
  const name = sel.id.slice(5);
  side.append(ihead("exit", "Выход «" + name + "»", "Чем комната может закончиться"));
  side.append(el("div", {class: "says"}, ({done: "Успешное завершение.", failed: "Ожидаемый провал: не получилось, но это не поломка.",
    escalate: "Передача человеку или на разбор (triage): считается неудачей."})[name] || "Именованный исход комнаты. Имя придумываете вы."));
  const nm = inp(name, () => {});
  nm.addEventListener("change", () => { if (renameExit(name, nm.value)) touch("exrename", true); else nm.value = name; });
  side.append(field("Имя", nm));
  side.append(field("Что после этого выхода", select([...TARGET_OPTIONS, ...META.rooms.filter((r) => r.name !== room.room).map((r) => [r.name, "запустить комнату «" + r.name + "»"])],
    (room.next || {})[name] || "END", (v) => { room.next = room.next || {}; room.next[name] = v; touch("exnext"); }),
    "«закончить» — прогон завершён. «на разбор» — комната triage решит, что делать."));
  const incoming = room.edges.map((e, i) => [e, i]).filter(([e]) => e.to === sel.id);
  side.append(el("h2", {}, "Сюда ведут"));
  if (!incoming.length) side.append(el("p", {class: "hint"}, "Пока ни одна стрелка. Выберите шаг и нажмите «→» у его правого края, затем на этот выход."));
  for (const [e, i] of incoming) side.append(el("button", {class: "btn sm", onclick: () => selectEdge(i)}, "«" + e.from + "» · " + edgeWords(e)));
  side.append(el("div", {class: "footer"}, el("button", {class: "btn danger", onclick: removeSelected}, icon("i-trash", "sm"), "Удалить выход")));
}

function nodePanel(side) {
  const n = nodeById(sel.id);
  if (!n) { sel = null; roomPanel(side); return; }
  const says = el("div", {class: "says"}, describeNode(n));
  const upd = (key) => { says.textContent = describeNode(n); touch(key); };
  side.append(ihead(n.type, typeLabel(n.type), TYPES[n.type] ? TYPES[n.type].short : ""));
  side.append(says);
  if (n.note) side.append(el("div", {class: "notice note"}, el("b", {}, "Зачем этот шаг. "), n.note));
  const idIn = inp(n.id, () => {});
  idIn.addEventListener("change", () => { if (renameNode(n.id, idIn.value)) touch("rename", true); else idIn.value = n.id; });
  side.append(field("Название шага", idIn, "Латиница, цифры, «_». Так шаг виден на схеме и в условиях."));
  commonProps(side, n, upd);

  const t = n.type;
  if (extNodeForm(side, n, upd)) {
    /* форма построена в ext.js */
  } else if (t === "tool") {
    side.append(field("Инструмент", select(META.tools, n.tool, (v) => { n.tool = v; touch("tool", true); })));
    if (n.tool === "run_command") {
      n.args = n.args || {};
      const cmd = area(n.args.command || "", (v) => { n.args.command = v; upd("cmd" + n.id); }, 2, true);
      side.append(field("Команда", withInserter(cmd), "Выполнится как есть. {{параметр}} подставляется из параметров комнаты."));
    }
    const rest = {...(n.args || {})}; if (n.tool === "run_command") delete rest.command;
    if (n.tool !== "run_command" || Object.keys(rest).length) {
      side.append(field("Параметры инструмента", kvEditor(n.tool === "run_command" ? rest : n.args, (v) => {
        n.args = n.tool === "run_command" ? {command: (n.args || {}).command || "", ...v} : v; upd("args" + n.id);
      }, {keyPh: "параметр", valPh: "значение", addLabel: "Добавить параметр"})));
    }
    side.append(field("Запомнить из результата", kvEditor(n.sets, (v) => { n.sets = v; upd("sets" + n.id); },
      {keyPh: "поле, напр. tests_passed", keyList: "fieldsList", valueChoices: META.derived.map((d) => [d, DERIVED_LABEL[d] || d]), addLabel: "Запомнить поле"}),
      "Вердикт даёт процесс, а не модель. Потом в условии: state.tests_passed."));
    side.append(check("Спросить вас перед запуском", n.confirm, (v) => { n.confirm = v; upd("confirm"); }, "Нет «да» — шаг не выполнится."));
    side.append(el("details", {class: "adv"}, el("summary", {}, "Дополнительно"), el("div", {class: "body"},
      check("Можно безопасно повторить после обрыва", n.idempotent !== false, (v) => { n.idempotent = v; touch("idem"); }, "Снимите для команд, меняющих мир."),
      field("Повторов при сбое", inp(n.retries || 0, (v) => { n.retries = v || 0; touch("ret"); }, {type: "number"}), "Только если инструмент выбросил ошибку; код возврата 1 — это данные, а не сбой."),
      field("Взять значение из поля (input_from)", inp(n.input_from, (v) => { n.input_from = v; touch("inf"); }, {ph: "state.поле", list: null})))));
  } else if (t === "agent") {
    side.append(field("Роль", select(META.roles.map((r) => [r, ROLE_LABEL[r] || r]), n.role || "Coder", (v) => { n.role = v; upd("role"); })));
    const ins = area(n.instruction, (v) => { n.instruction = v; touch("inst" + n.id); }, 6);
    side.append(field("Что сделать", withInserter(ins), "Что делать и что ответить: «… Ответь JSON-объектом: diagnosis=<причина>»."));
    side.append(field("Что вернуть (поля)", listInput(n.writes, (v) => { n.writes = v; upd("writes" + n.id); }, "diagnosis, fix_summary"),
      "Единственные поля, которые агент вправе записать; остальное движок отбросит."));
    const own = n.tools != null;
    side.append(el("div", {class: "lbl"}, "Инструменты"));
    side.append(check("Только выбранные", own, (v) => { if (v) n.tools = n.tools || []; else delete n.tools; upd("tools"); panel(); }, "Читающему агенту оставьте только чтение — он ничего не изменит."));
    if (own) {
      const box = el("div", {class: "tools"});
      for (const name of META.tools) {
        const c = el("input", {type: "checkbox"}); c.checked = n.tools.includes(name);
        c.addEventListener("change", () => { n.tools = c.checked ? [...n.tools, name] : n.tools.filter((x) => x !== name); upd("tools"); });
        box.append(el("label", {class: "check"}, c, el("span", {}, name)));
      }
      side.append(box);
    }
    n.budget = n.budget || {max_iterations: 6, max_tokens: 20000};
    side.append(el("details", {class: "adv"}, el("summary", {}, "Лимиты и дополнительно"), el("div", {class: "body"},
      el("div", {class: "row"},
        field("Шагов", inp(n.budget.max_iterations, (v) => { n.budget.max_iterations = v; touch("ab1"); }, {type: "number", min: 1})),
        field("Токенов", inp(n.budget.max_tokens, (v) => { n.budget.max_tokens = v; touch("ab2"); }, {type: "number", min: 1}))),
      field("Показать агенту значения полей (context)", listInput(n.context, (v) => { n.context = v; touch("ctx"); }, "error_summary"), "Они появятся блоком CONTEXT."),
      field("Повторов при сбое", inp(n.retries || 0, (v) => { n.retries = v || 0; touch("ret"); }, {type: "number"}), "Только если агент упал с ошибкой."))));
    agentExtras(side, n, upd);
  } else if (t === "condition") {
    side.append(field("Условие (подпись)", inp(n.condition, (v) => { n.condition = v; upd("cond" + n.id); }, {ph: "state.ok"}),
      "Маршрут выбирают условия на исходящих стрелках — добавьте «если» и «иначе»."));
  } else if (t === "human_pause") {
    side.append(field("Вопрос", withInserter(area(n.prompt, (v) => { n.prompt = v; upd("pr" + n.id); }, 3)), "Покажите факты, по которым вы решаете."));
    side.append(field("Сохранить ответ в поле", inp(n.into, (v) => { n.into = v; upd("into" + n.id); }, {ph: "go"}),
      "Дополнительно появится «<поле>_approved» — был ли ответ «да». По нему удобно ветвиться."));
  } else if (t === "set") {
    side.append(field("Значения", kvEditor(n.values, (v) => { n.values = v; upd("vals" + n.id); }, {keyPh: "поле", valPh: "значение, {{поле}} или = выражение", addLabel: "Добавить значение"}),
      "Начните значение с «=», чтобы посчитать: «= state.n + 1». Можно + − * / // % и скобки; поля — как state.имя."));
  } else if (t === "notify") {
    side.append(field("Сообщение", withInserter(area(n.message, (v) => { n.message = v; upd("msg" + n.id); }, 3))));
    side.append(field("Куда", select(META.channels.map((c) => [c, CHANNEL_LABEL[c] || c]), n.channel || "console", (v) => { n.channel = v; upd("ch"); })));
    side.append(el("details", {class: "adv"}, el("summary", {}, "Дополнительно"), el("div", {class: "body"},
      field("Заголовок", inp(n.title, (v) => { n.title = v; touch("title"); })))));
    notifyExtras(side, n, upd);
  } else if (t === "spawn_room") {
    side.append(el("p", {class: "hint"}, "Этот блок есть только в системной комнате triage. В своих комнатах он запрещён."));
    side.append(field("Задание модели", area(n.instruction, (v) => { n.instruction = v; touch("inst" + n.id); }, 5),
      "Что именно предложить. Результат — только данные, они проходят проверку и ваше утверждение."));
  } else if (t === "room") {
    side.append(field("Какую комнату запустить", select(META.rooms.filter((r) => r.name !== room.room).map((r) => r.name), n.room, (v) => { n.room = v; upd("subroom"); })));
    side.append(field("Что передать", kvEditor(n.inputs, (v) => { n.inputs = v; touch("in" + n.id); }, {strings: true, keyPh: "поле подкомнаты", valPh: "{{моё_поле}}", addLabel: "Передать поле"}),
      "{{task}} передаётся сам."));
    side.append(field("Что забрать обратно (поля)", listInput(n.outputs, (v) => { n.outputs = v; upd("outs" + n.id); }, "tests_passed, notes")));
    side.append(el("details", {class: "adv"}, el("summary", {}, "Дополнительно"), el("div", {class: "body"},
      field("Поле для имени выхода подкомнаты", inp(n.into, (v) => { n.into = v; upd("into" + n.id); }, {ph: n.id + "_exit"})),
      field("Повторов при сбое", inp(n.retries || 0, (v) => { n.retries = v || 0; touch("ret"); }, {type: "number"})))));
  }

  side.append(helpBlock(HELP[n.type], "Что делает этот блок"));
  stepTestBox(side, n);

  side.append(el("h2", {}, "Что дальше"));
  const mine = room.edges.map((e, i) => i).filter((i) => room.edges[i].from === n.id);
  if (!mine.length) side.append(el("p", {class: "hint"}, "Стрелок пока нет. Нажмите «→» у правого края шага и выберите, куда вести."));
  else side.append(el("p", {class: "hint"}, "Проверяются сверху вниз: первый подошедший «если»/«пока» выигрывает, иначе — «иначе»/«всегда»."));
  const outs = el("div", {class: "outs"});
  mine.forEach((i, k) => {
    const e = room.edges[i];
    outs.append(el("div", {class: "o"},
      el("button", {class: "btn sm go", onclick: () => selectEdge(i)}, (k + 1) + ". " + edgeWords(e) + " → " + targetWords(e.to)),
      el("button", {class: "btn icon sm", title: "Проверять раньше", "aria-label": "Выше", disabled: k === 0, onclick: () => { moveEdge(i, -1); touch("order", true); }}, icon("i-up", "sm")),
      el("button", {class: "btn icon sm", title: "Проверять позже", "aria-label": "Ниже", disabled: k === mine.length - 1, onclick: () => { moveEdge(i, 1); touch("order", true); }}, icon("i-down", "sm"))));
  });
  side.append(outs);
  side.append(el("button", {class: "btn sm", onclick: () => startConnect(n.id)}, icon("i-link", "sm"), "Добавить стрелку"));

  side.append(el("div", {class: "footer"},
    room.entry === n.id ? el("span", {class: "chip"}, "Первый шаг комнаты") : el("button", {class: "btn sm", onclick: () => { room.entry = n.id; touch("entry", true); }}, "Сделать первым"),
    el("button", {class: "btn sm", onclick: duplicateSelected}, icon("i-copy", "sm"), "Дублировать"),
    el("button", {class: "btn sm danger", onclick: removeSelected}, icon("i-trash", "sm"), "Удалить шаг")));
}

function edgePanel(side) {
  const e = room.edges[sel.index];
  if (!e) { sel = null; roomPanel(side); return; }
  const kind = edgeKind(e);
  const chip = el("span", {class: "chipic"}, icon("i-link")); chip.style.background = "var(--c-room)"; chip.style.color = "var(--k-room)";
  side.append(el("div", {class: "ihead"}, chip, el("div", {}, el("div", {class: "ttl"}, "Стрелка"), el("div", {class: "sub"}, "«" + e.from + "» → " + targetWords(e.to)))));
  side.append(el("div", {class: "says"}, `Когда «${e.from}» закончится: ${edgeWords(e)} — идти в ${targetWords(e.to)}.`));
  const ids = room.nodes.map((n) => n.id);
  side.append(el("div", {class: "row"},
    field("Из шага", select(ids, e.from, (v) => { e.from = v; touch("efrom", true); })),
    field("Куда", select([...ids.map((x) => [x, x]), ...exitNames().map((x) => ["EXIT:" + x, "выход «" + x + "»"])], e.to, (v) => { e.to = v; touch("eto", true); }))));
  side.append(field("Когда", select([["uncond", "всегда"], ["if", "если выполнено условие"], ["else", "иначе (ничего другое не подошло)"],
    ["while", "пока выполнено условие (цикл)"], ["error", "если шаг завершился ошибкой"]], kind, (v) => {
    delete e.if; delete e.while; delete e.max; delete e.id; e.else = false; e.on_error = false;
    if (v === "if") e.if = guessCondition(e.from);
    if (v === "else") e.else = true;
    if (v === "while") { e.while = "not " + guessCondition(e.from); e.max = 3; e.id = "loop_" + (sel.index + 1); }
    if (v === "error") e.on_error = true;
    touch("ekind", true);
  })));
  if (kind === "if") side.append(field("Условие", condBuilder(e.if, (v) => { e.if = v; touch("econd" + sel.index); })));
  if (kind === "while") {
    side.append(field("Повторять, пока", condBuilder(e.while, (v) => { e.while = v; touch("econd" + sel.index); })));
    side.append(el("div", {class: "row"},
      field("Не больше раз", inp(e.max, (v) => { e.max = v; touch("emax"); }, {type: "number", min: 1})),
      field("Имя счётчика", inp(e.id, (v) => { e.id = v; touch("eid"); }))));
  }
  side.append(helpBlock(Object.assign({sec: "g-edges"}, HELP.edge[kind]), "Как работает этот вид стрелки"));
  side.append(el("div", {class: "footer"},
    el("button", {class: "btn sm", title: "Проверять раньше других стрелок этого шага", onclick: () => { sel.index = moveEdge(sel.index, -1); touch("order", true); }}, icon("i-up", "sm"), "Раньше"),
    el("button", {class: "btn sm", title: "Проверять позже", onclick: () => { sel.index = moveEdge(sel.index, 1); touch("order", true); }}, icon("i-down", "sm"), "Позже"),
    el("button", {class: "btn sm danger", onclick: removeSelected}, icon("i-trash", "sm"), "Удалить стрелку")));
}

// ── палитра ───────────────────────────────────────────────────────────────────────
function buildPalette() {
  if (typeof buildPaletteExt === "function") { buildPaletteExt(); return; }
  const p = $("#paletteBody"); p.replaceChildren();
  p.append(el("h2", {}, "Добавить шаг"));
  for (const t of PALETTE_ORDER) {
    if (META.node_types.length && !META.node_types.includes(t)) continue;
    const info = TYPES[t];
    const chip = el("span", {class: "chipic"}, icon(info.icon)); chip.style.background = `var(--c-${t})`; chip.style.color = `var(--k-${t})`;
    p.append(el("button", {class: "pal", title: info.hint, onclick: () => addNode(t)}, chip,
      el("span", {}, el("b", {}, info.name), el("span", {class: "t"}, info.short))));
  }
  p.append(el("h2", {}, "Завершить комнату"));
  const exits = el("div", {class: "exits", id: "exitBox"});
  for (const [name, label, why] of EXIT_PRESETS) {
    exits.append(el("button", {class: "btn sm", title: why + " — выход «" + name + "»", onclick: () => addExit(name)}, icon("i-exit", "sm"), label));
  }
  p.append(exits);
  const custom = inp("", () => {}, {ph: "свой выход…"});
  custom.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && custom.value.trim()) { addExit(custom.value); custom.value = ""; } });
  p.append(el("div", {class: "customexit"}, custom, el("button", {class: "btn sm", "aria-label": "Добавить выход", onclick: () => { if (custom.value.trim()) { addExit(custom.value); custom.value = ""; } }}, icon("i-plus", "sm"))));
  p.append(el("div", {class: "tip"}, "Выберите шаг на схеме — новый блок добавится сразу после него и соединится стрелкой."));
  p.append(el("p", {class: "palnote"}, "Выход — это «чем закончилась комната». Из каждого шага должен быть путь хотя бы к одному выходу."));
}

// ── Словами / JSON ────────────────────────────────────────────────────────────────
function renderWords() {
  const host = $("#wordsBody"); host.replaceChildren();
  if (!room) { host.append(el("p", {class: "muted"}, "Откройте или создайте комнату.")); return; }
  host.append(el("h2", {}, room.room));
  host.append(el("p", {class: "lead"}, room.description || "Описания нет."));
  const b = room.budget;
  host.append(el("p", {}, `Лимиты: не более ${b.max_iterations} шагов и ${b.max_tokens} токенов` + (b.max_seconds ? `, ${b.max_seconds} секунд` : "") +
    ". Если превысит — остановится с выходом «budget»."));
  const defs = Object.entries(room.defaults || {});
  if (defs.length) host.append(el("p", {}, "Параметры по умолчанию: " + defs.map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join("; ") + "."));
  host.append(el("h3", {}, "Ход работы"));
  const ol = el("ol", {class: "steps"});
  bfsOrder().forEach((id, i) => {
    const n = nodeById(id);
    const chip = el("span", {class: "num"}, String(i + 1));
    const li = el("li", {}, el("div", {class: "st"}, chip, icon("i-" + n.type), el("span", {}, n.id + (id === room.entry ? " (старт)" : "")), el("span", {class: "muted"}, "· " + typeLabel(n.type))),
      el("div", {}, describeNode(n)));
    const outs = room.edges.filter((e) => e.from === id);
    if (outs.length) {
      const ul = el("ul", {class: "go"});
      for (const e of outs) ul.append(el("li", {}, el("b", {}, edgeWords(e)), " → " + targetWords(e.to)));
      li.append(ul);
    } else li.append(el("div", {class: "hint"}, "Из этого шага нет стрелок."));
    ol.append(li);
  });
  host.append(ol);
  host.append(el("h3", {}, "Чем может закончиться"));
  const ex = exitNames();
  if (!ex.length) host.append(el("p", {class: "hint"}, "Выходов пока нет."));
  else host.append(el("ul", {}, ex.map((x) => el("li", {}, "«" + x + "» — " + (({done: "успешно", failed: "провал", escalate: "передать человеку/на разбор"})[x] || "именованный исход") +
    ((room.next || {})[x] ? "; затем " + ((room.next || {})[x] === "END" ? "конец" : (room.next || {})[x] === "TRIAGE" ? "разбор сбоя" : "комната «" + room.next[x] + "»") : "")))));
  if (room.report) host.append(el("h3", {}, "Итоговый отчёт"), el("pre", {class: "mono"}, room.report));
  if (lastMermaid) {
    const d = el("details", {class: "adv"}, el("summary", {}, "Код диаграммы (Mermaid) — для README или issue"), el("div", {class: "body"},
      el("textarea", {class: "mono", rows: 10, readonly: true}, lastMermaid)));
    host.append(d);
  }
}
function renderJson() {
  const host = $("#jsonBody"); host.replaceChildren();
  if (!room) { host.append(el("p", {class: "muted"}, "Откройте или создайте комнату.")); return; }
  const t = area(JSON.stringify(cleaned(room), null, 2), () => {}, 24, true);
  const note = el("div", {class: "hint"}, "Правьте текст и нажмите «Применить». Проверка и сохранение — как обычно.");
  host.append(el("p", {}, "Полный текст комнаты. Удобно, чтобы скопировать, вставить чужую или поправить много полей сразу."), t,
    el("div", {class: "row"},
      el("button", {class: "btn primary", onclick: () => {
        try {
          const obj = JSON.parse(t.value);
          if (!obj || typeof obj !== "object" || !Array.isArray(obj.nodes) || !Array.isArray(obj.edges)) throw new Error("нужны поля nodes и edges");
          obj.tags = obj.tags || []; obj.defaults = obj.defaults || {}; obj.next = obj.next || {};
          obj.nodes.forEach((n) => { if (n.type === "tool") n.args = n.args || {}; });
          room = obj; layout = {}; autoLayout(); sel = null;
          touch("json", true); note.textContent = "Применено."; setView("graph");
        } catch (e) { note.textContent = "JSON не разобран: " + e.message; }
      }}, "Применить"),
      el("button", {class: "btn", onclick: () => { t.select(); try { navigator.clipboard.writeText(t.value); note.textContent = "Скопировано."; } catch (e) { note.textContent = "Выделено — нажмите Ctrl+C."; } }}, icon("i-copy", "sm"), "Копировать")),
    note);
}
function setView(v) {
  view = v;
  for (const [id, name] of [["tabGraph", "graph"], ["tabWords", "words"], ["tabJson", "json"], ["tabRuns", "runs"]]) $("#" + id).setAttribute("aria-selected", String(name === v));
  $("#canvasWrap").hidden = v !== "graph";
  $("#canvasTools").hidden = v !== "graph";
  $("#paneWords").hidden = v !== "words";
  $("#paneJson").hidden = v !== "json";
  $("#paneRuns").hidden = v !== "runs";
  if (v === "words") renderWords();
  if (v === "json") renderJson();
  if (v === "runs") renderRuns();
  if (v === "graph") draw();
}

// ── проверка, загрузка, сохранение ───────────────────────────────────────────────
const FRIENDLY = [
  [/nodes cannot reach any EXIT: \[(.*)\]/, (m) => [`Из шагов ${m[1].replace(/'([^']+)'/g, "«$1»")} не дойти до выхода.`,
    "Проведите стрелку к выходу (панель «Завершить комнату») или к шагу, из которого выход есть. Часто забыта стрелка «иначе» на цикле."]],
  [/edge '([^']+)' -> unknown target '([^']+)'/, (m) => [`Стрелка из «${m[1]}» ведёт в несуществующий шаг «${m[2]}».`, "Выберите цель стрелки заново."]],
  [/edge from unknown node '([^']+)'/, (m) => [`Есть стрелка из несуществующего шага «${m[1]}».`, "Удалите её или выберите существующий шаг."]],
  [/node '([^']+)': tool '([^']+)' does not exist/, (m) => [`Шаг «${m[1]}»: инструмента «${m[2]}» нет.`, "Выберите инструмент из списка."]],
  [/node '([^']+)': tools lists '([^']+)', which does not exist/, (m) => [`Агент «${m[1]}»: в списке инструментов есть несуществующий «${m[2]}».`, "Снимите его в списке инструментов."]],
  [/node '([^']+)': unknown role '([^']+)'/, (m) => [`Агент «${m[1]}»: неизвестная роль «${m[2]}».`, "Выберите роль из списка."]],
  [/edge '([^']+)'->'([^']+)': bad condition \((.*)\)/, (m) => [`Условие на стрелке «${m[1]}» → «${m[2]}» записано неверно (${m[3]}).`,
    "Соберите условие из поля, оператора и значения. В своём выражении: state.поле, ==, and/or/not, True/False."]],
  [/duplicate node ids: (.*)/, (m) => [`Два шага с одним названием: ${m[1]}.`, "Переименуйте один из них."]],
  [/entry node '([^']+)' is not in the room/, (m) => [`Первый шаг «${m[1]}» не найден.`, "Выберите первый шаг в свойствах комнаты (Дополнительно)."]],
  [/next: '([^']+)' is not an exit of this room/, (m) => [`В «Что после выхода» указан выход «${m[1]}», которого у комнаты нет.`, "Создайте такой выход или уберите строку."]],
  [/next: exit '([^']+)' has an empty target/, (m) => [`У выхода «${m[1]}» не выбрано, что делать дальше.`, "Выберите значение или уберите строку."]],
  [/node '([^']+)': more than one on_error edge/, (m) => [`У шага «${m[1]}» больше одной стрелки «если шаг упал».`, "Оставьте одну."]],
  [/node '([^']+)': a room cannot call itself/, (m) => [`Шаг «${m[1]}» вызывает эту же комнату.`, "Выберите другую комнату."]],
  [/node '([^']+)': room '([^']+)' does not exist/, (m) => [`Шаг «${m[1]}»: комнаты «${m[2]}» нет.`, "Выберите существующую комнату."]],
  [/rooms call each other in a circle/, () => ["Комнаты вызывают друг друга по кругу.", "Разорвите цикл: одна из подкомнат не должна вызывать вызывающую."]],
  [/room name '([^']*)' must be snake_case/, (m) => [`Имя комнаты «${m[1]}» не подходит.`, "Строчная латиница, цифры и «_», начинается с буквы."]],
  [/room has no nodes/, () => ["В комнате нет шагов.", "Добавьте шаг слева."]],
  [/spawn_room is only allowed/, () => ["Блок «spawn_room» доступен только системной комнате triage.", "Удалите его."]],
  [/requires a 'budget'/, () => ["У агента не задан лимит.", "Откройте «Лимиты и дополнительно» у агента и укажите шаги и токены."]],
  [/requires 'max'/, () => ["У цикла «пока» нет верхней границы.", "Заполните «Не больше раз» и «Имя счётчика» на стрелке."]],
  [/requires a 'tool'/, () => ["У команды не выбран инструмент.", "Выберите инструмент."]],
  [/requires a 'condition'/, () => ["У развилки пустая подпись условия.", "Впишите условие, например state.ok."]],
  [/requires 'values'/, () => ["В узле «Значение» нет ни одного значения.", "Добавьте хотя бы одно."]],
  [/requires a 'message'/, () => ["У уведомления пустое сообщение.", "Впишите сообщение."]],
  [/requires 'room'/, () => ["У подкомнаты не выбрана комната.", "Выберите, какую запускать."]],
  [/field names must be identifiers/, () => ["Имя поля должно быть словом из латиницы, цифр и «_».", "Исправьте имена полей."]],
  [/mcp node '([^']+)' requires 'server'/, (m) => [`У шага MCP «${m[1]}» не выбран сервер.`, "Выберите сервер из списка (серверы добавляются командой /mcp)."]],
  [/mcp node '([^']+)' requires 'tool'/, (m) => [`У шага MCP «${m[1]}» не выбран инструмент.`, "Нажмите «Загрузить» и выберите инструмент сервера."]],
  [/skill node '([^']+)' requires 'skill'/, (m) => [`У шага «${m[1]}» не выбран скилл.`, "Выберите скилл из списка."]],
  [/plugin node '([^']+)' requires 'command'/, (m) => [`У шага плагина «${m[1]}» не указана команда.`, "Выберите команду плагина."]],
  [/leeth node '([^']+)' \((delegate|fill|filter|scan)\) requires 'instruction'/, (m) => [`У шага Leeth «${m[1]}» не написано, что делать.`, "Заполните поле с задачей или вопросом."]],
  [/leeth node '([^']+)' \((filter|classify)\) requires 'input_from'/, (m) => [`У шага Leeth «${m[1]}» не указан список.`, "Укажите поле, где лежит список."]],
  [/leeth node '([^']+)' \(classify\) requires args.categories/, (m) => [`У шага Leeth «${m[1]}» меньше двух категорий.`, "Впишите хотя бы две категории."]],
  [/leeth node '([^']+)' \(click\) requires/, (m) => [`У шага Leeth «${m[1]}» не сказано, что нажимать.`, "Укажите поле с результатом оценки страницы."]],
  [/foreach node '([^']+)' requires 'input_from'/, (m) => [`У цикла «${m[1]}» не указан список.`, "Укажите поле со списком."]],
  [/wait node '([^']+)' requires 'seconds'/, (m) => [`У паузы «${m[1]}» не задано время.`, "Укажите число секунд (до 3600)."]],
  [/loop node '([^']+)' \((for|times)\) requires 'stop'/, (m) => [`У цикла «${m[1]}» не задан конец.`, "Укажите «До» или «Сколько раз»."]],
  [/loop node '([^']+)' \((while|do_while|until)\) requires a 'condition'/, (m) => [`У цикла «${m[1]}» нет условия.`, "Соберите условие."]],
  [/loop node '([^']+)' \(for_in\) requires 'input_from'/, (m) => [`У цикла «${m[1]}» не указан словарь или список.`, "Укажите поле."]],
  [/loop node '([^']+)': 'step' cannot be 0/, (m) => [`У цикла «${m[1]}» шаг равен нулю.`, "Шаг не может быть 0: цикл никогда бы не кончился."]],
  [/(break|continue) node '([^']+)' requires 'loop'/, (m) => [`У блока «${m[2]}» не выбран цикл.`, "Выберите цикл, к которому он относится."]],
  [/node '([^']+)': a (break|continue) must lead back to its loop '([^']+)'/, (m) => [`Блок «${m[1]}» должен вести обратно в цикл «${m[3]}».`, "Выберите блок и нажмите «Провести стрелку в цикл»."]],
  [/node '([^']+)': (break|continue) names '([^']*)'/, (m) => [`Блок «${m[1]}» ссылается на «${m[3]}», а такого цикла нет.`, "Выберите существующий цикл."]],
  [/power node '([^']+)' requires 'mode'/, (m) => [`У блока питания «${m[1]}» не выбрано действие.`, "Выберите: выключить, перезагрузить, сон…"]],
  [/power node '([^']+)': (shutdown|restart) waits at least (\d+)/, (m) => [`Блок «${m[1]}»: пауза перед выключением должна быть не меньше ${m[3]} с.`, "Так вы успеете отменить действие."]],
  [/hint node '([^']+)' requires a 'message'/, (m) => [`У подсказки «${m[1]}» пустой текст.`, "Впишите подсказку или инструкцию для модели."]],
  [/assert node '([^']+)' requires a 'condition'/, (m) => [`У проверки «${m[1]}» нет условия.`, "Соберите условие."]],
  [/node '([^']+)': (.*) cannot run — (.*)/, (m) => [`Шаг «${m[1]}» не сможет выполниться: правила комнаты это запрещают.`, "Уберите шаг или смягчите правила комнаты («Правила комнаты» в её свойствах)."]],
  [/node '([^']+)': (?:field '([^']+)': )?bad expression \((.*)\)/, (m) => [`Шаг «${m[1]}»: выражение записано неверно (${m[3]}).`, "Можно + − * / // % , числа и state.поле."]],
  [/node '([^']+)': bad condition \((.*)\)/, (m) => [`Шаг «${m[1]}»: условие записано неверно (${m[2]}).`, "Соберите условие из поля, оператора и значения."]],
  [/node '([^']+)': (temperature|top_p|top_k|min_p|repeat_penalty|thinking_budget|reasoning_effort) /, (m) => [`Шаг «${m[1]}»: недопустимое значение «${m[2]}».`, "Значения ограничены допустимым диапазоном — он указан под полем."]],
  [/tool '([^']+)' starts another run and cannot be a step of a room/, (m) => [`Инструмент «${m[1]}» запускает другой прогон и не может быть шагом комнаты.`, "Для вызова другой комнаты есть блок «Подкомната»."]],
  [/packaged room and is read-only/, () => ["Комната из комплекта только для чтения.", "Сохраните копию под другим именем."]],
];
function explain(raw) {
  for (const [re, fn] of FRIENDLY) {
    const m = String(raw).match(re);
    if (m) { const [text, fix] = fn(m); return {raw, text, fix}; }
  }
  return {raw, text: String(raw), fix: ""};
}

function scheduleValidate() { clearTimeout(validateTimer); validateTimer = setTimeout(validate, 450); }
async function validate() {
  if (!room) return;
  const {status, data} = await api("/api/validate", {room: cleaned(room)});
  errors = data.errors || [];
  friendly = errors.map(explain);
  lastFlow = data.flow || ""; lastMermaid = data.mermaid || "";
  if (status !== 200) { say("Ошибка проверки: " + (data.error || status), "bad"); return; }
  const notes = (data.warnings || []).map((w) => ({raw: w, text: w, fix: ""}));
  if (data.ok) say(`Комната корректна: ${room.nodes.length} ${plural(room.nodes.length, "шаг", "шага", "шагов")}, ${room.edges.length} ${plural(room.edges.length, "стрелка", "стрелки", "стрелок")}.` +
    (dirty ? " Не сохранена." : "") + (notes.length ? ` Замечаний: ${notes.length}.` : ""), "ok", notes);
  else say(`Комната пока не годится (${errors.length}). Нажмите на пункт — подсветится шаг.`, "bad", friendly);
  draw();
  if (view === "words") renderWords();
}
function focusFromError(text) {
  for (const m of String(text).matchAll(/'([^']+)'/g)) {
    if (nodeById(m[1])) { sel = {kind: "node", id: m[1]}; inspectMode = null; setView("graph"); panel(); draw(); revealNode(m[1]); return; }
  }
}

async function openRoom(name) {
  if (dirty && !confirm("Есть несохранённые изменения. Открыть другую комнату без сохранения?")) { $("#roomSelect").value = room ? room.room : ""; return; }
  const {status, data} = await api("/api/room?name=" + encodeURIComponent(name));
  if (status !== 200) { say("Не удалось открыть: " + (data.error || status), "bad"); return; }
  $("#roomSelect").value = name;
  resetState(data.room, data.layout || {}, data.source, false);
  if (data.source === "starter") say("Комната из комплекта: смотрите и меняйте сколько угодно, сохранится копия под вашим именем.", "");
}
async function save(replace) {
  if (!room) return;
  if (!room.room) { say("У комнаты нет имени.", "bad"); return; }
  const body = {room: cleaned(room), layout, replace: !!replace || source === "user"};
  let {status, data} = await api("/api/save", body);
  if (status === 409 && data.exists) {
    if (!confirm("Комната «" + room.room + "» уже есть. Заменить её?")) return;
    ({status, data} = await api("/api/save", {...body, replace: true}));
  } else if (status === 409) {
    saveAs = {name: (room.room.replace(/_custom$/, "") + "_custom")};
    inspectMode = null; sel = null; panel(); say("Комната из комплекта только для чтения — сохраните копию справа.", "");
    return;
  }
  if (status === 200 && data.ok) {
    dirty = false; source = "user"; baseline = snap();
    await refreshList(room.room);
    say("Сохранено: " + data.saved, "ok");
    chrome();
  } else {
    errors = data.errors || []; friendly = errors.map(explain);
    say("Не сохранено: " + errors.length + " проблем(ы).", "bad", friendly); draw();
  }
}
async function removeRoom() {
  if (!room || source !== "user") { say("Удалять можно только свои сохранённые комнаты.", "bad"); return; }
  if (!confirm("Удалить комнату «" + room.room + "»? Это нельзя отменить.")) return;
  const {status, data} = await api("/api/delete", {name: room.room});
  if (status === 200) { room = null; sel = null; dirty = false; baseline = ""; await refreshList(); fullRender(); say("Удалено.", "ok"); }
  else say("Не удалено: " + (data.errors || []).join("; "), "bad");
}
async function refreshList(keep) {
  const {data} = await api("/api/state");
  META = data;
  const s = $("#roomSelect"); s.replaceChildren(el("option", {value: ""}, "— выберите комнату —"));
  const mine = META.rooms.filter((r) => r.source !== "starter"), base = META.rooms.filter((r) => r.source === "starter");
  if (mine.length) { const g = el("optgroup", {label: "Мои комнаты"}); for (const r of mine) g.append(el("option", {value: r.name}, r.name)); s.append(g); }
  if (base.length) { const g = el("optgroup", {label: "Комплект (копия при сохранении)"}); for (const r of base) g.append(el("option", {value: r.name}, r.name)); s.append(g); }
  s.value = keep || (room ? room.room : "");
}

// ── общий рендер ──────────────────────────────────────────────────────────────────
function chrome() {
  $("#btnUndo").disabled = !undoStack.length;
  $("#btnRedo").disabled = !redoStack.length;
  $("#btnSave").disabled = !room;
  const st = $("#roomState");
  if (!room) st.hidden = true;
  else {
    st.hidden = false;
    st.className = "chip " + (dirty ? "dirty" : source === "starter" ? "starter" : "");
    st.textContent = (dirty ? "не сохранена · " : "") + (source === "starter" ? "комплект" : source === "user" ? "моя" : "новая");
  }
  document.title = (dirty ? "● " : "") + (room ? room.room + " · " : "") + "Argent · редактор комнат";
}
function fullRender() { chrome(); panel(); draw(); if (view === "words") renderWords(); if (view === "json") renderJson(); }

// ── гайд: выдвижная панель, не модальная ─────────────────────────────────────────
let guideReady = false;
async function loadGuide() {
  const c = $("#gcontent"), nav = $("#gnav");
  try {
    const r = await fetch("/guide", {headers: {"X-Rooms-Token": TOKEN}});
    if (!r.ok) throw new Error("HTTP " + r.status);
    c.innerHTML = await r.text();           // упакованный статический файл программы, не пользовательские данные
  } catch (e) { c.textContent = "Не удалось загрузить гайд: " + e.message; return; }
  nav.replaceChildren();
  let group = "";
  for (const s of c.querySelectorAll(".g-sec")) {
    if (s.dataset.group && s.dataset.group !== group) { group = s.dataset.group; nav.append(el("h4", {"data-for": s.id}, group)); }
    nav.append(el("a", {"data-sec": s.id, onclick: () => scrollGuide(s.id)}, s.dataset.title || s.id));
  }
  const host = $("#g-examples");
  if (host) for (const ex of EXAMPLES) {
    host.append(el("div", {class: "g-ex"}, el("h5", {}, ex.title), el("p", {}, ex.summary), el("button", {class: "btn sm", "data-example": ex.name}, "Открыть как новую комнату")));
  }
  guideReady = true;
}
function scrollGuide(id) {
  const t = document.getElementById(id);
  if (t) { t.classList.remove("hid"); t.scrollIntoView({block: "start"}); }
}
async function openGuide(sec) {
  if (!guideReady) await loadGuide();
  $("#guide").hidden = false;
  if (sec) scrollGuide(sec);
}
function closeGuide() { $("#guide").hidden = true; }
function flashReal(selector) {
  const t = document.querySelector(selector);
  if (!t) return;
  t.scrollIntoView({block: "nearest"});
  t.classList.remove("flash"); void t.offsetWidth; t.classList.add("flash");
  setTimeout(() => t.classList.remove("flash"), 3000);
}
function loadExample(name) {
  const ex = EXAMPLES.find((e) => e.name === name);
  if (!ex) return;
  if (dirty && !confirm("Есть несохранённые изменения. Открыть пример без сохранения?")) return;
  $("#roomSelect").value = "";
  resetState(clone(ex.room), {}, "new", true);
  say("Пример «" + ex.title + "» открыт как новая несохранённая комната. Нажимайте на шаги и читайте справа.", "");
}

$("#gcontent").addEventListener("click", (ev) => {
  const f = ev.target.closest("[data-flash]"), x = ev.target.closest("[data-example]"), g = ev.target.closest("[data-goto]");
  if (x) { loadExample(x.dataset.example); }
  else if (f) flashReal(f.dataset.flash);
  else if (g) scrollGuide(g.dataset.goto);
});
$("#gcontent").addEventListener("scroll", () => {
  const top = $("#gcontent").getBoundingClientRect().top + 30;
  let cur = null;
  for (const s of $("#gcontent").querySelectorAll(".g-sec:not(.hid)")) if (s.getBoundingClientRect().top <= top) cur = s.id;
  for (const a of $("#gnav").querySelectorAll("a")) a.classList.toggle("cur", a.dataset.sec === cur);
});
$("#gsearch").addEventListener("input", (ev) => {
  const q = ev.target.value.trim().toLowerCase();
  for (const s of $("#gcontent").querySelectorAll(".g-sec")) {
    const hit = !q || s.textContent.toLowerCase().includes(q) || (s.dataset.title || "").toLowerCase().includes(q);
    s.classList.toggle("hid", !hit);
  }
  for (const a of $("#gnav").querySelectorAll("a")) a.classList.toggle("hid", !!q && document.getElementById(a.dataset.sec).classList.contains("hid"));
  for (const h of $("#gnav").querySelectorAll("h4")) h.classList.toggle("hid", !!q);
});

// ── подключение ───────────────────────────────────────────────────────────────────
window.addEventListener("keydown", (ev) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
  const mod = ev.ctrlKey || ev.metaKey;
  if (ev.key === "F1" || (ev.key === "?" && !typing)) { ev.preventDefault(); openGuide(); return; }
  if (mod && ev.key.toLowerCase() === "s") { ev.preventDefault(); save(); return; }
  if (mod && !typing && ev.key.toLowerCase() === "z") { ev.preventDefault(); ev.shiftKey ? redo() : undo(); return; }
  if (mod && !typing && ev.key.toLowerCase() === "y") { ev.preventDefault(); redo(); return; }
  if (mod && !typing && ev.key.toLowerCase() === "d") { ev.preventDefault(); duplicateSelected(); return; }
  if (mod && ev.key.toLowerCase() === "k") { ev.preventDefault(); focusPaletteSearch(); return; }
  if (mod && ev.key === "Enter") { ev.preventDefault(); runOpen(true); return; }
  if (mod && !typing && ev.key.toLowerCase() === "a" && room && view === "graph") { ev.preventDefault(); multi = new Set(room.nodes.map((n) => n.id)); sel = null; panel(); draw(); return; }
  if (ev.key === "Delete" && !typing) removeSelected();
  if (ev.key === "Escape") {
    if (multi.size && !typing) { multi.clear(); panel(); draw(); }
    else if (connectMode) { connectMode = false; connectFrom = null; draw(); }
    else if (!$("#guide").hidden && !typing) closeGuide();
  }
});
window.addEventListener("beforeunload", (ev) => { if (dirty) { ev.preventDefault(); ev.returnValue = ""; } });
$("#roomSelect").addEventListener("change", (ev) => { if (ev.target.value) openRoom(ev.target.value); });
$("#btnNew").onclick = () => { inspectMode = "new"; sel = null; saveAs = null; panel(); };
$("#btnSave").onclick = () => save();
$("#btnUndo").onclick = undo;
$("#btnRedo").onclick = redo;
$("#btnGuide").onclick = () => ($("#guide").hidden ? openGuide() : closeGuide());
$("#gclose").onclick = closeGuide;
$("#btnLayout").onclick = () => { if (room) { autoLayout(); touch("layout"); } };
$("#btnConnect").onclick = () => { if (!room) return; connectMode = !connectMode; connectFrom = null; draw(); };
$("#tabGraph").onclick = () => setView("graph");
$("#tabWords").onclick = () => setView("words");
$("#tabJson").onclick = () => setView("json");
$("#tabRuns").onclick = () => setView("runs");
$("#btnZoomIn").onclick = () => setZoom(zoom * 1.2);
$("#btnZoomOut").onclick = () => setZoom(zoom / 1.2);
$("#btnZoomReset").onclick = () => setZoom(1);
$("#btnFit").onclick = fitView;
$("#btnRun").onclick = () => runOpen(false);

initRun();

(async function init() {
  if (!TOKEN) { say("Нет токена доступа: откройте адрес, который напечатала команда /rooms editor.", "bad"); return; }
  await refreshList();
  try { await loadCatalogs(); } catch (e) { /* без каталогов тоже работает: поля вводятся руками */ }
  buildPalette();
  try { const ex = await api("/examples"); if (Array.isArray(ex.data)) EXAMPLES = ex.data; } catch (e) { /* без примеров тоже работает */ }
  chrome(); panel(); draw();
  const wanted = params.get("room");
  if (wanted) await openRoom(wanted);
  else say("Откройте комнату в списке, создайте новую или загляните в «Гайд».", "");
})();
