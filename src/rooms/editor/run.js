"use strict";
/* Запуск из редактора: панель запуска с ходом по шагам, подсветка на схеме, вопросы и подтверждения
   в браузере, вкладка «Прогоны» и проверка одного шага. Загружается до app.js; общие имена (el, api, room,
   META, draw …) используются только внутри функций. */

let RUN = null;                       // текущий запуск: {id, since, events, snap, timer}
let runInput = {task: "", extra: {}}; // входные данные: задача и свои поля (общие для запуска и проверки шага)
let lastRunState = {};                // чем закончился прошлый полный запуск (значения — текст)

// ── входные данные ───────────────────────────────────────────────────────────────
function collectRunParams() {
  const p = {};
  for (const [k, v] of Object.entries(room.defaults || {})) p[k] = v;
  for (const [k, v] of Object.entries(runInput.extra || {})) if (String(k).trim()) p[k] = v;
  p.task = runInput.task || "";
  return p;
}
function renderRunParams() {
  const host = $("#runParams"); host.replaceChildren();
  if (!room) { host.append(el("p", {class: "hint"}, "Откройте комнату.")); return; }
  host.append(el("div", {class: "lbl"}, "Задача (поле task)"));
  const t = el("textarea", {rows: 3, placeholder: "что должна сделать комната", spellcheck: "false"}); t.value = runInput.task || "";
  t.addEventListener("input", () => { runInput.task = t.value; });
  host.append(t);
  const defs = Object.keys(room.defaults || {});
  if (defs.length) host.append(el("p", {class: "hint"}, "Параметры комнаты: " + defs.join(", ") + " — значения по умолчанию берутся из её свойств."));
  host.append(el("div", {class: "lbl", style: "margin-top:10px"}, "Свои данные для запуска"));
  host.append(kvEditor(runInput.extra, (v) => { runInput.extra = v; }, {strings: true, keyPh: "поле", valPh: "значение (список — JSON)", addLabel: "Добавить поле"}));
  host.append(el("p", {class: "hint"}, "Запуск идёт здесь, по текущему состоянию схемы, без сохранения. Шаги с «Спросить вас» и вопросы придут сюда же."));
}

// ── ход запуска ──────────────────────────────────────────────────────────────────
function foldEvents(events, roomName) {
  const steps = [], open = {};
  for (const e of events) {
    const key = (e.room || "") + "|" + (e.node || "");
    if (e.type === "node_start") {
      const s = {node: e.node, room: e.room, kind: e.kind, status: "active", t: e.t, attempts: 0, branch: e.branch};
      steps.push(s); open[key] = s;
    } else if (e.type === "node_result" && open[key]) {
      open[key].output = e.output; open[key].changed = e.changed;
    } else if (e.type === "node_end" && open[key]) {
      Object.assign(open[key], {status: e.failed ? "failed" : "done", seconds: e.seconds});
    } else if (e.type === "node_error" && open[key]) {
      open[key].error = e.error; open[key].attempts = e.attempt;
    } else if (e.type === "ask") {
      const s = [...steps].reverse().find((x) => x.node === e.node && x.status === "active");
      if (s) { s.status = "waiting"; s.question = e.text; }
    }
  }
  return steps;
}
function runMapFrom(steps, roomName) {
  const m = {};
  for (const s of steps) if (!s.room || s.room === roomName) m[s.node] = s.status;
  return m;
}
const STATUS_WORDS = {running: "идёт", waiting: "ждёт вас", done: "готово", error: "ошибка", stopped: "остановлен"};
const KIND_NAME = {tool: "команда", agent: "агент", condition: "развилка", human_pause: "вопрос вам", set: "значение", notify: "уведомление", room: "подкомната",
  mcp: "MCP", skill: "скилл", plugin: "плагин", leeth: "Leeth", foreach: "для каждого", wait: "пауза", loop: "цикл", break: "break", continue: "continue",
  power: "питание ПК", hint: "подсказка", forget: "забыть", assert: "проверка", parallel: "параллельно", join: "слияние", git: "git", http: "запрос в сеть"};

function stepView(s, i) {
  const d = el("details", {class: "rstep " + s.status});
  d.append(el("summary", {}, el("span", {class: "dot"}), el("span", {class: "nm"}, `${i + 1}. ${s.node}`),
    el("span", {class: "muted"}, (s.branch ? "ветка " + s.branch + " · " : "") + (s.room && room && s.room !== room.room ? s.room + " · " : "") + (KIND_NAME[s.kind] || s.kind || "")),
    el("span", {class: "tm"}, s.status === "waiting" ? "ждёт вас" : s.seconds != null ? s.seconds + " с" : s.status === "active" ? "…" : "")));
  const body = el("div", {class: "rbody"});
  if (s.question) body.append(el("div", {class: "hint"}, "Вопрос: " + s.question));
  if (s.error) body.append(el("pre", {}, "Ошибка: " + s.error));
  if (s.output) body.append(el("div", {class: "lbl"}, "Вывод"), el("pre", {}, s.output));
  const ch = Object.entries(s.changed || {}).filter(([k]) => !k.endsWith(".output"));
  if (ch.length) {
    const tb = el("table");
    for (const [k, v] of ch) tb.append(el("tr", {}, el("td", {}, k), el("td", {}, v)));
    body.append(el("div", {class: "lbl"}, "Изменилось в состоянии"), tb);
  }
  if (!body.childNodes.length) body.append(el("div", {class: "hint"}, "Без вывода."));
  d.append(body);
  if (s.status === "failed") d.setAttribute("open", "");
  return d;
}
function outcomeView(snap) {
  const o = snap.outcome;
  if (!o && !snap.error) return null;
  const box = el("div", {class: "rout"});
  if (snap.error) box.append(el("b", {}, "Запуск остановился с ошибкой"), el("pre", {}, snap.error));
  if (o) {
    const exit = {done: "успешно", failed: "провал", budget: "вышел лимит", stuck: "некуда идти", human_pause: "ждёт вас", declined: "вы отказались", stopped: "остановлен", step: "шаг выполнен"}[o.exit] || o.exit;
    box.append(el("b", {}, `Итог: «${o.exit}» — ${exit}`), el("span", {class: "muted"}, ` · шагов: ${o.steps}` + (o.reason ? " · " + o.reason : "")));
    const st = Object.entries(o.state || {}).filter(([k]) => !k.includes(".") && !k.endsWith("_visits") && k !== "_steps");
    if (st.length) {
      const d = el("details", {class: "adv"}, el("summary", {}, `Состояние в конце (${st.length})`));
      const tb = el("table", {class: "rtab"});
      for (const [k, v] of st) tb.append(el("tr", {}, el("td", {}, k), el("td", {}, v)));
      d.append(el("div", {class: "body"}, tb));
      box.append(d);
    }
  }
  return box;
}
function askView(snap, onAnswer) {
  const p = snap.prompt;
  if (!p) return null;
  const box = el("div", {class: "rask"});
  if (p.kind === "confirm") {
    box.append(el("b", {}, `Шаг «${p.node}» просит разрешения`), el("div", {}, p.text));
    box.append(el("div", {class: "row"},
      el("button", {class: "btn primary sm", onclick: () => onAnswer(p, true)}, "Да, выполнить"),
      el("button", {class: "btn sm", onclick: () => onAnswer(p, false)}, "Нет")));
  } else if ((p.options || []).length) {
    box.append(el("b", {}, `Вопрос от шага «${p.node}»`), el("div", {}, p.text),
      el("div", {class: "row optrow"}, p.options.map((o) => el("button", {class: "btn primary sm", onclick: () => onAnswer(p, o)}, o))));
  } else {
    const inpt = el("input", {type: "text", placeholder: "ваш ответ"});
    const send = () => onAnswer(p, inpt.value);
    inpt.addEventListener("keydown", (ev) => { if (ev.key === "Enter") send(); });
    box.append(el("b", {}, `Вопрос от шага «${p.node}»`), el("div", {}, p.text), inpt,
      el("div", {class: "row"}, el("button", {class: "btn primary sm", onclick: send}, "Ответить")));
    setTimeout(() => inpt.focus(), 0);
  }
  return box;
}

function paintRun() {
  const snap = RUN && RUN.snap;
  const log = $("#runLog"); log.replaceChildren();
  const chip = $("#runStatus");
  if (!snap) { chip.className = "chip"; chip.textContent = "не запускалась"; $("#runStop").disabled = true; return; }
  chip.className = "chip run-" + snap.status;
  chip.textContent = STATUS_WORDS[snap.status] || snap.status;
  $("#runStop").disabled = !(snap.status === "running" || snap.status === "waiting");
  $("#runGo").disabled = snap.status === "running" || snap.status === "waiting";
  const ask = askView(snap, runAnswer);
  if (ask) log.append(ask);
  const steps = foldEvents(RUN.events, room && room.room);
  steps.forEach((s, i) => log.append(stepView(s, i)));
  const out = outcomeView(snap);
  if (out) log.append(out);
  if (!steps.length && !ask && !out) log.append(el("p", {class: "hint"}, "Идёт подготовка…"));
  runMap = runMapFrom(steps, room && room.room);
  if (typeof draw === "function" && view === "graph") draw();
  log.scrollTop = log.scrollHeight;
}

async function runStart() {
  if (!room) { say("Сначала откройте или создайте комнату.", "bad"); return; }
  runOpen(false);
  const {status, data} = await api("/api/run", {room: cleaned(room), params: collectRunParams()});
  if (status !== 200 || !data.ok) {
    const errs = (data.errors || [data.error || "HTTP " + status]).map(explain);
    const log = $("#runLog"); log.replaceChildren(el("div", {class: "rask"}, el("b", {}, "Запуск не начат: комната пока не годится"),
      el("ul", {}, errs.map((e) => el("li", {}, e.text + (e.fix ? " — " + e.fix : ""))))));
    return;
  }
  RUN = {id: data.run, since: 0, events: [], snap: null, timer: null};
  runMap = {};
  paintRun();
  RUN.timer = setInterval(runPoll, 350);
  runPoll();
}
async function runPoll() {
  if (!RUN) return;
  const {status, data} = await api(`/api/run?id=${encodeURIComponent(RUN.id)}&since=${RUN.since}`);
  if (status !== 200 || !RUN) return;
  RUN.events.push(...(data.events || []));
  RUN.since = data.next;
  RUN.snap = data;
  paintRun();
  if (data.status !== "running" && data.status !== "waiting") {
    clearInterval(RUN.timer); RUN.timer = null;
    if (data.outcome && data.outcome.state) lastRunState = data.outcome.state;
    if (view === "runs") renderRuns();
  }
}
async function runAnswer(prompt, value) {
  if (!RUN) return;
  await api("/api/answer", {run: RUN.id, prompt: prompt.id, answer: value});
  runPoll();
}
async function runStop() {
  if (RUN) { await api("/api/stop", {run: RUN.id}); runPoll(); }
}
function runOpen(go) {
  $("#runDock").hidden = false;
  renderRunParams();
  paintRun();
  if (go) runStart();
}
function initRun() {                    // вызывается из app.js, когда $ и разметка уже есть
  $("#runGo").addEventListener("click", runStart);
  $("#runStop").addEventListener("click", runStop);
  $("#runClose").addEventListener("click", () => { $("#runDock").hidden = true; });
  $("#runClear").addEventListener("click", () => { runMap = {}; draw(); });
}

// ── вкладка «Прогоны» ────────────────────────────────────────────────────────────
async function renderRuns() {
  const host = $("#runsBody"); host.replaceChildren(el("h2", {}, "Прогоны"));
  const {data} = await api("/api/runs");
  const runs = (data && data.runs) || [];
  if (!runs.length) { host.append(el("p", {class: "muted"}, "Пока ничего не запускали. Кнопка «Запустить» вверху или Ctrl+Enter.")); return; }
  host.append(el("p", {class: "hint"}, "Это запуски из редактора за время работы страницы. Нажмите, чтобы посмотреть ход по шагам."));
  const list = el("div", {class: "rl"});
  for (const r of runs) {
    const started = new Date(r.started * 1000).toLocaleTimeString();
    list.append(el("button", {class: "ritem", onclick: () => showRun(r.id)},
      el("span", {class: "chip run-" + r.status}, STATUS_WORDS[r.status] || r.status),
      el("b", {}, r.room + (r.kind === "node" ? " · один шаг" : "")),
      el("span", {class: "muted"}, `${started} · ${r.seconds} с · шагов: ${r.steps}` + (r.exit ? ` · выход «${r.exit}»` : "")),
      r.error ? el("span", {class: "muted"}, short(r.error, 60)) : null));
  }
  host.append(list);
  host.append(el("div", {id: "runDetail"}));
}
async function showRun(id) {
  const {status, data} = await api("/api/run?id=" + encodeURIComponent(id));
  const host = $("#runDetail"); host.replaceChildren();
  if (status !== 200) { host.append(el("p", {}, "Прогон не найден.")); return; }
  host.append(el("h3", {}, `${data.room} · ${STATUS_WORDS[data.status] || data.status}`));
  if (Object.keys(data.params || {}).length) host.append(el("p", {class: "hint"}, "Данные запуска: " + Object.entries(data.params).map(([k, v]) => `${k}=${short(v, 40)}`).join("; ")));
  const steps = foldEvents(data.events || [], data.room);
  steps.forEach((s, i) => host.append(stepView(s, i)));
  const out = outcomeView(data);
  if (out) host.append(out);
  host.append(el("div", {class: "row", style: "margin-top:8px"},
    el("button", {class: "btn sm", onclick: () => { runMap = runMapFrom(steps, room && room.room); setView("graph"); }}, "Показать на схеме")));
  host.scrollIntoView({block: "nearest"});
}

// ── проверка одного шага ─────────────────────────────────────────────────────────
const NOT_ALONE = new Set(["human_pause", "room", "spawn_room", "condition", "break", "continue"]);
function stepTestBox(side, n) {
  if (NOT_ALONE.has(n.type)) return;
  const d = el("details", {class: "adv"}, el("summary", {}, "Проверить этот шаг"));
  const body = el("div", {class: "body"});
  body.append(el("p", {class: "hint"}, "Выполнит только этот шаг, с данными прошлого запуска и вашими полями. Шаги, которые действуют (команды, MCP, плагины, Leeth, уведомления вне консоли, агенты), сначала спросят."));
  body.append(kvEditor(runInput.extra, (v) => { runInput.extra = v; }, {strings: true, keyPh: "поле", valPh: "тестовое значение", addLabel: "Добавить тестовое поле"}));
  const result = el("div", {class: "stepbox", hidden: true});
  const go = el("button", {class: "btn primary sm", onclick: () => runStepOnce(n, result, go)}, icon("i-play", "sm"), "Выполнить шаг");
  body.append(el("div", {class: "row", style: "margin:8px 0"}, go), result);
  d.append(body);
  side.append(d);
}
async function runStepOnce(n, box, button) {
  box.hidden = false; box.replaceChildren(el("p", {class: "loading"}, "Выполняю…"));
  button.disabled = true;
  const params = {...lastRunState, ...collectRunParams()};
  const {status, data} = await api("/api/run_node", {room: cleaned(room), node: n.id, params});
  if (status !== 200 || !data.ok) {
    box.replaceChildren(el("p", {}, (data.errors || [data.error || "HTTP " + status]).map((e) => explain(e).text).join("; ")));
    button.disabled = false;
    return;
  }
  const id = data.run;
  for (let i = 0; i < 600; i++) {                // до ~3 минут
    const r = await api("/api/run?id=" + encodeURIComponent(id));
    const snap = r.data;
    if (r.status !== 200) break;
    box.replaceChildren();
    if (snap.prompt) {
      const ask = askView(snap, async (p, v) => { await api("/api/answer", {run: id, prompt: p.id, answer: v}); });
      box.append(ask);
    } else if (snap.status !== "running") {
      const steps = foldEvents(snap.events || [], room.room);
      const s = steps[0];
      if (snap.status === "error") box.append(el("pre", {}, "Ошибка: " + snap.error));
      else if (snap.outcome && snap.outcome.exit === "declined") box.append(el("p", {}, "Вы отказались — шаг не выполнялся."));
      else if (s) {
        box.append(el("div", {}, el("b", {}, s.status === "failed" ? "Не получилось" : "Готово"), el("span", {class: "muted"}, s.seconds != null ? ` · ${s.seconds} с` : "")));
        if (s.output) box.append(el("pre", {}, s.output));
        const ch = Object.entries(s.changed || {}).filter(([k]) => !k.endsWith(".output"));
        if (ch.length) { const tb = el("table"); for (const [k, v] of ch) tb.append(el("tr", {}, el("td", {}, k), el("td", {}, v))); box.append(tb); }
      }
      button.disabled = false;
      return;
    } else box.append(el("p", {class: "loading"}, "Выполняю…"));
    await new Promise((r2) => setTimeout(r2, 300));
  }
  button.disabled = false;
}
