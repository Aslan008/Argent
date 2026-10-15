"use strict";
/* Расширения редактора комнат: справочники (инструменты, MCP, скиллы, плагины, Leeth), формы блоков,
   палитра с поиском, правила комнаты. Загружается ДО app.js; имена из app.js (el, field, inp, room, META …)
   используются только внутри функций, то есть когда они уже определены. */

// ── справочники с сервера ─────────────────────────────────────────────────────────
const CAT = {mcpServers: null, mcpTools: {}, mcpState: {}, skills: null, headings: {}, plugins: null};

async function loadCatalogs() {
  const [m, s, p] = await Promise.all([api("/api/mcp"), api("/api/skills"), api("/api/plugins")]);
  CAT.mcpServers = m.status === 200 ? (m.data.servers || {}) : {};
  CAT.skills = s.status === 200 ? (s.data.skills || []) : [];
  CAT.plugins = p.status === 200 ? (p.data.plugins || {}) : {};
}
async function loadMcpTools(server, force) {
  if (!server) return;
  if (!force && (CAT.mcpTools[server] || CAT.mcpState[server] === "loading")) return;
  CAT.mcpState[server] = "loading";
  const {status, data} = await api("/api/mcp?server=" + encodeURIComponent(server));
  if (status === 200) { CAT.mcpTools[server] = data.tools || []; CAT.mcpState[server] = "ok"; }
  else { CAT.mcpState[server] = data.error || ("HTTP " + status); }
}
async function loadHeadings(name) {
  if (!name || CAT.headings[name]) return;
  const {status, data} = await api("/api/skills?name=" + encodeURIComponent(name));
  CAT.headings[name] = status === 200 ? (data.headings || []) : [];
}
const VOC = () => Object.assign({loop_modes: [], power_modes: [], hint_modes: [], forget_modes: [], memory_modes: [],
  sampling: {}, reasoning_efforts: [], loop_default_max: 100, loop_max: 1000, power_min_delay: 10, power_max_delay: 3600, extra_derived: []},
  (typeof META !== "undefined" && META.vocab) || {});
const toolInfo = (name) => ((typeof META !== "undefined" && META.tool_info) || []).find((t) => t.name === name);
const fieldNames = () => [...knownFields().keys()];

// ── подписи ───────────────────────────────────────────────────────────────────────
const LOOP_LABEL = {
  for: ["Счётчик (for)", "от начала до конца с шагом: i = 0, 1, 2 …"],
  times: ["Повторить N раз", "просто N кругов, счётчик с нуля"],
  while: ["Пока (while)", "условие проверяется до каждого круга; может не выполниться ни разу"],
  do_while: ["Делать, пока (do-while)", "круг выполнится хотя бы раз, потом условие проверяется"],
  until: ["Повторять, пока не (repeat-until)", "круг выполнится хотя бы раз и повторяется, пока условие не станет верным"],
  for_in: ["По ключам (for-in)", "по словарю: ключ и значение; по списку: номер и элемент"],
};
const POWER_LABEL = {
  shutdown: ["Выключить ПК", "через заданное время; можно отменить"],
  restart: ["Перезагрузить ПК", "через заданное время; можно отменить"],
  sleep: ["Спящий режим", "сразу"],
  hibernate: ["Гибернация", "сразу (нет на macOS)"],
  lock: ["Заблокировать экран", "сразу"],
  logout: ["Выйти из сессии", "сразу; несохранённое в других программах пропадёт"],
  cancel: ["Отменить выключение", "отменяет запланированное выключение или перезагрузку"],
};
const HINT_LABEL = {
  next: ["Следующему агенту", "прозвучит один раз, перед ближайшим шагом-агентом"],
  all: ["Всем агентам до отмены", "будет сопровождать каждый шаг-агент, пока не снимете"],
  clear: ["Снять подсказки", "убирает «всем агентам» (или только выбранным)"],
};
const FORGET_LABEL = {
  all: ["Память и подсказки", "агенты начинают с чистого листа"],
  memory: ["Только память", "что делали прошлые шаги"],
  hints: ["Только подсказки", "убрать накопленные подсказки"],
};
const SAMPLING_LABEL = {
  temperature: ["Температура", "0 — строго и повторяемо, выше — разнообразнее", 0.1],
  top_p: ["top_p", "доля вероятности, из которой выбирается слово (0–1)", 0.05],
  top_k: ["top_k", "из скольких самых вероятных слов выбирать", 1],
  min_p: ["min_p", "отсечь слова, вероятность которых ниже доли от лучшего (0–1)", 0.01],
  repeat_penalty: ["Штраф за повторы", "1 — без штрафа, 1.1–1.2 — против зацикливания текста", 0.05],
  thinking_budget: ["Бюджет рассуждения", "сколько токенов модель может потратить на размышление", 100],
};

// ── поля по схеме инструмента ────────────────────────────────────────────────────
const LONG_PARAMS = /^(command|code|text|content|query|instruction|prompt|task|description|criterion|items|message|body|script|input|questions)$/;
function scalarOrTemplate(v) {
  const s = String(v).trim();
  if (s === "") return "";
  return /^-?\d+(\.\d+)?$/.test(s) ? Number(s) : s;
}
function paramForm(params, args, onChange, o = {}) {
  const box = el("div", {class: "params"});
  const known = new Set(params.map((p) => p.name));
  const set = (name, v) => {
    if (v === "" || v == null || (Array.isArray(v) && !v.length)) delete args[name]; else args[name] = v;
    onChange(args);
  };
  for (const p of params) {
    if (o.skip && o.skip.includes(p.name)) continue;
    const cur = args[p.name];
    const label = p.name + (p.required ? " *" : "");
    const hintText = p.description || "";
    if (p.enum) {
      box.append(field(label, select([["", p.required ? "— выберите —" : "— не задавать —"], ...p.enum.map((x) => [String(x), String(x)])],
        cur == null ? "" : String(cur), (v) => set(p.name, v)), hintText));
    } else if (p.type === "boolean") {
      box.append(check(label, cur ?? p.default ?? false, (v) => set(p.name, v), hintText));
    } else if (p.type === "integer" || p.type === "number") {
      box.append(field(label, inp(cur ?? "", (v) => set(p.name, scalarOrTemplate(v)), {ph: p.default != null ? String(p.default) : "число или {{поле}}"}), hintText));
    } else if (p.type === "array" || p.type === "object") {
      box.append(field(label, jsonArea(cur ?? (p.type === "array" ? [] : {}), (v) => set(p.name, v), 3), hintText || "JSON"));
    } else {
      const long = LONG_PARAMS.test(p.name) || (typeof cur === "string" && cur.includes("\n"));
      const ctl = long ? area(cur ?? "", (v) => set(p.name, v), 3, p.name === "command" || p.name === "code")
                       : inp(cur ?? "", (v) => set(p.name, v), {ph: p.default != null ? String(p.default) : ""});
      box.append(field(label, withInserter(ctl), hintText));
    }
  }
  const extra = {};
  for (const [k, v] of Object.entries(args || {})) if (!known.has(k) && !(o.skip || []).includes(k)) extra[k] = v;
  if (Object.keys(extra).length || o.allowExtra) {
    box.append(field("Прочие параметры", kvEditor(extra, (v) => {
      for (const k of Object.keys(args)) if (!known.has(k) && !(o.skip || []).includes(k)) delete args[k];
      Object.assign(args, v); onChange(args);
    }, {keyPh: "параметр", valPh: "значение", addLabel: "Добавить параметр"})));
  }
  return box;
}

// ── «Запомнить из результата» ─────────────────────────────────────────────────────
function setsEditor(sets, onChange) {
  const V = VOC();
  const base = [...(META.derived || []), ...(V.extra_derived || [])];
  const label = (v) => DERIVED_LABEL[v] || v;
  const rows = Object.entries(sets || {}).map(([k, v]) => {
    const s = String(v);
    if (s.startsWith("json:")) return {name: k, spec: "json:", arg: s.slice(5)};
    if (s.startsWith("regex:")) return {name: k, spec: "regex:", arg: s.slice(6)};
    return {name: k, spec: s, arg: ""};
  });
  const box = el("div", {class: "kv"});
  const commit = () => {
    const out = {};
    for (const r of rows) {
      const n = r.name.trim();
      if (!n) continue;
      out[n] = r.spec === "json:" ? "json:" + (r.arg || "").trim() : r.spec === "regex:" ? "regex:" + (r.arg || "") : r.spec;
    }
    onChange(out);
  };
  const render = () => {
    box.replaceChildren();
    rows.forEach((r, i) => {
      const opts = [...base.map((d) => [d, label(d)]), ["json:", "значение внутри JSON (по пути)"], ["regex:", "по шаблону (regex)"]];
      const row = el("div", {class: "kvrow"},
        inp(r.name, (v) => { r.name = v; commit(); }, {ph: "поле, напр. found", list: "fieldsList"}),
        select(opts, r.spec, (v) => { r.spec = v; commit(); render(); }),
        el("button", {class: "btn icon ghost sm", title: "Убрать", "aria-label": "Убрать", onclick: () => { rows.splice(i, 1); commit(); render(); }}, icon("i-close", "sm")));
      box.append(row);
      if (r.spec === "json:" || r.spec === "regex:") {
        box.append(el("div", {class: "kvrow"}, inp(r.arg, (v) => { r.arg = v; commit(); },
          {ph: r.spec === "json:" ? "путь, напр. yes или a.b.0" : "шаблон с группой, напр. id=(\\d+)"})));
      }
    });
    box.append(el("div", {}, el("button", {class: "btn sm", onclick: () => { rows.push({name: "", spec: (META.derived || ["output"])[0] === "ok" ? "output" : "output", arg: ""}); render(); }},
      icon("i-plus", "sm"), "Запомнить поле")));
  };
  render();
  return box;
}

// ── что игнорировать ─────────────────────────────────────────────────────────────
const IGN_KINDS = [["mcp", "Сервер MCP целиком"], ["mcp-tool", "Инструмент MCP"], ["skill", "Скилл целиком"], ["skill-sec", "Раздел скилла"]];
function parseIgnore(rule) {
  const s = String(rule);
  if (s.startsWith("mcp:")) {
    const body = s.slice(4), dot = body.indexOf(".");
    return dot < 0 ? {kind: "mcp", a: body, b: ""} : {kind: "mcp-tool", a: body.slice(0, dot), b: body.slice(dot + 1)};
  }
  if (s.startsWith("skill:")) {
    const body = s.slice(6), h = body.indexOf("#");
    return h < 0 ? {kind: "skill", a: body, b: ""} : {kind: "skill-sec", a: body.slice(0, h), b: body.slice(h + 1)};
  }
  return {kind: "mcp", a: s, b: ""};
}
function buildIgnore(r) {
  if (!r.a.trim()) return "";
  if (r.kind === "mcp") return "mcp:" + r.a.trim();
  if (r.kind === "mcp-tool") return r.b.trim() ? "mcp:" + r.a.trim() + "." + r.b.trim() : "mcp:" + r.a.trim();
  if (r.kind === "skill") return "skill:" + r.a.trim();
  return r.b.trim() ? "skill:" + r.a.trim() + "#" + r.b.trim() : "skill:" + r.a.trim();
}
function ignoreEditor(rules, onChange, o = {}) {
  const kinds = IGN_KINDS.filter(([k]) => (k.startsWith("mcp") ? o.mcp !== false : o.skill !== false));
  const rows = (rules || []).map(parseIgnore);
  const box = el("div", {class: "kv"});
  const commit = () => onChange(rows.map(buildIgnore).filter(Boolean));
  const render = () => {
    box.replaceChildren();
    rows.forEach((r, i) => {
      const a = el("input", {type: "text", value: r.a, placeholder: r.kind.startsWith("mcp") ? "сервер (можно *)" : "скилл", list: r.kind.startsWith("mcp") ? "mcpServerList" : "skillList"});
      a.addEventListener("input", () => { r.a = a.value; commit(); });
      const b = el("input", {type: "text", value: r.b, placeholder: r.kind === "mcp-tool" ? "инструмент (delete_*, *write*)" : "заголовок раздела",
        list: r.kind === "skill-sec" ? "headingList" : null});
      b.addEventListener("input", () => { r.b = b.value; commit(); });
      if (r.kind === "skill-sec" && r.a) loadHeadings(r.a).then(() => fillHeadingList(r.a));
      box.append(el("div", {class: "ruleline"}, select(kinds, r.kind, (v) => { r.kind = v; if (!v.endsWith("tool") && !v.endsWith("sec")) r.b = ""; commit(); render(); }),
        a, (r.kind === "mcp-tool" || r.kind === "skill-sec") ? b : null,
        el("button", {class: "btn icon ghost sm", title: "Убрать", "aria-label": "Убрать правило", onclick: () => { rows.splice(i, 1); commit(); render(); }}, icon("i-close", "sm"))));
    });
    box.append(el("div", {}, el("button", {class: "btn sm", onclick: () => { rows.push({kind: kinds[0][0], a: "", b: ""}); render(); }},
      icon("i-plus", "sm"), "Добавить правило")));
  };
  render();
  return box;
}
function fillHeadingList(skill) {
  const dl = $("#headingList"); if (!dl) return;
  dl.replaceChildren();
  for (const h of CAT.headings[skill] || []) dl.append(el("option", {value: h.title}));
}
function ensureLists() {
  const mk = (id) => { let d = document.getElementById(id); if (!d) { d = el("datalist", {id}); document.body.append(d); } return d; };
  const s = mk("mcpServerList"); s.replaceChildren();
  for (const k of Object.keys(CAT.mcpServers || {})) s.append(el("option", {value: k}));
  const k = mk("skillList"); k.replaceChildren();
  for (const x of CAT.skills || []) k.append(el("option", {value: x.name}, x.description));
  mk("headingList");
}

// ── параметры модели ─────────────────────────────────────────────────────────────
function samplingEditor(sampling, onChange, o = {}) {
  const V = VOC();
  const cur = {...(sampling || {})};
  const box = el("div", {class: "rulebox"});
  const commit = () => onChange({...cur});
  const row = (key, [title, why, stepSize]) => {
    const [lo, hi] = (V.sampling && V.sampling[key]) || [0, 1];
    const on = key in cur;
    const num = el("input", {type: "number", min: lo, max: hi, step: stepSize, value: on ? cur[key] : "", placeholder: o.inherit && o.inherit[key] != null ? "из комнаты: " + o.inherit[key] : "по умолчанию", disabled: !on});
    num.addEventListener("input", () => { if (num.value === "") delete cur[key]; else cur[key] = Number(num.value); commit(); });
    const chk = el("input", {type: "checkbox"}); chk.checked = on;
    chk.addEventListener("change", () => {
      num.disabled = !chk.checked;
      if (chk.checked) { const d = o.inherit && o.inherit[key] != null ? o.inherit[key] : (key === "temperature" ? 0.3 : key === "top_p" ? 0.9 : key === "top_k" ? 40 : key === "min_p" ? 0.05 : key === "repeat_penalty" ? 1.1 : 2000); num.value = d; cur[key] = d; }
      else { delete cur[key]; num.value = ""; }
      commit();
    });
    return el("div", {class: "field"}, el("label", {class: "check"}, chk, el("span", {}, title, el("small", {}, why + ` · допустимо ${lo}…${hi}`))), num);
  };
  for (const [key, meta] of Object.entries(SAMPLING_LABEL)) box.append(row(key, meta));
  const eff = el("select");
  for (const [v, l] of [["", "по умолчанию"], ...(V.reasoning_efforts || []).map((x) => [x, x])]) { const op = el("option", {value: v}, l); if ((cur.reasoning_effort || "") === v) op.selected = true; eff.append(op); }
  eff.addEventListener("change", () => { if (eff.value) cur.reasoning_effort = eff.value; else delete cur.reasoning_effort; commit(); });
  box.append(field("Уровень рассуждений", eff, "none — без рассуждений, xhigh — самое глубокое. Не все модели это поддерживают: тогда параметр игнорируется."));
  box.append(el("p", {class: "hint"}, "Отмеченные значения действуют только пока работает этот шаг; ваши настройки Argent не меняются."));
  return box;
}

// ── общие свойства шага ──────────────────────────────────────────────────────────
function commonProps(side, n, upd) {
  side.append(check("Отключить шаг (пропускать при запуске)", n.disabled, (v) => { n.disabled = v; upd("dis" + n.id); draw(); }, "Схема остаётся как есть, шаг просто обходится — удобно для проверок."));
  side.append(field("Заметка", area(n.note, (v) => { n.note = v; touch("note" + n.id); draw(); }, 2), "Для того, кто будет читать комнату. На работу не влияет."));
}

// ── формы блоков ─────────────────────────────────────────────────────────────────
function fieldPick(label, value, onChange, hintText, ph) {
  return field(label, inp(value, onChange, {ph: ph || "поле, напр. posts", list: "fieldsList"}), hintText);
}
function modeCards(map, current, onPick) {
  const wrap = el("div");
  const render = () => {
    wrap.replaceChildren();
    for (const [k, [title, why]] of Object.entries(map)) {
      wrap.append(el("button", {class: "modecard", "aria-pressed": String(k === current), onclick: () => { current = k; onPick(k); render(); }},
        el("b", {}, title), el("span", {}, why)));
    }
  };
  render();
  return wrap;
}

function toolNodeForm(side, n, upd) {
  const info = (META.tool_info || []);
  const forbidden = new Set(META.not_in_rooms || []);
  const groups = META.tool_groups || {};
  const byGroup = {};
  for (const t of info) { if (forbidden.has(t.name) && t.name !== n.tool) continue; (byGroup[t.group] = byGroup[t.group] || []).push(t); }
  const sel = el("select");
  for (const [g, title] of Object.entries(groups)) {
    if (!byGroup[g]) continue;
    const og = el("optgroup", {label: title});
    for (const t of byGroup[g]) { const op = el("option", {value: t.name}, t.name + (t.mutating ? "  ⚠ меняет" : "")); if (t.name === n.tool) op.selected = true; og.append(op); }
    sel.append(og);
  }
  if (n.tool && !info.some((t) => t.name === n.tool)) sel.prepend(el("option", {value: n.tool, selected: true}, n.tool + " (нет в списке)"));
  sel.addEventListener("change", () => { n.tool = sel.value; n.args = {}; touch("tool", true); });
  side.append(field("Инструмент", sel));
  const ti = toolInfo(n.tool);
  if (ti) {
    if (ti.description) side.append(el("p", {class: "hint"}, short(ti.description, 320)));
    if (ti.mutating) side.append(el("div", {}, el("span", {class: "warnchip"}, "меняет файлы или внешний мир"), n.confirm ? null : el("span", {class: "warnchip"}, "лучше включить «Спросить вас»")));
    n.args = n.args || {};
    side.append(paramForm(ti.params || [], n.args, () => upd("args" + n.id), {allowExtra: true}));
  } else {
    n.args = n.args || {};
    side.append(field("Параметры инструмента", kvEditor(n.args, (v) => { n.args = v; upd("args" + n.id); }, {keyPh: "параметр", valPh: "значение"})));
  }
  side.append(field("Запомнить из результата", setsEditor(n.sets, (v) => { n.sets = v; upd("sets" + n.id); }),
    "Вердикт даёт процесс, а не модель. Потом в условии: state.<поле>. Для списков — «строки» и «JSON»."));
  side.append(check("Спросить вас перед запуском", n.confirm, (v) => { n.confirm = v; upd("confirm"); }, "Нет «да» — шаг не выполнится."));
  side.append(el("details", {class: "adv"}, el("summary", {}, "Дополнительно"), el("div", {class: "body"},
    check("Можно безопасно повторить после обрыва", n.idempotent !== false, (v) => { n.idempotent = v; touch("idem"); }, "Снимите для команд, меняющих мир."),
    field("Повторов при сбое", inp(n.retries || 0, (v) => { n.retries = v || 0; touch("ret"); }, {type: "number"}), "Только если инструмент выбросил ошибку; код возврата 1 — это данные, а не сбой."),
    fieldPick("Взять значение из поля", n.input_from, (v) => { n.input_from = v; touch("inf"); }, "Попадёт в первый обязательный параметр, который вы не заполнили.", "state.поле"))));
}

function mcpNodeForm(side, n, upd) {
  const servers = Object.keys(CAT.mcpServers || {});
  const state = CAT.mcpState[n.server];
  side.append(field("Сервер MCP", el("div", {}, inp(n.server, (v) => { n.server = v; upd("srv" + n.id); }, {ph: "имя сервера", list: "mcpServerList"})),
    servers.length ? "Настроены: " + servers.map((s) => s + (CAT.mcpServers[s].running ? " ●" : "")).join(", ") + ". Остальные добавляются командой /mcp." : "Серверы не настроены: добавьте их командой /mcp."));
  if (n.server && servers.includes(n.server) && !CAT.mcpTools[n.server] && state !== "loading") loadMcpTools(n.server).then(() => panel());
  side.append(field("Что сделать", select([["call", "Вызвать инструмент"], ["list", "Получить список инструментов сервера"]], n.mode || "call",
    (v) => { n.mode = v === "call" ? undefined : v; touch("mcpmode", true); })));
  if ((n.mode || "call") === "call") {
    const tools = CAT.mcpTools[n.server];
    const tInfo = (tools || []).find((t) => t.name === n.tool);
    const row = el("div", {class: "row"});
    if (tools && tools.length) {
      row.append(select([["", "— выберите инструмент —"], ...tools.map((t) => [t.name, t.name])], n.tool || "", (v) => { n.tool = v; n.args = {}; touch("mcptool", true); }));
    } else {
      row.append(inp(n.tool, (v) => { n.tool = v; upd("mtool" + n.id); }, {ph: "имя инструмента"}));
    }
    row.append(el("button", {class: "btn sm", title: "Подключиться к серверу и получить его инструменты", style: "flex:0 0 auto",
      onclick: async () => { await loadMcpTools(n.server, true); panel(); }}, "Загрузить"));
    side.append(field("Инструмент", row));
    if (state === "loading") side.append(el("p", {class: "loading"}, "Подключаюсь к серверу…"));
    else if (state && state !== "ok") side.append(el("p", {class: "hint"}, "Не удалось получить инструменты: " + state + ". Имя можно ввести вручную."));
    if (tInfo) {
      if (tInfo.description) side.append(el("p", {class: "hint"}, short(tInfo.description, 300)));
      n.args = n.args || {};
      side.append(paramForm(tInfo.params || [], n.args, () => upd("args" + n.id), {allowExtra: true}));
    } else {
      n.args = n.args || {};
      side.append(field("Аргументы", kvEditor(n.args, (v) => { n.args = v; upd("args" + n.id); }, {keyPh: "аргумент", valPh: "значение или {{поле}}"})));
    }
    side.append(fieldPick("Подставить значение из поля", n.input_from, (v) => { n.input_from = v; touch("inf"); }, "Попадёт в первый обязательный аргумент, который вы не заполнили.", "state.поле"));
  } else {
    side.append(field("Фильтр по имени", inp((n.args || {}).filter || "", (v) => { n.args = v ? {filter: v} : {}; upd("mf" + n.id); }, {ph: "необязательно"})));
  }
  side.append(field("Запомнить из результата", setsEditor(n.sets, (v) => { n.sets = v; upd("sets" + n.id); }), "Результат сервера — текст; «JSON» разберёт его в список или словарь."));
  side.append(check("Спросить вас перед вызовом", n.confirm, (v) => { n.confirm = v; upd("confirm"); }, "Внешний сервер может что угодно менять — по умолчанию лучше спрашивать."));
  side.append(el("details", {class: "adv"}, el("summary", {}, "Что игнорировать и дополнительно"), el("div", {class: "body"},
    field("Игнорировать", ignoreEditor(n.ignore, (v) => { n.ignore = v; touch("ign" + n.id); }, {skill: false}), "Правило «mcp:сервер.инструмент*» запретит такой вызов на самом деле, а не просто попросит модель."),
    check("Можно безопасно повторить после обрыва", n.idempotent === true, (v) => { n.idempotent = v; touch("idem"); }, "Включайте только для чтения."),
    field("Повторов при сбое", inp(n.retries || 0, (v) => { n.retries = v || 0; touch("ret"); }, {type: "number"})))));
}

function skillNodeForm(side, n, upd) {
  const names = (CAT.skills || []).map((s) => s.name);
  const desc = (CAT.skills || []).find((s) => s.name === n.skill);
  side.append(field("Скилл", inp(n.skill, (v) => { n.skill = v; upd("skill" + n.id); }, {ph: "имя скилла", list: "skillList"}),
    desc ? desc.description : names.length ? "Установлено скиллов: " + names.length : "Скиллы не найдены."));
  if (n.skill && names.includes(n.skill) && !CAT.headings[n.skill]) loadHeadings(n.skill).then(() => panel());
  const heads = CAT.headings[n.skill] || [];
  if (heads.length) {
    side.append(el("div", {class: "lbl"}, "Не читать разделы"));
    const box = el("div", {class: "tools"});
    for (const h of heads) {
      const rule = "skill:" + n.skill + "#" + h.title;
      const c = el("input", {type: "checkbox"}); c.checked = (n.ignore || []).includes(rule);
      c.addEventListener("change", () => { n.ignore = (n.ignore || []).filter((r) => r !== rule); if (c.checked) n.ignore.push(rule); touch("ign" + n.id); });
      box.append(el("label", {class: "check"}, c, el("span", {style: `padding-left:${(h.level - 1) * 10}px`}, h.title)));
    }
    side.append(box);
  }
  side.append(field("Запомнить из результата", setsEditor(n.sets, (v) => { n.sets = v; upd("sets" + n.id); }), "Текст скилла лежит в поле «<шаг>.output»; здесь можно взять из него число, строки или JSON."));
  side.append(el("details", {class: "adv"}, el("summary", {}, "Правила игнорирования"), el("div", {class: "body"},
    ignoreEditor(n.ignore, (v) => { n.ignore = v; touch("ign" + n.id); }, {mcp: false}))));
}

function pluginNodeForm(side, n, upd) {
  const plugins = CAT.plugins || {};
  const mode = n.mode || "command";
  side.append(field("Что сделать", select([["command", "Выполнить команду плагина"], ["hook", "Вызвать событие (хук)"]], mode, (v) => { n.mode = v === "command" ? undefined : v; n.command = ""; touch("plmode", true); })));
  const pn = Object.keys(plugins);
  side.append(field("Плагин (необязательно)", select([["", "любой"], ...pn.map((p) => [p, p])], n.plugin || "", (v) => { n.plugin = v || undefined; touch("plg", true); })));
  const all = [];
  for (const [p, d] of Object.entries(plugins)) {
    if (n.plugin && n.plugin !== p) continue;
    for (const c of (mode === "hook" ? d.hooks : d.commands) || []) all.push(c);
  }
  const dl = el("datalist", {id: "pluginCmdList"}); for (const c of all) dl.append(el("option", {value: c}));
  side.append(dl);
  side.append(field(mode === "hook" ? "Событие" : "Команда", inp(n.command, (v) => { n.command = v; upd("pcmd" + n.id); }, {ph: "имя", list: "pluginCmdList"}),
    all.length ? "Доступно: " + all.join(", ") : "Плагины не загружены."));
  n.args = n.args || {};
  side.append(field("Аргументы", kvEditor(n.args, (v) => { n.args = v; upd("args" + n.id); }, {strings: true, keyPh: "имя", valPh: "значение или {{поле}}"}),
    mode === "command" ? "Команде отдаются значения по порядку, как слова после команды." : "Передаются хуку как именованные аргументы."));
  side.append(field("Запомнить из результата", setsEditor(n.sets, (v) => { n.sets = v; upd("sets" + n.id); })));
  side.append(check("Спросить вас перед запуском", n.confirm, (v) => { n.confirm = v; upd("confirm"); }, "Плагин — это код: он может сделать что угодно."));
}

const LEETH_MODE_LABEL = {
  delegate: ["Поручить задачу", "Leeth выполнит рутинное действие или цепочку (прочитать файл, grep, команда)"],
  filter: ["Отобрать по вопросу", "ответит «да/нет» про каждый элемент списка за миллисекунды"],
  classify: ["Разложить по категориям", "каждый элемент отнесёт к одной из заданных категорий"],
  scan: ["Оценить открытую страницу", "прочитает ссылки, кнопки или посты и отберёт нужные"],
  fill: ["Заполнить форму на странице", "найдёт поля и впишет значения из вашей инструкции"],
  click: ["Нажать найденное", "нажмёт то, что отобрал «Оценить страницу»"],
};
const LEETH_TOOL_OF = {delegate: "delegate_to_leeth", filter: "leeth_map", classify: "leeth_map", scan: "leeth_scan_page", fill: "leeth_fill_form", click: "leeth_click_many"};
const LEETH_MANAGED = {
  delegate: ["task"], filter: ["items", "criterion", "mode", "categories"], classify: ["items", "criterion", "mode", "categories"],
  scan: ["criterion", "mode", "categories"], fill: ["instruction"], click: ["indices"],
};
function leethNodeForm(side, n, upd) {
  const mode = n.mode || "delegate";
  if (META.leeth && META.leeth.available === false) side.append(el("div", {class: "notice warn"}, "Leeth здесь не загружен: шаг ответит, что не может. Проверьте веса модели."));
  side.append(el("div", {class: "lbl"}, "Что поручить Leeth"));
  side.append(modeCards(LEETH_MODE_LABEL, mode, (v) => {
    n.mode = v;
    if (["filter", "classify", "click"].includes(v) === false) delete n.input_from;
    touch("lmode", true);
  }));
  n.args = n.args || {};
  const a = n.args;
  const setA = (k, v) => { if (v === "" || v == null || (Array.isArray(v) && !v.length)) delete a[k]; else a[k] = v; upd("larg" + n.id); };
  if (mode === "delegate") {
    side.append(field("Задача", withInserter(area(n.instruction, (v) => { n.instruction = v; upd("li" + n.id); }, 3)), "Обычными словами: «прочитай config.py», «проверь фоновые задачи»."));
  } else if (mode === "filter" || mode === "classify") {
    side.append(fieldPick("Список для разбора", n.input_from, (v) => { n.input_from = v; upd("lf" + n.id); }, "Поле со списком: JSON или по строке на элемент (например результат «строки» из предыдущего шага)."));
    side.append(field(mode === "filter" ? "Вопрос «да/нет» про один элемент" : "Вопрос (необязательно)", withInserter(area(n.instruction, (v) => { n.instruction = v; upd("li" + n.id); }, 2)),
      mode === "filter" ? "Например: «Это пост про животных?»" : "Например: «О чём эта закладка?»"));
    if (mode === "classify") {
      side.append(field("Категории (по одной в строке)", area((a.categories || []).join("\n"), (v) => setA("categories", v.split("\n").map((s) => s.trim()).filter(Boolean)), 3), "Не меньше двух."));
    } else {
      side.append(field("Примеры «подходит» (по строке)", area((a.positive_examples || []).join("\n"), (v) => setA("positive_examples", v.split("\n").map((s) => s.trim()).filter(Boolean)), 3),
        "2–3 текста, которые отвечают на вопрос «да». Сильно повышают точность."));
      side.append(field("Примеры «не подходит» (по строке)", area((a.negative_examples || []).join("\n"), (v) => setA("negative_examples", v.split("\n").map((s) => s.trim()).filter(Boolean)), 3)));
    }
  } else if (mode === "scan") {
    side.append(field("Вопрос «да/нет» про один элемент страницы", withInserter(area(n.instruction, (v) => { n.instruction = v; upd("li" + n.id); }, 2)), "Например: «В посте говорится про животных?»"));
    side.append(field("Что оценивать", select([["links", "ссылки"], ["buttons", "кнопки"], ["all", "всё кликабельное"], ["posts", "посты / карточки (текст блоков)"]], a.include || "links", (v) => setA("include", v)),
      "Для постов результат содержит и их кнопки — выберите нужную в шаге «Нажать найденное»."));
    side.append(field("Прокрутить страницу раз", inp(a.scroll_pages ?? "", (v) => setA("scroll_pages", v === null ? "" : v), {type: "number"}), "Для бесконечных лент: сколько раз пролистать вниз (до 30)."));
    side.append(field("Примеры «подходит» (по строке)", area((a.positive_examples || []).join("\n"), (v) => setA("positive_examples", v.split("\n").map((s) => s.trim()).filter(Boolean)), 3), "Как выглядит подходящий пост или ссылка."));
    side.append(field("Примеры «не подходит» (по строке)", area((a.negative_examples || []).join("\n"), (v) => setA("negative_examples", v.split("\n").map((s) => s.trim()).filter(Boolean)), 3)));
  } else if (mode === "fill") {
    side.append(field("Что вписать", withInserter(area(n.instruction, (v) => { n.instruction = v; upd("li" + n.id); }, 3)), "Своими словами: «логин Валера, рост 175, город Гонконг»."));
    side.append(check("Только показать план, ничего не менять", a.dry_run, (v) => setA("dry_run", v || ""), "Хороший первый запуск."));
    side.append(check("Нажать кнопку отправки", a.submit, (v) => setA("submit", v || ""), "Платёжные и удаляющие кнопки Leeth не нажимает никогда."));
  } else if (mode === "click") {
    side.append(fieldPick("Что нажать (поле с результатом оценки страницы)", n.input_from, (v) => { n.input_from = v; upd("lf" + n.id); }, "Поле, куда шаг «Оценить страницу» записал JSON (запомните его как «JSON»)."));
    side.append(field("Какую кнопку нажать в каждом посте", inp(a.button || "", (v) => setA("button", v), {ph: "закладк|bookmark|в избранное"}),
      "Шаблон по подписи кнопки (регистр не важен). Нужен, если найденное — посты: без него комната откажется, чтобы не нажать лишнее."));
    side.append(check("Брать и «неуверенные»", a.include_unsure, (v) => setA("include_unsure", v || ""), "Кроме «точно да» — ещё и те, в ком Leeth не уверен."));
  }
  const ti = toolInfo(LEETH_TOOL_OF[mode]);
  if (ti) {
    const managed = LEETH_MANAGED[mode] || [];
    const rest = (ti.params || []).filter((p) => !managed.includes(p.name) && !["positive_examples", "negative_examples", "include", "scroll_pages", "dry_run", "submit", "confirm"].includes(p.name));
    if (rest.length) side.append(el("details", {class: "adv"}, el("summary", {}, "Тонкая настройка"), el("div", {class: "body"},
      paramForm(rest, a, () => upd("larg" + n.id)))));
  }
  side.append(field("Запомнить из результата", setsEditor(n.sets, (v) => { n.sets = v; upd("sets" + n.id); }),
    mode === "filter" || mode === "scan" || mode === "classify" ? "Ответ — JSON. Для оценки страницы запомните «JSON» целиком и передайте его в шаг «Нажать найденное»; для списка «да» — путь «yes»." : null));
  side.append(check("Спросить вас перед запуском", n.confirm, (v) => { n.confirm = v; upd("confirm"); },
    ["fill", "click", "delegate"].includes(mode) ? "Это действие: лучше спрашивать." : "Только чтение: можно не спрашивать."));
}

function foreachNodeForm(side, n, upd) {
  side.append(fieldPick("Список", n.input_from, (v) => { n.input_from = v; upd("fe" + n.id); }, "Поле со списком: JSON или по строке на элемент."));
  side.append(field("Имя переменной элемента", inp(n.item || "", (v) => { n.item = v; upd("fi" + n.id); }, {ph: "item"}), `В тексте шагов: {{${n.item || "item"}}}. Ещё: «${n.id}_index», «${n.id}_count», «${n.id}_done».`));
  side.append(field("Не больше элементов", inp(n.max_items ?? "", (v) => { n.max_items = v || undefined; touch("fm"); }, {type: "number", min: 1}), "Страховка: до 500."));
  side.append(loopWiringHelp(n));
}
function loopWiringHelp(n) {
  return el("div", {class: "says"}, "Как соединить: стрелка «если не ", el("code", {}, `state.${n.id}_done`), "» ведёт в тело цикла, стрелка «иначе» — после цикла, а из конца тела — обратно сюда. Первая стрелка из этого блока получит нужное условие сама.");
}
function waitNodeForm(side, n, upd) {
  side.append(field("Пауза, секунд", inp(n.seconds ?? "", (v) => { n.seconds = v; upd("w" + n.id); }, {type: "number", min: 0}), "До 3600. Нужна, например, чтобы страница успела подгрузиться."));
}

function loopNodeForm(side, n, upd) {
  const V = VOC(), mode = n.mode || "for";
  side.append(modeCards(LOOP_LABEL, mode, (v) => { n.mode = v; touch("lmode", true); }));
  side.append(field("Имя переменной", inp(n.item || "", (v) => { n.item = v; upd("it" + n.id); }, {ph: mode === "for_in" ? "key" : "i"}),
    mode === "for_in" ? "Сюда попадёт ключ словаря (или номер в списке)." : "Сюда попадает значение счётчика на каждом круге."));
  if (mode === "for") {
    side.append(el("div", {class: "row"},
      field("От", inp(n.start ?? "", (v) => { n.start = scalarOrTemplate(v) === "" ? undefined : scalarOrTemplate(v); upd("ls" + n.id); }, {ph: "0"})),
      field("До (не включая)", inp(n.stop ?? "", (v) => { n.stop = scalarOrTemplate(v) === "" ? undefined : scalarOrTemplate(v); upd("le" + n.id); }, {ph: "10"})),
      field("Шаг", inp(n.step ?? "", (v) => { n.step = scalarOrTemplate(v) === "" ? undefined : scalarOrTemplate(v); upd("lp" + n.id); }, {ph: "1"}))));
    side.append(el("p", {class: "hint"}, "Можно подставлять {{поле}} или считать: «= state.n + 1». Шаг может быть отрицательным (счёт вниз)."));
  } else if (mode === "times") {
    side.append(field("Сколько раз", inp(n.stop ?? "", (v) => { n.stop = scalarOrTemplate(v) === "" ? undefined : scalarOrTemplate(v); upd("le" + n.id); }, {ph: "5"}), "Число, {{поле}} или «= выражение»."));
  } else if (mode === "while" || mode === "do_while" || mode === "until") {
    side.append(field(mode === "until" ? "Повторять, пока НЕ выполнено" : "Повторять, пока выполнено", condBuilder(n.condition || "state.ok", (v) => { n.condition = v; upd("lc" + n.id); }),
      "Проверяется перед каждым кругом" + (mode === "while" ? "." : ", кроме первого — он выполняется всегда.")));
  } else if (mode === "for_in") {
    side.append(fieldPick("Словарь или список", n.input_from, (v) => { n.input_from = v; upd("lf" + n.id); }, "Поле с JSON-словарём ({…}) или списком."));
    side.append(field("Имя переменной значения", inp(n.item_value || "", (v) => { n.item_value = v; upd("lv" + n.id); }, {ph: "value"})));
  }
  side.append(field("Не больше кругов (страховка)", inp(n.max_items ?? "", (v) => { n.max_items = v || undefined; touch("lmax"); }, {type: "number", min: 1, ph: String(V.loop_default_max)}),
    `Если пусто — ${V.loop_default_max}; потолок ${V.loop_max}. Когда лимит сработал, «${n.id}_capped» станет истиной.`));
  side.append(el("div", {class: "says"}, `Создаёт поля: ${n.item || (mode === "for_in" ? "key" : "i")}, ${n.id}_index, ${n.id}_done, ${n.id}_broken, ${n.id}_capped` + (["for", "times", "for_in"].includes(mode) ? `, ${n.id}_count` : "") + "."));
  side.append(loopWiringHelp(n));
}
function breakNodeForm(side, n, upd) {
  const loops = room.nodes.filter((x) => x.type === "loop" || x.type === "foreach").map((x) => [x.id, x.id + (x.type === "foreach" ? " (для каждого)" : " (цикл)")]);
  side.append(field("К какому циклу относится", loops.length ? select([["", "— выберите —"], ...loops], n.loop || "", (v) => { n.loop = v; touch("brk", true); }) : el("p", {class: "hint"}, "В комнате пока нет циклов.")));
  const has = room.edges.some((e) => e.from === n.id && e.to === n.loop);
  if (n.loop && !has) {
    side.append(el("div", {class: "notice warn"}, "Из этого блока должна вести стрелка обратно в цикл — иначе он ничего не прервёт.",
      el("div", {class: "row"}, el("button", {class: "btn sm primary", onclick: () => { room.edges.push({from: n.id, to: n.loop}); touch("brk", true); }}, "Провести стрелку в цикл"))));
  }
  side.append(el("div", {class: "says"}, n.type === "break"
    ? "Break: при ближайшем заходе в цикл он закончится, и комната пойдёт по стрелке «иначе» после цикла."
    : "Continue: пропустить остаток тела и перейти к следующему кругу — просто путь назад в цикл."));
}
function powerNodeForm(side, n, upd) {
  const V = VOC(), mode = n.mode || "shutdown";
  side.append(el("div", {}, el("span", {class: "warnchip dangerchip"}, "действует на ваш компьютер"),
    n.confirm ? null : el("span", {class: "warnchip"}, "без вопроса — пока вы спите, так и надо")));
  side.append(modeCards(POWER_LABEL, mode, (v) => { n.mode = v; if (v !== "shutdown" && v !== "restart" && v !== "cancel" && n.seconds == null) {/* сразу */} touch("pmode", true); }));
  if (mode === "shutdown" || mode === "restart") {
    side.append(field("Через сколько секунд", inp(n.seconds ?? 60, (v) => { n.seconds = v; upd("psec" + n.id); }, {type: "number", min: V.power_min_delay}),
      `Не меньше ${V.power_min_delay} — чтобы успеть отменить (блок «Отменить выключение» или команда shutdown /a).`));
    side.append(field("Сообщение в системе", inp(n.message || "", (v) => { n.message = v; touch("pmsg" + n.id); }, {ph: "Argent: комната завершила работу"})));
  }
  n.args = n.args || {};
  side.append(check("Пробный запуск: только показать команду", n.args.dry_run, (v) => { if (v) n.args.dry_run = true; else delete n.args.dry_run; upd("pdry" + n.id); }, "Ничего не выключит — хорошо для проверки комнаты."));
  side.append(check("Спросить вас перед действием", n.confirm, (v) => { n.confirm = v; upd("confirm"); }, "Если комната работает без вас — оставьте выключенным и поставьте перед этим блоком уведомление в Telegram."));
}
function hintNodeForm(side, n, upd) {
  const mode = n.mode || "next";
  side.append(modeCards(HINT_LABEL, mode, (v) => { n.mode = v; touch("hmode", true); }));
  if (mode !== "clear") {
    side.append(field(n.compose ? "Что должна написать модель" : "Текст подсказки", withInserter(area(n.message, (v) => { n.message = v; upd("hm" + n.id); }, 4)),
      n.compose ? "Это инструкция: модель напишет подсказку сама по тому, что уже известно." : "Например: «Ты зациклился: попробуй другой запрос» или «Цель — вежливо договориться о встрече»."));
    side.append(check("Текст пишет модель по ситуации", n.compose, (v) => { n.compose = v; touch("hcomp", true); }, "Одна короткая просьба к модели без инструментов."));
  } else {
    n.message = n.message || "снять";
  }
  const agents = room.nodes.filter((x) => x.type === "agent").map((x) => x.id);
  if (agents.length) {
    side.append(el("div", {class: "lbl"}, mode === "clear" ? "Снять только для шагов (пусто — для всех)" : "Только для шагов (пусто — для любого)"));
    const box = el("div", {class: "tools"});
    for (const id of agents) {
      const c = el("input", {type: "checkbox"}); c.checked = (n.targets || []).includes(id);
      c.addEventListener("change", () => { const t = new Set(n.targets || []); c.checked ? t.add(id) : t.delete(id); n.targets = [...t]; if (!n.targets.length) delete n.targets; touch("htg" + n.id); });
      box.append(el("label", {class: "check"}, c, el("span", {}, id)));
    }
    side.append(box);
  }
  side.append(el("div", {class: "says"}, "Подсказка попадает в начало задания шага-агента блоком «HINTS». Чтобы подавать её при зацикливании, поставьте на стрелку условие вроде state." + (room.nodes[0] ? room.nodes[0].id : "шаг") + "_visits >= 3."));
}
function forgetNodeForm(side, n, upd) {
  side.append(modeCards(FORGET_LABEL, n.mode || "all", (v) => { n.mode = v; touch("fmode", true); }));
  side.append(field("Очистить ещё и поля состояния", listInput(n.clear, (v) => { n.clear = v; upd("fc" + n.id); }, "posts, found"), "Через запятую. Эти поля исчезнут из состояния комнаты."));
  side.append(el("div", {class: "says"}, "Помните: с чистого листа начинают только агенты. Что сделано во внешнем мире (файлы, сообщения), останется."));
}
function assertNodeForm(side, n, upd) {
  side.append(field("Должно быть верно", condBuilder(n.condition || "state.ok", (v) => { n.condition = v; upd("ac" + n.id); })));
  side.append(field("Сообщение, если неверно", withInserter(area(n.message, (v) => { n.message = v; upd("am" + n.id); }, 2)), "Шаг упадёт; если есть стрелка «если шаг упал», пойдём по ней, иначе прогон остановится."));
}

// параллельные ветки: блок «Параллельно» запускает все исходящие стрелки сразу, «Слияние» их собирает
function parallelNodeForm(side, n, upd) {
  const V = VOC();
  const joins = room.nodes.filter((x) => x.type === "join").map((x) => [x.id, x.id]);
  side.append(field("Где ветки сходятся (блок «Слияние»)",
    joins.length ? select([["", "— выберите —"], ...joins], n.join || "", (v) => { n.join = v; touch("pjoin", true); })
      : el("p", {class: "hint"}, "В комнате пока нет блока «Слияние»."),
    "Каждая ветка должна дойти до этого блока."));
  if (!joins.length || (n.join && !room.nodes.some((x) => x.id === n.join))) {
    side.append(el("div", {class: "row"}, el("button", {class: "btn sm primary", onclick: () => {
      const j = defaultNode("join"); room.nodes.push(j); n.join = j.id; touch("pjoin", true);
    }}, "Создать блок «Слияние»")));
  }
  side.append(field("Сколько веток одновременно", inp(n.workers ?? "", (v) => { n.workers = v ? Number(v) : undefined; upd("pw" + n.id); },
    {type: "number", min: 1, max: V.parallel_max_branches || 8, ph: "все сразу"}),
    "Пусто — все ветки разом. 1 — по очереди, удобно для отладки: порядок всегда один и тот же."));
  const branches = room.edges.filter((e) => e.from === n.id);
  side.append(el("div", {class: "says"}, branches.length < 2
    ? "Проведите от этого блока минимум две стрелки — каждая будет отдельной веткой."
    : `Ветки (${branches.length}): ` + branches.map((e) => e.to).join(", ") + ". Порядок — как нарисованы стрелки; по нему сливаются конфликты."));
  side.append(el("div", {class: "notice"}, "У ветки своя копия состояния. Записывайте в разные поля: одно и то же поле из двух веток — конфликт (его решает блок «Слияние»). Вопросов вам внутри ветки быть не может."));
}
function joinNodeForm(side, n, upd) {
  const mode = (k, d) => n[k] || d;
  side.append(field("Чего ждать", select([["all", "Все ветки"], ["any", "Первую, которая придёт (остальные остановятся)"]], mode("wait_for", "all"),
    (v) => { n.wait_for = v; touch("jw", true); }),
    "«Первая» полезна, когда нужен один из нескольких путей; порядок прихода не детерминирован."));
  side.append(field("Если две ветки записали разное в одно поле", select([["error", "Остановить комнату с ошибкой"], ["first", "Взять значение первой ветки"],
    ["last", "Взять значение последней ветки"], ["list", "Положить все значения списком"]], mode("merge", "error"), (v) => { n.merge = v; touch("jm", true); }),
    "«Первая» и «последняя» — по порядку нарисованных стрелок, а не по тому, кто закончил раньше."));
  side.append(field("Если ветка не дошла (ошибка, выход, лимит)", select([["fail", "Остановить комнату"], ["continue", "Игнорировать эту ветку и идти дальше"]],
    mode("on_branch_fail", "fail"), (v) => { n.on_branch_fail = v; touch("jf", true); })));
  side.append(el("div", {class: "says"}, `Создаёт поля: ${n.id}_arrived (кто дошёл), ${n.id}_failed (кто нет), ${n.id}.output.`));
}

// агент: память, подсказка, параметры модели, скиллы, MCP, игнорирование
function agentExtras(side, n, upd) {
  const rr = (room.rules || {});
  side.append(el("details", {class: "adv"}, el("summary", {}, "Память, подсказка, параметры модели"), el("div", {class: "body"},
    field("Память", select([["", "как в комнате (" + (rr.memory === "shared" ? "общая" : "своя") + ")"], ["isolated", "своя: не знает о прошлых шагах"], ["shared", "общая: знает, что делали прошлые шаги"]],
      n.memory || "", (v) => { if (v) n.memory = v; else delete n.memory; touch("mem" + n.id); })),
    field("Подсказка этому шагу", withInserter(area(n.hint, (v) => { n.hint = v; touch("hint" + n.id); }, 3)), "Показывается только ему, каждый раз."),
    el("div", {class: "lbl"}, "Параметры модели для этого шага"),
    samplingEditor(n.sampling, (v) => { n.sampling = v; touch("smp" + n.id); }, {inherit: rr.sampling}))));
  const skills = CAT.skills || [];
  const servers = Object.keys(CAT.mcpServers || {});
  const body = el("div", {class: "body"});
  body.append(el("div", {class: "lbl"}, "Скиллы, чьи инструкции дать агенту"));
  if (skills.length) {
    const box = el("div", {class: "tools"});
    for (const s of skills) {
      const c = el("input", {type: "checkbox"}); c.checked = (n.skills || []).includes(s.name);
      c.addEventListener("change", () => { const t = new Set(n.skills || []); c.checked ? t.add(s.name) : t.delete(s.name); n.skills = [...t]; if (!n.skills.length) delete n.skills; touch("sk" + n.id); });
      box.append(el("label", {class: "check", title: s.description}, c, el("span", {}, s.name)));
    }
    body.append(box);
  } else body.append(el("p", {class: "hint"}, "Скиллы не найдены."));
  body.append(el("div", {class: "lbl"}, "MCP-серверы, которыми можно пользоваться"));
  body.append(select([["any", "любыми настроенными"], ["none", "никакими"], ["pick", "только выбранными"]], n.mcp == null ? "any" : (n.mcp.length ? "pick" : "none"),
    (v) => { if (v === "any") delete n.mcp; else if (v === "none") n.mcp = []; else n.mcp = servers.slice(0, 1); touch("mcp" + n.id, true); }));
  if (n.mcp && n.mcp.length) {
    const box = el("div", {class: "tools"});
    for (const s of servers) {
      const c = el("input", {type: "checkbox"}); c.checked = n.mcp.includes(s);
      c.addEventListener("change", () => { n.mcp = c.checked ? [...n.mcp, s] : n.mcp.filter((x) => x !== s); touch("mcp" + n.id); });
      box.append(el("label", {class: "check"}, c, el("span", {}, s)));
    }
    body.append(box);
  }
  body.append(el("div", {class: "lbl"}, "Игнорировать"));
  body.append(ignoreEditor(n.ignore, (v) => { n.ignore = v; touch("ign" + n.id); }));
  body.append(el("p", {class: "hint"}, "Правила действуют по-настоящему: запрещённый вызов не пройдёт, а скрытые разделы скиллов агент не увидит."));
  side.append(el("details", {class: "adv"}, el("summary", {}, "Скиллы, MCP и что игнорировать"), body));
}
function notifyExtras(side, n, upd) {
  side.append(check("Текст пишет модель", n.compose, (v) => { n.compose = v; touch("ncomp", true); }, "Тогда «Сообщение» — это инструкция: модель напишет текст по итогам работы."));
  side.append(el("p", {class: "hint"}, n.channel === "telegram" ? "Telegram должен быть настроен в Argent. Если доставка не удалась, прогон не падает, а поле «" + n.id + ".delivered» станет ложью."
    : n.channel === "desktop" ? "Всплывающее окно покажет служба уведомлений вашего ПК." : "Сообщение появится в консоли Argent."));
}

// главная точка входа из app.js: true — форма блока построена здесь
function extNodeForm(side, n, upd) {
  const t = n.type;
  if (t === "tool" && (META.tool_info || []).length) { toolNodeForm(side, n, upd); return true; }
  const forms = {mcp: mcpNodeForm, skill: skillNodeForm, plugin: pluginNodeForm, leeth: leethNodeForm, foreach: foreachNodeForm,
    wait: waitNodeForm, loop: loopNodeForm, break: breakNodeForm, continue: breakNodeForm, power: powerNodeForm, hint: hintNodeForm,
    forget: forgetNodeForm, assert: assertNodeForm, parallel: parallelNodeForm, join: joinNodeForm};
  if (forms[t]) { forms[t](side, n, upd); return true; }
  return false;
}

// ── запуск по событию: расписание и webhook ─────────────────────────────────────
function roomTriggersPanel(side) {
  side.append(el("h2", {}, "Запуск по событию"));
  side.append(el("p", {class: "hint"}, "Комната может стартовать сама: по расписанию или по HTTP-запросу (webhook). Запуск идёт без вас за клавиатурой: шаги с подтверждением не выполняются, вопрос к человеку останавливает комнату."));
  if (!room || !room.room || source === "new" || source === "example") {
    side.append(el("div", {class: "notice"}, "Сначала сохраните комнату — триггер привязывается к сохранённой."));
    return;
  }
  const box = el("div", {class: "triggers"});
  const reveal = el("div");
  side.append(box, reveal);
  const when = inp("", () => {}, {ph: "каждый день в 03:00"});
  const hookName = inp("", () => {}, {ph: room.room});
  const call = async (body) => {
    const r = await api("/api/trigger", Object.assign({room: room.room}, body));
    if (!(r.data && r.data.ok)) say((r.data.errors || [r.data.error || "не получилось"]).join("; "), "bad");
    return r.data || {};
  };
  const refresh = async () => {
    const r = await api("/api/triggers?room=" + encodeURIComponent(room.room));
    const items = (r.data && r.data.triggers) || [];
    box.replaceChildren();
    if (!items.length) box.append(el("p", {class: "hint"}, "Триггеров пока нет."));
    for (const t of items) {
      const title = t.kind === "schedule" ? "Расписание: " + t.schedule : "Webhook: POST " + t.url;
      const row = el("div", {class: "trig" + (t.enabled ? "" : " off")},
        el("div", {}, el("b", {}, t.name), " · ", title, t.last_run ? el("div", {class: "hint"}, `последний запуск: ${t.last_run} (${t.last_status || "?"})`) : null),
        el("div", {class: "row"},
          el("button", {class: "btn sm", onclick: async () => { await call({action: t.enabled ? "disable" : "enable", kind: t.kind, name: t.name}); refresh(); }}, t.enabled ? "Выключить" : "Включить"),
          t.kind === "webhook" ? el("button", {class: "btn sm", onclick: async () => {
            const d = await call({action: "new_secret", kind: "webhook", name: t.name});
            if (d.secret) showSecret(t.url, d.secret);
          }}, "Новый секрет") : null,
          el("button", {class: "btn sm", onclick: async () => { await call({action: "remove", kind: t.kind, name: t.name}); reveal.replaceChildren(); refresh(); }}, "Убрать")));
      box.append(row);
    }
  };
  const showSecret = (url, secret) => {
    const curl = `curl -X POST ${url} -H "X-Argent-Secret: ${secret}" -H "Content-Type: application/json" -d "{}"`;
    reveal.replaceChildren(el("div", {class: "notice warn"},
      el("b", {}, "Секрет показан один раз — сохраните его. "), "У нас хранится только его хэш.",
      el("pre", {class: "says"}, "Адрес: " + url + "\nСекрет: " + secret + "\n\n" + curl)));
  };
  side.append(field("Добавить расписание", when, "Например: «каждый день в 03:00», «каждый час», «по будням в 9:00», «при запуске»."));
  side.append(el("div", {class: "row"}, el("button", {class: "btn sm primary", onclick: async () => {
    if (!when.value.trim()) { say("Напишите, когда запускать.", "bad"); return; }
    const d = await call({action: "add_schedule", schedule: when.value.trim()});
    if (d.ok) { when.value = ""; say("Расписание добавлено.", "ok"); refresh(); }
  }}, "Добавить расписание")));
  side.append(field("Создать webhook", hookName, "Имя в адресе; пусто — по названию комнаты. Поля, которые принимает запрос, — параметры комнаты."));
  side.append(el("div", {class: "row"}, el("button", {class: "btn sm primary", onclick: async () => {
    const d = await call({action: "add_webhook", name: hookName.value.trim()});
    if (d.ok) { hookName.value = ""; showSecret(d.trigger.url, d.secret); refresh(); }
  }}, "Создать webhook")));
  refresh();
}

// ── правила всей комнаты ─────────────────────────────────────────────────────────
function roomRulesPanel(side) {
  room.rules = room.rules || {};
  const r = room.rules;
  const set = (k, v, key) => {
    if (v === false || v === "" || v == null || (Array.isArray(v) && !v.length) || (v && typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length)) delete r[k]; else r[k] = v;
    touch(key || ("rr" + k));
  };
  side.append(el("h2", {}, "Правила комнаты"));
  side.append(el("p", {class: "hint"}, "Не подключаем, а запрещаем: правила действуют на все шаги и на все комнаты, которые эта комната вызывает."));
  side.append(check("Запретить любые инструменты", r.no_tools, (v) => { set("no_tools", v); panel(); }, "Агенты работают без инструментов, шаги-команды, MCP, плагины и Leeth не запускаются. Уведомления и вопросы вам работают."));
  side.append(check("Только чтение: запретить изменяющие", r.read_only, (v) => { set("read_only", v); panel(); }, "Запись и удаление файлов, команды оболочки, git-откаты, действия в браузере, MCP-вызовы, плагины."));
  side.append(el("details", {class: "adv"}, el("summary", {}, "Запретить отдельные инструменты" + ((r.deny_tools || []).length ? ` · ${r.deny_tools.length}` : "")), el("div", {class: "body"}, (() => {
    const box = el("div", {class: "tools"});
    for (const t of META.tool_info || []) {
      const c = el("input", {type: "checkbox"}); c.checked = (r.deny_tools || []).includes(t.name);
      c.addEventListener("change", () => { const s = new Set(r.deny_tools || []); c.checked ? s.add(t.name) : s.delete(t.name); set("deny_tools", [...s].sort(), "rdeny"); });
      box.append(el("label", {class: "check", title: t.description}, c, el("span", {}, t.name + (t.mutating ? " ⚠" : ""))));
    }
    return box;
  })())));
  side.append(el("details", {class: "adv"}, el("summary", {}, "Что игнорировать во всей комнате" + ((r.ignore || []).length ? ` · ${r.ignore.length}` : "")), el("div", {class: "body"},
    ignoreEditor(r.ignore, (v) => set("ignore", v, "rign")))));
  side.append(el("details", {class: "adv"}, el("summary", {}, "Память агентов"), el("div", {class: "body"},
    field("Режим", select([["isolated", "своя у каждого шага"], ["shared", "общая: агенты знают, что делали прошлые шаги"]], r.memory || "isolated", (v) => { if (v === "isolated") delete r.memory; else r.memory = v; touch("rmem"); })),
    field("Сколько символов помнить", inp(r.memory_chars ?? 6000, (v) => { if (v && v !== 6000) r.memory_chars = v; else delete r.memory_chars; touch("rmc"); }, {type: "number", min: 500}), "Старое забывается первым. Очистить целиком можно блоком «Забыть»."))));
  side.append(el("details", {class: "adv"}, el("summary", {}, "Параметры модели по умолчанию"), el("div", {class: "body"},
    samplingEditor(r.sampling, (v) => set("sampling", v, "rsmp")))));
  side.append(field("Общая подсказка всем агентам", withInserter(area(r.hint, (v) => set("hint", v, "rhint"), 3)), "Звучит в каждом шаге-агенте. Например: «Отвечай коротко, по-русски»."));
}

// ── мастер «Switch по полю» ──────────────────────────────────────────────────────
function switchPanel(side) {
  const targets = [...room.nodes.map((n) => [n.id, n.id]), ...exitNames().map((x) => ["EXIT:" + x, "выход «" + x + "»"])];
  const st = {field: "", cases: [{value: "", to: targets[0] ? targets[0][0] : ""}], other: targets[0] ? targets[0][0] : ""};
  side.append(el("h2", {}, "Switch по значению поля"));
  side.append(el("p", {class: "hint"}, "Создаст развилку со стрелками «если поле равно значению → цель» и запасной «иначе»."));
  side.append(fieldPick("Поле", st.field, (v) => { st.field = v.trim(); }, null, "поле, напр. category"));
  const list = el("div", {class: "kv"});
  const render = () => {
    list.replaceChildren();
    st.cases.forEach((c, i) => list.append(el("div", {class: "kvrow"},
      inp(c.value, (v) => { c.value = v; }, {ph: "значение"}),
      select(targets, c.to, (v) => { c.to = v; }),
      el("button", {class: "btn icon ghost sm", "aria-label": "Убрать", onclick: () => { st.cases.splice(i, 1); render(); }}, icon("i-close", "sm")))));
    list.append(el("button", {class: "btn sm", onclick: () => { st.cases.push({value: "", to: targets[0] ? targets[0][0] : ""}); render(); }}, icon("i-plus", "sm"), "Добавить значение"));
  };
  render();
  side.append(el("div", {class: "lbl"}, "Значения и куда идти"), list);
  side.append(field("Иначе — куда", select(targets, st.other, (v) => { st.other = v; })));
  side.append(el("div", {class: "footer"},
    el("button", {class: "btn primary", onclick: () => {
      if (!/^[A-Za-z_]\w*$/.test(st.field)) { say("Укажите поле: латиница, цифры, «_».", "bad"); return; }
      const cases = st.cases.filter((c) => c.value !== "" && c.to);
      if (!cases.length) { say("Добавьте хотя бы одно значение.", "bad"); return; }
      const n = {id: uniqueId("switch_" + st.field), type: "condition", condition: "state." + st.field};
      room.nodes.push(n);
      const S = sel && sel.kind === "node" ? nodeById(sel.id) : null;
      ensureLayout();
      const [x, y] = S ? pos(S.id) : [40, 56];
      layout[n.id] = freeSpot(x, y + NH + 70);
      if (S && !room.edges.some((e) => e.from === S.id && ["uncond", "else"].includes(edgeKind(e)))) room.edges.push({from: S.id, to: n.id});
      for (const c of cases) room.edges.push({from: n.id, to: c.to, if: buildCond({field: st.field, op: "eq", value: c.value})});
      if (st.other) room.edges.push({from: n.id, to: st.other, else: true});
      inspectMode = null; sel = {kind: "node", id: n.id};
      touch("switch", true); revealNode(n.id);
    }}, "Создать"),
    el("button", {class: "btn", onclick: () => { inspectMode = null; panel(); }}, "Отмена")));
}

// ── палитра: каталог блоков, поиск, категории ────────────────────────────────────
function paletteEntries() {
  const E = [];
  const add = (cat, type, title, sub, preset, kw) => E.push({cat, type, title, sub, preset: preset || {}, kw: (kw || "") + " " + title + " " + sub});
  const C = {logic: "Логика", loops: "Циклы", agent: "Агент и память", people: "Люди и уведомления", pc: "Компьютер",
    argent: "Инструменты Argent", mcp: "MCP", skills: "Скиллы", plugins: "Плагины", leeth: "Leeth", rooms: "Другие комнаты"};
  add(C.logic, "condition", "Развилка", "Выбрать путь по условию", {}, "if условие ветка");
  add(C.logic, "set", "Значение", "Записать или посчитать поле", {}, "variable переменная set счётчик");
  add(C.logic, "switch", "Switch по полю", "Несколько веток по значению", {}, "case switch выбор");
  add(C.logic, "assert", "Проверка (assert)", "Остановиться, если неверно", {}, "assert проверка ошибка");
  add(C.logic, "parallel", "Параллельно (fork)", "Несколько веток одновременно", {}, "parallel fork параллельно одновременно ветки");
  add(C.logic, "join", "Слияние (join)", "Собрать ветки вместе", {}, "join merge слияние собрать ветки");
  add(C.logic, "wait", "Пауза", "Подождать N секунд", {seconds: 5}, "sleep delay ждать");
  add(C.logic, "room", "Подкомната", "Запустить другую комнату", {}, "sub workflow");
  add(C.loops, "loop", "Цикл for", "Счётчик от и до", {mode: "for", start: 0, stop: 5, step: 1, item: "i"}, "for range диапазон");
  add(C.loops, "loop", "Повторить N раз", "Просто N кругов", {mode: "times", stop: 3}, "times repeat раз");
  add(C.loops, "loop", "Цикл while", "Пока условие верно", {mode: "while", condition: "state.go"}, "while пока");
  add(C.loops, "loop", "Цикл do-while", "Сначала круг, потом условие", {mode: "do_while", condition: "state.go"}, "do while делать пока");
  add(C.loops, "loop", "Повторять, пока не (repeat-until)", "До выполнения условия", {mode: "until", condition: "state.done"}, "repeat until пока не");
  add(C.loops, "loop", "Цикл for-in", "По ключам словаря", {mode: "for_in", item: "key", item_value: "value"}, "for in dict ключ словарь");
  add(C.loops, "foreach", "Для каждого (foreach)", "Пройти по списку", {item: "item", max_items: 100}, "foreach each список");
  add(C.loops, "break", "Break", "Выйти из цикла", {}, "break выйти прервать");
  add(C.loops, "continue", "Continue", "К следующему кругу", {}, "continue продолжить");
  add(C.agent, "agent", "Агент", "Поручить работу модели", {}, "llm model ai");
  add(C.agent, "hint", "Подсказка следующему агенту", "Один раз", {mode: "next", message: "Подсказка: "}, "hint подсказка совет");
  add(C.agent, "hint", "Подсказка всем агентам", "До отмены", {mode: "all", message: "Помни цель: "}, "hint sticky цель");
  add(C.agent, "hint", "Подсказка при зацикливании", "Пишет модель по ситуации", {mode: "next", compose: true, message: "Агент ходит по кругу. Коротко подскажи, как выбраться."}, "loop зациклился");
  add(C.agent, "hint", "Снять подсказки", "Убрать «всем агентам»", {mode: "clear", message: "снять"}, "clear hint");
  add(C.agent, "forget", "Забыть память", "Очистить память и подсказки", {mode: "all"}, "memory forget очистить");
  add(C.people, "human_pause", "Вопрос вам", "Остановиться и спросить", {}, "ask human вопрос");
  add(C.people, "notify", "Уведомление в Telegram", "Сообщение в чат", {channel: "telegram", message: "Готово"}, "telegram телеграм");
  add(C.people, "notify", "Telegram: текст пишет модель", "По итогам работы", {channel: "telegram", compose: true, message: "Коротко расскажи, чем закончилась работа и что получилось."}, "telegram llm");
  add(C.people, "notify", "Всплывающее уведомление ПК", "Окно в системе", {channel: "desktop", message: "Готово"}, "toast desktop окно");
  add(C.people, "notify", "Сообщение в консоль", "В окно Argent", {channel: "console", message: "Готово"}, "console консоль");
  for (const m of VOC().power_modes) add(C.pc, "power", POWER_LABEL[m][0], POWER_LABEL[m][1], {mode: m, ...(m === "shutdown" || m === "restart" ? {seconds: 60} : {})}, "power питание пк компьютер " + m);
  for (const t of META.tool_info || []) {
    if ((META.not_in_rooms || []).includes(t.name)) continue;
    const g = (META.tool_groups || {})[t.group] || t.group;
    add(C.argent + " · " + g, "tool", t.name, short(t.description || "", 70), {tool: t.name, args: {}, ...(t.mutating ? {confirm: true} : {})}, "tool инструмент " + t.group);
  }
  const servers = Object.keys(CAT.mcpServers || {});
  add(C.mcp, "mcp", "Вызвать инструмент MCP", "Сервер → инструмент", servers[0] ? {server: servers[0], confirm: true} : {confirm: true}, "mcp server сервер");
  for (const s of servers) {
    add(C.mcp, "mcp", "MCP «" + s + "»: инструмент", "Выбрать инструмент сервера", {server: s, confirm: true}, "mcp " + s);
    add(C.mcp, "mcp", "MCP «" + s + "»: список инструментов", "Что умеет сервер", {server: s, mode: "list"}, "mcp list " + s);
  }
  add(C.skills, "skill", "Прочитать скилл", "Инструкции скилла в состояние", {}, "skill скилл");
  for (const s of CAT.skills || []) add(C.skills, "skill", "Скилл «" + s.name + "»", short(s.description || "", 70), {skill: s.name}, "skill " + s.name);
  add(C.plugins, "plugin", "Команда плагина", "Выполнить команду", {confirm: true}, "plugin плагин");
  for (const [p, d] of Object.entries(CAT.plugins || {})) {
    for (const c of d.commands || []) add(C.plugins, "plugin", "Плагин: " + c, p, {command: c, confirm: true}, "plugin " + p + " " + c);
  }
  for (const [m, [title, why]] of Object.entries(LEETH_MODE_LABEL)) {
    const pre = {mode: m, ...(m === "filter" || m === "classify" ? {input_from: "items"} : {}), ...(m === "click" ? {input_from: "found", args: {button: ""}} : {}),
      ...(m === "classify" ? {args: {categories: ["категория 1", "категория 2"]}} : {}), ...(m === "fill" ? {args: {dry_run: true}} : {}),
      ...(["fill", "click", "delegate"].includes(m) ? {confirm: true} : {})};
    add(C.leeth, "leeth", "Leeth: " + title.toLowerCase(), why, pre, "leeth лит быстрая модель " + m);
  }
  for (const r of META.rooms || []) {
    if (room && r.name === room.room) continue;
    add(C.rooms, "room", "Комната «" + r.name + "»", short(r.description || "", 70), {room: r.name}, "room комната " + r.name);
  }
  return E;
}
let PAL_ENTRIES = [];
function buildPaletteExt() {
  const p = $("#paletteBody"); p.replaceChildren();
  ensureLists();
  PAL_ENTRIES = paletteEntries();
  const search = el("input", {type: "text", id: "palSearch", placeholder: "Найти блок…", autocomplete: "off", "aria-label": "Поиск блока"});
  p.append(el("div", {class: "palsearch"}, el("div", {class: "box"}, icon("i-search", "sm"), search, el("kbd", {}, "Ctrl+K"))));
  const host = el("div", {id: "palHost"});
  p.append(host);
  const item = (e, hit) => {
    const info = TYPES[e.type === "switch" ? "condition" : e.type] || TYPES.tool;
    const t = e.type === "switch" ? "condition" : e.type;
    const chip = el("span", {class: "chipic"}, icon("i-" + t)); chip.style.background = `var(--c-${t})`; chip.style.color = `var(--k-${t})`;
    return el("button", {class: "pal compact" + (hit ? " hit" : ""), title: info.hint, onclick: () => addFromEntry(e)}, chip,
      el("span", {}, el("b", {}, e.title), e.sub ? el("span", {class: "t"}, e.sub) : null));
  };
  const render = () => {
    host.replaceChildren();
    const q = search.value.trim().toLowerCase();
    if (q) {
      const words = q.split(/\s+/);
      const found = PAL_ENTRIES.filter((e) => words.every((w) => e.kw.toLowerCase().includes(w)));
      if (!found.length) host.append(el("div", {class: "palempty"}, "Ничего не найдено. Попробуйте «цикл», «telegram», «leeth», «выключить»…"));
      found.slice(0, 40).forEach((e, i) => host.append(item(e, i === 0)));
      return;
    }
    const cats = [];
    for (const e of PAL_ENTRIES) if (!cats.includes(e.cat)) cats.push(e.cat);
    const open = new Set(["Логика", "Циклы", "Агент и память", "Люди и уведомления"]);
    for (const c of cats) {
      const items = PAL_ENTRIES.filter((e) => e.cat === c);
      const d = el("details", {class: "palcat"}, el("summary", {}, c, el("span", {class: "n"}, String(items.length))));
      if (open.has(c)) d.setAttribute("open", "");
      for (const e of items) d.append(item(e, false));
      host.append(d);
    }
  };
  search.addEventListener("input", render);
  search.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      const q = search.value.trim().toLowerCase();
      const words = q.split(/\s+/);
      const first = PAL_ENTRIES.find((e) => words.every((w) => e.kw.toLowerCase().includes(w)));
      if (q && first) { addFromEntry(first); search.value = ""; render(); }
    } else if (ev.key === "Escape") { search.value = ""; render(); search.blur(); }
  });
  render();
  p.append(el("h2", {}, "Завершить комнату"));
  const exits = el("div", {class: "exits", id: "exitBox"});
  for (const [name, label, why] of EXIT_PRESETS) {
    exits.append(el("button", {class: "btn sm", title: why + " — выход «" + name + "»", onclick: () => addExit(name)}, icon("i-exit", "sm"), label));
  }
  p.append(exits);
  const custom = inp("", () => {}, {ph: "свой выход…"});
  custom.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && custom.value.trim()) { addExit(custom.value); custom.value = ""; } });
  p.append(el("div", {class: "customexit"}, custom, el("button", {class: "btn sm", "aria-label": "Добавить выход", onclick: () => { if (custom.value.trim()) { addExit(custom.value); custom.value = ""; } }}, icon("i-plus", "sm"))));
  p.append(el("div", {class: "tip"}, "Выберите шаг на схеме — новый блок добавится сразу после него и соединится стрелкой. Ctrl+K — поиск блока."));
}
function addFromEntry(e) {
  if (e.type === "switch") {
    if (!room) { say("Сначала откройте или создайте комнату.", "bad"); return; }
    inspectMode = "switch"; panel(); return;
  }
  addNode(e.type, e.preset);
}
function focusPaletteSearch() {
  const s = $("#palSearch");
  if (s) { s.focus(); s.select(); }
}
