"use strict";
/* Вид «Стек»: комната как столбик блоков, в духе Scratch.

   Схема (граф) остаётся главной и единственной правдой: стек строится из неё каждый раз заново и
   ничего своего не хранит. Что здесь можно: выбрать блок (открывается та же панель свойств), поправить его
   главный текст прямо в карточке, вставить новый блок в нужное место, убрать блок со склейкой, переставить
   блок вверх/вниз. Всё, что не укладывается в столбик (произвольные переходы), показано метками
   «↩ к шагу …» и правится на «Схеме».

   Как граф раскладывается в столбик (buildStack, чистая функция — её же гоняют тесты под node):
     * цепочка шагов со стрелкой «всегда» — просто блоки один под другим;
     * шаг с несколькими стрелками — «развилка»: по рукаву на стрелку; если рукава сходятся в общий шаг,
       он стоит ниже развилки, а не повторяется в каждом рукаве;
     * шаг, лежащий на цикле, — «повторять»: внутри то, что идёт по кругу, ниже — выход из цикла;
     * «параллельно» — рукава-ветки, ниже — шаг слияния; выход — «закончить».
   Загружается после simple.js/profiles.js; к функциям app.js обращается только во время работы. */

// ── раскладка графа в столбик (без DOM) ────────────────────────────────────────────
function stackKind(e) {
  return e.on_error ? "error" : e.while != null ? "while" : e.if != null ? "if" : e.else ? "else" : "uncond";
}

function buildStack(room) {
  const nodes = new Map(room.nodes.map((n) => [n.id, n]));
  const out = new Map();
  for (const e of room.edges) { if (!out.has(e.from)) out.set(e.from, []); out.get(e.from).push(e); }
  const exitOf = (to) => (String(to).startsWith("EXIT:") ? String(to).slice(5) : null);
  const targets = (id) => (out.get(id) || []).map((e) => e.to).filter((t) => nodes.has(t));

  const reachMemo = new Map();
  function reach(id) {                     // что достижимо из шага (сам шаг — только если он на цикле)
    if (reachMemo.has(id)) return reachMemo.get(id);
    const seen = new Set(), stack = [...targets(id)];
    while (stack.length) {
      const t = stack.pop();
      if (seen.has(t)) continue;
      seen.add(t);
      for (const n of targets(t)) if (!seen.has(n)) stack.push(n);
    }
    reachMemo.set(id, seen);
    return seen;
  }
  // то же, но не заходя в уже открытые циклы: шаг, чей круг проходит через них, свой «повтор» не открывает
  const avoidMemo = new Map();
  function reachAvoiding(id, heads) {
    const key = id + "|" + [...heads].sort().join(",");
    if (avoidMemo.has(key)) return avoidMemo.get(key);
    const seen = new Set(), stack = targets(id).filter((t) => !heads.has(t));
    while (stack.length) {
      const t = stack.pop();
      if (seen.has(t)) continue;
      seen.add(t);
      for (const n of targets(t)) if (!seen.has(n) && !heads.has(n)) stack.push(n);
    }
    avoidMemo.set(key, seen);
    return seen;
  }
  function firstCommon(from, ts) {         // ближайший к `from` шаг, в который сходятся все рукава
    const sets = ts.map((t) => new Set([t, ...reach(t)]));
    const common = [...sets[0]].filter((x) => sets.every((s) => s.has(x)));
    if (!common.length) return null;
    const want = new Set(common), seen = new Set([from]), queue = [from];
    while (queue.length) {
      const id = queue.shift();
      for (const t of targets(id)) {
        if (want.has(t)) return t;
        if (!seen.has(t)) { seen.add(t); queue.push(t); }
      }
    }
    return common[0];
  }

  const rendered = new Set();
  function walk(start, stop, heads) {
    const items = [];
    let cur = start;
    while (cur != null) {
      const exit = exitOf(cur);
      if (exit !== null) { items.push({kind: "exit", name: exit}); break; }
      if (!nodes.has(cur)) { items.push({kind: "missing", id: cur}); break; }
      if (heads.has(cur)) { items.push({kind: "again", id: cur}); break; }     // замыкает цикл, который уже открыт выше
      if (stop.has(cur)) break;
      if (rendered.has(cur)) { items.push({kind: "goto", id: cur}); break; }
      rendered.add(cur);
      const node = nodes.get(cur);
      const outs = out.get(cur) || [];
      const back = reachAvoiding(cur, heads);
      const loopEdges = back.has(cur)
        ? outs.filter((e) => nodes.has(e.to) && !heads.has(e.to) && (e.to === cur || reachAvoiding(e.to, heads).has(cur)))
        : [];
      const rest = outs.filter((e) => !loopEdges.includes(e));
      const head = {kind: "block", id: cur, edge: null};
      if (loopEdges.length) {
        const inner = new Set([...heads, cur]);
        const arms = loopEdges.map((e) => ({edge: e, kind: stackKind(e), items: walk(e.to, stop, inner)}));
        const body = arms.length === 1          // один путь по кругу: условие уже в подписи «Повторять, пока …»
          ? arms[0].items
          : [{kind: "branch", id: cur, arms}];
        items.push({kind: "repeat", id: cur, head, body, edges: loopEdges, exits: rest});
      } else {
        items.push(head);
      }
      if (!rest.length) break;
      if (rest.length === 1 && stackKind(rest[0]) === "uncond") {
        if (!loopEdges.length) head.edge = rest[0];
        cur = rest[0].to;
        continue;
      }
      const parallel = node.type === "parallel";
      const live = rest.map((e) => e.to).filter((t) => nodes.has(t));
      let merge = parallel && node.join && nodes.has(node.join) ? node.join : (live.length > 1 ? firstCommon(cur, live) : null);
      const armStop = new Set([...stop]);
      if (merge) armStop.add(merge);
      const arms = rest.map((e) => ({
        edge: e, kind: parallel ? "parallel" : stackKind(e),
        items: walk(e.to, armStop, heads),
      }));
      items.push({kind: "branch", id: cur, parallel, arms});
      if (merge && !stop.has(merge)) { cur = merge; continue; }
      break;
    }
    return items;
  }

  const entry = room.entry || (room.nodes[0] && room.nodes[0].id);
  const items = walk(entry, new Set(), new Set());
  const loose = room.nodes.filter((n) => !rendered.has(n.id)).map((n) => n.id);   // шаги, до которых не дойти
  return {items, loose, entry};
}

if (typeof module !== "undefined" && module.exports) module.exports = {buildStack, stackKind};

// ── показ и правка (только в браузере) ─────────────────────────────────────────────
let PENDING_INSERT = null;     // куда встанет следующий блок из палитры: {edge} | {after} | {entry: true}

function takePendingInsert() {
  const p = PENDING_INSERT;
  PENDING_INSERT = null;
  return p;
}
function stackAskInsert(where, text) {
  PENDING_INSERT = {...where, text};
  renderStack();
  const s = document.getElementById("palSearch");
  if (s) { s.focus(); s.select(); }
}

// у блока есть главное поле, которое удобно править прямо в карточке
function stackPrimary(n) {
  const a = n.args || (n.args = {});
  switch (n.type) {
    case "agent": return {label: "Что сделать", get: () => n.instruction || "", set: (v) => { n.instruction = v; }};
    case "human_pause": return {label: "Вопрос", get: () => n.prompt || "", set: (v) => { n.prompt = v; }};
    case "notify": return {label: "Сообщение", get: () => n.message || "", set: (v) => { n.message = v; }};
    case "hint": return {label: "Подсказка", get: () => n.message || "", set: (v) => { n.message = v; }};
    case "condition": return {label: "Условие", get: () => n.condition || "", set: (v) => { n.condition = v; }};
    case "assert": return {label: "Проверить, что", get: () => n.condition || "", set: (v) => { n.condition = v; }};
    case "http": return {label: "Адрес", get: () => a.url || "", set: (v) => { a.url = v.trim(); }, one: true};
    case "git":
      return n.mode === "commit" ? {label: "Сообщение коммита", get: () => n.message || "", set: (v) => { n.message = v; }, one: true} : null;
    case "tool":
      return n.tool === "run_command" ? {label: "Команда", get: () => a.command || "", set: (v) => { a.command = v; }, one: true} : null;
    default: return null;
  }
}

function stackSingleOut(id) {
  const outs = room.edges.filter((e) => e.from === id);
  return outs.length === 1 && edgeKind(outs[0]) === "uncond" ? outs[0] : null;
}
function stackSingleIn(id) {
  const ins = room.edges.filter((e) => e.to === id);
  return ins.length === 1 && edgeKind(ins[0]) === "uncond" ? ins[0] : null;
}
// P → X → Q → R  становится  P → Q → X → R   (только для простой цепочки)
function stackCanMoveDown(id) {
  const into = stackSingleIn(id), outX = stackSingleOut(id);
  if (!into || !outX || !nodeById(outX.to)) return false;
  const q = outX.to;
  return q !== id && !!stackSingleOut(q) && !!stackSingleIn(q) && nodeById(into.from) !== undefined;
}
function stackMoveDown(id) {
  if (!stackCanMoveDown(id)) return false;
  const into = stackSingleIn(id), outX = stackSingleOut(id);
  const q = outX.to, outQ = stackSingleOut(q);
  const r = outQ.to;
  into.to = q;            // P → Q
  outQ.to = id;           // Q → X
  outX.to = r;            // X → R
  return true;
}
function stackCanMoveUp(id) {
  const into = stackSingleIn(id);
  return !!into && nodeById(into.from) !== undefined && stackCanMoveDown(into.from);
}
function stackMoveUp(id) {
  const into = stackSingleIn(id);
  return !!into && stackMoveDown(into.from);
}
// убрать блок и сомкнуть цепочку
function stackRemove(id) {
  if (room.nodes.length <= 1) { say("Нельзя удалить последний шаг.", "bad"); return false; }
  const outs = room.edges.filter((e) => e.from === id);
  const ins = room.edges.filter((e) => e.to === id);
  const through = outs.length === 1 && edgeKind(outs[0]) === "uncond" && outs[0].to !== id ? outs[0] : null;
  if (through) for (const e of ins) e.to = through.to;
  room.nodes = room.nodes.filter((n) => n.id !== id);
  room.edges = room.edges.filter((e) => e.from !== id && !(e.to === id));
  delete layout[id];
  if (room.entry === id) room.entry = through && nodeById(through.to) ? through.to : room.nodes[0].id;
  if (sel && sel.kind === "node" && sel.id === id) sel = null;
  return true;
}

// ── DOM ─────────────────────────────────────────────────────────────────────────────
function renderStack() {
  const host = document.getElementById("stackBody");
  if (!host) return;
  const keep = host.parentElement ? host.parentElement.scrollTop : 0;
  host.replaceChildren();
  if (!room) { host.append(el("p", {class: "muted"}, "Откройте или создайте комнату.")); return; }
  host.append(el("p", {class: "hint stackhint"}, "Комната столбиком. Нажмите на блок — справа его настройки; «+» вставляет новый блок (выбирайте в палитре слева). Сложные переходы правятся на «Схеме»."));
  if (PENDING_INSERT) {
    host.append(el("div", {class: "stackpending"}, el("b", {}, "Выберите блок в палитре слева"), " — он встанет: " + PENDING_INSERT.text + ".",
      el("button", {class: "btn sm", onclick: () => { PENDING_INSERT = null; renderStack(); }}, "Отмена")));
  }
  const tree = buildStack(room);
  const slot = (where, text) => el("button", {class: "sslot", title: "Вставить блок " + text, "aria-label": "Вставить блок " + text,
    onclick: (ev) => { ev.stopPropagation(); stackAskInsert(where, text); }}, "+");
  host.append(el("div", {class: "sstart"}, "Старт", slot({entry: true}, "в самое начало")));
  host.append(stackList(tree.items, slot));
  if (tree.loose.length) {
    host.append(el("div", {class: "sloose"}, el("b", {}, "Не соединены со стартом: "),
      tree.loose.map((id) => el("button", {class: "btn sm ghost", onclick: () => { sel = {kind: "node", id}; panel(); renderStack(); }}, id))));
  }
  if (host.parentElement) host.parentElement.scrollTop = keep;
}

function stackList(items, slot) {
  const col = el("div", {class: "scol"});
  for (const it of items) col.append(stackItem(it, slot));
  return col;
}

function stackItem(it, slot) {
  if (it.kind === "exit") {
    return el("div", {class: "sexit"}, icon("i-exit", "sm"), "Закончить: «" + it.name + "»" +
      (((room.next || {})[it.name] && (room.next[it.name] !== "END")) ? " → " + (room.next[it.name] === "TRIAGE" ? "на разбор" : "комната «" + room.next[it.name] + "»") : ""));
  }
  if (it.kind === "goto") {
    return el("button", {class: "sgoto", title: "Перейти к этому блоку", onclick: () => { sel = {kind: "node", id: it.id}; panel(); renderStack(); }},
      "↩ дальше как в шаге «" + it.id + "»");
  }
  if (it.kind === "again") {
    return el("button", {class: "sgoto", title: "К началу цикла", onclick: () => { sel = {kind: "node", id: it.id}; panel(); renderStack(); }},
      "↻ снова к началу цикла «" + it.id + "»");
  }
  if (it.kind === "missing") return el("div", {class: "sgoto bad"}, "Стрелка ведёт в несуществующий шаг «" + it.id + "»");
  if (it.kind === "block") return stackBlock(it, slot);
  if (it.kind === "repeat") {
    const inner = el("div", {class: "sinner"}, stackBlock(it.head, slot, true), stackList(it.body, slot));
    return el("div", {class: "srepeat"}, el("div", {class: "scap"}, icon("i-loop", "sm"), stackRepeatWords(it)), inner,
      el("div", {class: "sbottom"}, "и снова с начала блока «" + it.id + "»"));
  }
  if (it.kind === "branch") {
    const wrap = el("div", {class: "sbranch" + (it.parallel ? " par" : "")},
      el("div", {class: "scap"}, icon(it.parallel ? "i-parallel" : "i-condition", "sm"), it.parallel ? "Одновременно" : "Развилка"));
    it.arms.forEach((arm, i) => {
      const words = it.parallel ? "ветка " + (i + 1) : (arm.kind === "uncond" ? "иначе" : edgeWords(arm.edge));
      const body = el("div", {class: "sarm"}, el("div", {class: "slabel"}, words, slot({edge: arm.edge}, "в начало этой ветки")),
        stackList(arm.items, slot));
      wrap.append(body);
    });
    return wrap;
  }
  return el("div", {});
}

// «Повторять…» словами: что удерживает цикл (или что его заканчивает)
function stackRepeatWords(it) {
  const loop = it.edges.length === 1 ? it.edges[0] : null;
  const kind = loop ? edgeKind(loop) : "";
  if (kind === "if") return "Повторять, пока " + condWords(loop.if);
  if (kind === "while") return "Повторять, пока " + condWords(loop.while) + " (не более " + loop.max + " раз)";
  if (kind === "else") {
    const stop = (it.exits || []).find((e) => edgeKind(e) === "if");
    if (stop) return "Повторять, пока не " + condWords(stop.if);
  }
  return "Повторять";
}

function stackBlock(it, slot, inRepeat) {
  const n = nodeById(it.id);
  if (!n) return el("div", {class: "sgoto bad"}, "Нет шага «" + it.id + "»");
  const info = TYPES[n.type] || {icon: "i-tool", name: n.type};
  const card = el("div", {class: "sblock" + (sel && sel.kind === "node" && sel.id === n.id ? " sel" : "") + (n.disabled ? " off" : "") +
    (runMap[n.id] ? " run-" + runMap[n.id] : "") + (inRepeat ? " head" : ""), "data-id": n.id});
  card.style.setProperty("--bc", `var(--c-${n.type})`);
  card.style.setProperty("--bk", `var(--k-${n.type})`);
  const chip = el("span", {class: "chipic"}, icon(info.icon || "i-" + n.type));
  chip.style.background = `var(--c-${n.type})`; chip.style.color = `var(--k-${n.type})`;
  const act = (title, text, fn, disabled) => el("button", {class: "btn icon ghost sm", title, "aria-label": title, disabled: disabled || null,
    onclick: (ev) => { ev.stopPropagation(); fn(); }}, text);
  const tools = el("span", {class: "sacts"},
    act("Выше", "↑", () => { if (stackMoveUp(n.id)) touch("stackup", true); }, !stackCanMoveUp(n.id)),
    act("Ниже", "↓", () => { if (stackMoveDown(n.id)) touch("stackdown", true); }, !stackCanMoveDown(n.id)),
    act("Убрать блок (цепочка сомкнётся)", "✕", () => { if (stackRemove(n.id)) touch("stackrm", true); }));
  card.append(el("div", {class: "shead"}, chip, el("b", {}, n.id), el("span", {class: "muted"}, info.name || n.type), tools));
  card.append(el("div", {class: "sdesc"}, describeNode(n)));
  const prim = stackPrimary(n);
  if (prim) {
    const common = {class: "sinline", "aria-label": prim.label, placeholder: prim.label};
    const ctl = prim.one ? el("input", {...common, type: "text", value: prim.get()}) : el("textarea", {...common, rows: 2}, prim.get());
    ctl.addEventListener("click", (ev) => ev.stopPropagation());
    ctl.addEventListener("input", () => { prim.set(ctl.value); touch("stack" + n.id); });
    card.append(ctl);
  }
  card.addEventListener("click", () => { sel = {kind: "node", id: n.id}; inspectMode = null; panel(); renderStack(); });
  const wrap = el("div", {class: "sitem"}, card);
  if (it.edge) wrap.append(el("div", {class: "sjoin"}, slot({edge: it.edge}, "после «" + n.id + "»")));
  else if (!inRepeat && !room.edges.some((e) => e.from === n.id)) wrap.append(el("div", {class: "sjoin"}, slot({after: n.id}, "после «" + n.id + "»")));
  return wrap;
}
