"use strict";
/* Профили MCP-серверов: готовые блоки и простые формы для серверов, чьё устройство мы знаем.

   Сейчас известен один: Sarper (отправляет запросы в веб-ИИ: ChatGPT, Claude, Gemini, DeepSeek, Qwen, Grok).
   Любой другой сервер остаётся доступен как раньше — общим блоком «Вызвать инструмент MCP»; а инструменты
   сервера, которые уже загружены кнопкой «Загрузить», попадают в палитру отдельными блоками.
   Сервер распознаётся по имени («sarper») или по набору инструментов (query_web_llm + start_task).
   Загружается после simple.js; к функциям app.js/ext.js обращается только во время работы. */

const SARPER_AI = [["chatgpt", "ChatGPT"], ["claude", "Claude"], ["gemini", "Gemini"], ["deepseek", "DeepSeek"], ["qwen", "Qwen"], ["grok", "Grok"]];
const SARPER_THINK = [["default", "Как настроено на сайте"], ["off", "Не рассуждать — быстро"], ["low", "Немного"], ["medium", "Средне"], ["high", "Глубоко"]];
const SARPER_BAGS = "regex_all:Bag ID: ([0-9a-f]+)";
const SARPER_ANSWER = "json:task.result.text";
const WRITE_BAG = "write_bag_to_file";

// что умеет каждый инструмент: подпись, поля формы (ключ аргумента → вид поля), что он оставляет
const SARPER_TOOLS = {
  query_web_llm: {
    title: "Спросить веб-ИИ", why: "Отправляет вопрос в веб-ИИ и ждёт ответ текстом",
    kw: "спросить веб ии вопрос совет chatgpt claude gemini deepseek qwen grok",
    preset: {args: {prompt: "{{task}}", domain: "chatgpt"}, save_as: "web_answer"},
    fields: [["prompt", "prompt"], ["domain", "ai"], ["model", "model"], ["reasoning_level", "think"], ["fresh", "fresh"]],
    bags: true,
  },
  start_task: {
    title: "Спросить веб-ИИ в фоне", why: "Запускает вопрос и сразу идёт дальше; ответ забирается блоком «Дождаться ответа»",
    kw: "фон асинхронно start_task запустить задача",
    preset: {args: {prompt: "{{task}}", domain: "chatgpt"}, sets: {job: "json"}},
    fields: [["prompt", "prompt"], ["domain", "ai"], ["domains", "chain"], ["model", "model"], ["reasoning_level", "think"], ["fresh", "fresh"], ["close_tab", "close"]],
    saved: ["json", "Запомнить номер задачи как", "job", "Дальше его берёт «Дождаться ответа веб-ИИ»: {{job.task_id}}."],
  },
  wait_task: {
    title: "Дождаться ответа веб-ИИ", why: "Ждёт, пока задача из «в фоне» закончится, и берёт ответ",
    kw: "дождаться ждать wait_task ответ",
    preset: {args: {task_id: "{{job.task_id}}"}, sets: {web_answer: SARPER_ANSWER}},
    fields: [["task_id", "text", "Какая задача", "{{job.task_id}}", "Номер, который вернул блок «в фоне»."],
             ["timeout_s", "number", "Ждать не дольше, секунд", "100", "Argent ждёт ответ MCP-сервера ограниченное время (см. подсказку выше); если не дождались — блок вернёт «ещё выполняется»."]],
    saved: [SARPER_ANSWER, "Запомнить ответ как", "web_answer", ""],
    bags: true,
  },
  get_task: {
    title: "Посмотреть, готов ли ответ", why: "Проверяет задачу без ожидания: ждёт, идёт или готово",
    kw: "статус задачи get_task готов",
    preset: {args: {task_id: "{{job.task_id}}"}, sets: {task_status: "json:task.status"}},
    fields: [["task_id", "text", "Какая задача", "{{job.task_id}}", ""]],
    saved: ["json:task.status", "Запомнить состояние (queued / running / done / error) как", "task_status", ""],
  },
  cancel_task: {
    title: "Отменить задачу веб-ИИ", why: "Останавливает вопрос, который ещё ждёт или выполняется",
    kw: "отменить cancel_task остановить",
    preset: {args: {task_id: "{{job.task_id}}"}},
    fields: [["task_id", "text", "Какая задача", "{{job.task_id}}", ""]],
  },
  get_status: {
    title: "Готов ли Sarper", why: "Проверяет, подключено ли расширение браузера и какие веб-ИИ открыты",
    kw: "статус готов подключено расширение get_status",
    preset: {args: {}, sets: {sarper_ready: "json:extension_connected"}},
    fields: [],
    saved: ["json:extension_connected", "Запомнить «подключено ли расширение» как", "sarper_ready", "Потом можно ветвиться: «если sarper_ready»."],
  },
  read_bag_content: {
    title: "Прочитать блок кода", why: "Достаёт текст блока кода, который Sarper вынул из ответа",
    kw: "прочитать блок кода bag read_bag_content",
    preset: {args: {bag_id: "{{bag}}"}, save_as: "code_text"},
    fields: [["bag_id", "text", "Номер блока", "{{bag}}", "Берётся из «Достать блоки кода из ответа»; обычно внутри цикла «Для каждого»."]],
  },
  write_bag_to_file: {
    title: "Сохранить блок кода в файл", why: "Записывает блок кода из ответа веб-ИИ в файл на диске",
    kw: "сохранить блок кода файл записать bag write_bag_to_file",
    preset: {args: {bag_id: "{{bag}}", target_path: ""}, confirm: true},
    fields: [["bag_id", "text", "Номер блока", "{{bag}}", "Обычно внутри цикла «Для каждого» по номерам блоков."],
             ["target_path", "path", "Куда сохранить (полный путь)", "C:/проект/файл.py", "Файл будет создан или перезаписан — поэтому по умолчанию блок спрашивает вас."]],
  },
};

// ── какой это сервер ───────────────────────────────────────────────────────────────
function mcpProfileOf(server) {
  if (!server || !CAT.mcpServers || !(server in CAT.mcpServers)) return null;
  if (/sarper/i.test(server)) return "sarper";
  const names = (CAT.mcpTools[server] || []).map((t) => t.name);
  return names.includes("query_web_llm") && names.includes("start_task") ? "sarper" : null;
}
const profiledServers = () => Object.keys(CAT.mcpServers || {}).filter((s) => mcpProfileOf(s));

function uniqueSaveAs(base) {
  if (!base || typeof room === "undefined" || !room) return base;
  const used = new Set(room.nodes.map((n) => n.save_as).filter(Boolean));
  let name = base, i = 2;
  while (used.has(name)) name = base + "_" + (i++);
  return name;
}

// ── палитра ────────────────────────────────────────────────────────────────────────
function profilePaletteEntries(add) {
  for (const server of Object.keys(CAT.mcpServers || {})) {
    const profile = mcpProfileOf(server);
    if (profile === "sarper") {
      const cat = "Веб-ИИ · " + server;
      // читающие вызовы идут без вопроса (вы сами поставили блок); запись на диск спрашивает
      const mcp = (tool, extra) => ({server, tool, confirm: false, ...JSON.parse(JSON.stringify(SARPER_TOOLS[tool].preset)), ...(extra || {})});
      const q = SARPER_TOOLS.query_web_llm;
      add(cat, "mcp", q.title, q.why, mcp("query_web_llm"), q.kw + " " + server);
      for (const [domain, label] of SARPER_AI) {
        add(cat, "mcp", "Спросить " + label, "Вопрос уйдёт в " + label, mcp("query_web_llm", {args: {prompt: "{{task}}", domain}}), "спросить " + label + " " + domain);
      }
      add(cat, "mcp", "Спросить глубоко", "Веб-ИИ рассуждает подробно", mcp("query_web_llm", {args: {prompt: "{{task}}", domain: "chatgpt", reasoning_level: "high"}}), "глубоко подробно рассуждения high думать");
      add(cat, "mcp", "Спросить и вынуть блоки кода", "Номера блоков кода — в {{code_blocks}}",
        mcp("query_web_llm", {sets: {code_blocks: SARPER_BAGS}}), "блоки кода код bag вынуть");
      for (const t of ["start_task", "wait_task", "get_task", "cancel_task", "get_status", "read_bag_content", "write_bag_to_file"]) {
        const s = SARPER_TOOLS[t];
        add(cat, "mcp", s.title, s.why, mcp(t), s.kw + " " + server);
      }
      add(cat, "agent", "Советник: спрашивает веб-ИИ, сам ничего не меняет", "Только вопросы в веб-ИИ; файлы и команды запрещены",
        {role: "Researcher", tools: [], mcp: [server], ignore: ["mcp:" + server + "." + WRITE_BAG], save_as: "advice",
          instruction: "Задача: {{task}}\n\n" + webAiParagraph(server, {domain: "chatgpt", advisor: true})}, "советник совет консультант веб ии агент");
      add(cat, "agent", "Исполнитель, которому помогает веб-ИИ", "Работает сам и при сомнениях советуется",
        {role: "Coder", mcp: [server], save_as: "result",
          instruction: "Задача: {{task}}\n\n" + webAiParagraph(server, {domain: "chatgpt", advisor: false})}, "исполнитель помощник веб ии агент код");
    } else if ((CAT.mcpTools[server] || []).length) {
      const cat = "MCP · " + server;
      for (const t of CAT.mcpTools[server]) {
        add(cat, "mcp", t.name, short(t.description || "", 70), {server, tool: t.name, args: {}, confirm: true}, "mcp " + server + " " + t.name);
      }
    }
  }
  // сервер без известного профиля: инструменты в палитру по кнопке «загрузить»
  for (const server of Object.keys(CAT.mcpServers || {})) {
    if (mcpProfileOf(server) || (CAT.mcpTools[server] || []).length) continue;
    add("MCP · " + server, "mcp", "Показать инструменты «" + server + "» в палитре", "Подключится к серверу и добавит по блоку на инструмент",
      {server, confirm: true}, "mcp " + server + " инструменты загрузить показать", async () => {
        say("Подключаюсь к «" + server + "»…", "");
        await loadMcpTools(server, true);
        buildPalette();
        say(CAT.mcpState[server] === "ok" ? "Инструменты «" + server + "» добавлены в палитру."
          : "Не удалось получить инструменты «" + server + "»: " + CAT.mcpState[server], CAT.mcpState[server] === "ok" ? "ok" : "bad");
      });
  }
}

// ── форма блока ────────────────────────────────────────────────────────────────────
function profileNodeForm(side, n, upd) {
  if (n.type !== "mcp" || n.mode === "list" || mcpProfileOf(n.server) !== "sarper") return false;
  n.args = n.args || {};
  const known = Object.keys(SARPER_TOOLS);
  if (!n.tool || !known.includes(n.tool)) {
    if (n.tool) return false;                       // чужой инструмент сервера — общая форма
    side.append(field("Что сделать через Sarper", select([["", "— выберите —"], ...known.map((k) => [k, SARPER_TOOLS[k].title])], "", (v) => {
      const s = SARPER_TOOLS[v];
      n.tool = v; n.args = JSON.parse(JSON.stringify(s.preset.args || {}));
      if (s.preset.sets) n.sets = JSON.parse(JSON.stringify(s.preset.sets));
      if (s.preset.save_as && !n.save_as) n.save_as = uniqueSaveAs(s.preset.save_as);
      if (s.preset.confirm) n.confirm = true;
      upd("sarpertool", true); panel();
    })));
    allSettings(side, (body) => mcpNodeForm(body, n, upd));
    return true;
  }
  const spec = SARPER_TOOLS[n.tool];
  const a = n.args;
  const set = (key, v) => {
    if (v === "" || v == null || v === false || (Array.isArray(v) && !v.length)) delete a[key]; else a[key] = v;
    upd("sp" + key + n.id);
  };
  const wait = (typeof META !== "undefined" && META.vocab && META.vocab.mcp_call_timeout) || 120;
  if (n.tool === "query_web_llm") {
    side.append(el("p", {class: "hint"}, `Argent ждёт ответ MCP-сервера до ${Math.round(wait)} с (настройка mcp_call_timeout). Если веб-ИИ думает дольше, возьмите «Спросить веб-ИИ в фоне» и «Дождаться ответа».`));
  }
  side.append(field("Что делает блок", select(known.map((k) => [k, SARPER_TOOLS[k].title]), n.tool, (v) => {
    const s = SARPER_TOOLS[v];
    n.tool = v; n.args = JSON.parse(JSON.stringify(s.preset.args || {}));
    n.sets = s.preset.sets ? JSON.parse(JSON.stringify(s.preset.sets)) : {};
    n.confirm = !!s.preset.confirm;
    upd("sarpertool", true); panel();
  }), spec.why + "."));
  for (const [key, kind, label, ph, hint] of spec.fields) {
    const cur = a[key];
    if (kind === "prompt") {
      side.append(field("Что спросить", withInserter(area(cur || "", (v) => set(key, v), 5)),
        "Пишите как человеку. {{task}} — задача комнаты; кнопки вставляют ответы прошлых шагов. Вопрос уходит во внешний веб-сервис."));
    } else if (kind === "ai") {
      const opts = [...SARPER_AI];
      if (cur && !opts.some(([k]) => k === cur)) opts.push([cur, cur]);
      side.append(field("Кого спросить", select(opts, cur || "chatgpt", (v) => set(key, v)), "Нужная вкладка должна быть открыта в браузере с расширением Sarper."));
    } else if (kind === "chain") {
      side.append(field("Запасные ИИ по очереди", inp(Array.isArray(cur) ? cur.join(", ") : "", (v) => set(key, v.split(",").map((x) => x.trim()).filter(Boolean)),
        {ph: "например: claude, gemini"}), "Если первый не ответил — следующий. Пусто — только основной."));
    } else if (kind === "model") {
      side.append(field("Конкретная модель", inp(cur || "", (v) => set(key, v.trim()), {ph: "например: Gemini 2.5 Pro"}),
        "Пусто — какая выбрана на сайте. Имя пишется так, как в выпадающем списке сайта."));
    } else if (kind === "think") {
      side.append(field("Как глубоко думать", select(SARPER_THINK, cur || "default", (v) => set(key, v === "default" ? "" : v)),
        "Работает у моделей с режимом рассуждения (например, Gemini); у остальных игнорируется."));
    } else if (kind === "fresh") {
      side.append(check("Новая вкладка без прошлой переписки", !!cur, (v) => set(key, v), "Выключено — ответ может опираться на предыдущий разговор в этой вкладке."));
    } else if (kind === "close") {
      side.append(check("Закрыть вкладку после ответа", !!cur, (v) => set(key, v)));
    } else if (kind === "number") {
      side.append(field(label, inp(cur ?? "", (v) => set(key, v === "" ? "" : Number(v)), {type: "number", ph}), hint));
    } else {
      side.append(field(label, withInserter(inp(cur || "", (v) => set(key, v), {ph})), hint));
    }
  }
  if (spec.saved) {
    const [derived, label, ph, hint] = spec.saved;
    const key = Object.keys(n.sets || {}).find((k) => n.sets[k] === derived) || "";
    const box = el("div", {class: "hint"}, hint || "");
    side.append(el("div", {class: "field"}, el("label", {}, label),
      inp(key, (v) => {
        const name = v.trim();
        const rest = {};
        for (const [k, x] of Object.entries(n.sets || {})) if (x !== derived) rest[k] = x;
        n.sets = name && /^[A-Za-z_]\w*$/.test(name) ? {[name]: derived, ...rest} : rest;
        upd("sparsaved" + n.id);
      }, {ph}), box));
  }
  if (spec.bags) {
    const key = Object.keys(n.sets || {}).find((k) => n.sets[k] === SARPER_BAGS) || "";
    side.append(check("Достать блоки кода из ответа", !!key, (v) => {
      const rest = {};
      for (const [k, x] of Object.entries(n.sets || {})) if (x !== SARPER_BAGS) rest[k] = x;
      n.sets = v ? {code_blocks: SARPER_BAGS, ...rest} : rest;
      upd("bags" + n.id, true); panel();
    }, "Большие блоки кода Sarper не присылает текстом, а прячет в «сумку» под номером. Номера попадут в {{" + (key || "code_blocks") +
      "}}: пройдите по ним блоком «Для каждого» и сохраните каждый блоком «Сохранить блок кода в файл»."));
  }
  if (!spec.saved || spec.preset.save_as) saveAsField(side, n, upd);
  side.append(check("Спросить вас перед запуском", !!n.confirm, (v) => { n.confirm = v; upd("confirm"); },
    n.tool === WRITE_BAG ? "Блок пишет на диск." : "Вопрос уходит во внешний сервис — включите, если в нём может быть личное."));
  allSettings(side, (body) => mcpNodeForm(body, n, upd));
  return true;
}

// ── агент, которому разрешён веб-ИИ ────────────────────────────────────────────────
const WEBAI_RE = /^Веб-ИИ \([^)]*\):.*$/m;
function webAiParagraph(server, o) {
  const tail = o.advisor
    ? " Сам ничего не меняй: ни файлов, ни команд. Верни совет и найденный код текстом."
    : " Ответ веб-ИИ — совет: проверь его, прежде чем применять.";
  const bits = [`domain="${o.domain || "chatgpt"}"`];
  if (o.model) bits.push(`model="${o.model}"`);
  if (o.think && o.think !== "default") bits.push(`reasoning_level="${o.think}"`);
  return `Веб-ИИ (${server}): когда нужен совет или чужой взгляд, спроси через MCP-сервер «${server}», инструмент query_web_llm — ` +
    bits.join(", ") + "." + tail;
}
function webAiState(n) {
  const text = n.instruction || "";
  const m = text.match(WEBAI_RE);
  const para = m ? m[0] : "";
  const grab = (re) => { const x = para.match(re); return x ? x[1] : ""; };
  return {has: !!m, domain: grab(/domain="([^"]*)"/) || "chatgpt", model: grab(/model="([^"]*)"/), think: grab(/reasoning_level="([^"]*)"/) || "default"};
}
function writeWebAi(n, server, o) {
  const para = webAiParagraph(server, o);
  const text = n.instruction || "";
  n.instruction = WEBAI_RE.test(text) ? text.replace(WEBAI_RE, para) : (text.trim() ? text.replace(/\s+$/, "") + "\n\n" : "") + para;
}
function dropWebAi(n) {
  n.instruction = (n.instruction || "").replace(WEBAI_RE, "").replace(/\n{3,}/g, "\n\n").replace(/\s+$/, "");
}
function profileAgentCard(side, n, upd) {
  const servers = profiledServers();
  if (!servers.length) return;
  const used = (n.mcp || []).find((s) => servers.includes(s));
  const server = used || servers[0];
  const st = webAiState(n);
  const advisor = Array.isArray(n.tools) && n.tools.length === 0;
  const body = el("div", {class: "profilecard"});
  const refresh = () => { upd("webai" + n.id, true); panel(); };
  side.append(el("div", {class: "lbl"}, "Веб-ИИ"), body);
  body.append(check("Может спрашивать веб-ИИ (через «" + server + "»)", !!used, (v) => {
    if (v) {
      n.mcp = [...new Set([...(n.mcp || []), server])];
      writeWebAi(n, server, {domain: st.domain, model: st.model, think: st.think, advisor});
    } else {
      n.mcp = (n.mcp || []).filter((s) => s !== used);
      if (!n.mcp.length) delete n.mcp;
      n.ignore = (n.ignore || []).filter((r) => r !== "mcp:" + used + "." + WRITE_BAG);
      if (!n.ignore.length) delete n.ignore;
      dropWebAi(n);
    }
    refresh();
  }, "Агент сможет отправлять вопросы в ChatGPT, Claude, Gemini и др. Другие MCP-серверы ему при этом станут недоступны."));
  if (!used) return;
  const rewrite = (patch) => { writeWebAi(n, used, {...st, advisor, ...patch}); refresh(); };
  body.append(field("Кого спрашивать", select(SARPER_AI.some(([k]) => k === st.domain) ? SARPER_AI : [...SARPER_AI, [st.domain, st.domain]],
    st.domain, (v) => rewrite({domain: v}))));
  const model = el("input", {type: "text", value: st.model, placeholder: "например: Gemini 2.5 Pro"});
  model.addEventListener("change", () => rewrite({model: model.value.trim()}));
  body.append(field("Конкретная модель", model, "Пусто — какая выбрана на сайте."));
  body.append(field("Как глубоко думать", select(SARPER_THINK, st.think, (v) => rewrite({think: v}))));
  body.append(check("Только советоваться: ничего не менять самому", advisor, (v) => {
    const rule = "mcp:" + used + "." + WRITE_BAG;
    n.ignore = (n.ignore || []).filter((r) => r !== rule);
    if (v) { n.tools = []; n.role = "Researcher"; n.ignore.push(rule); }
    else { delete n.tools; n.role = "Coder"; }
    if (!n.ignore.length) delete n.ignore;
    writeWebAi(n, used, {...st, advisor: v});
    refresh();
  }, "Это не просьба в тексте, а запрет на самом деле: у агента остаются только вопросы в веб-ИИ — ни файлов, ни команд, ни записи блоков кода."));
}
