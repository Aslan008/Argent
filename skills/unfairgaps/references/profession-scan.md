# Reference: profession-scan

Operation-specific details for the `unfairgaps profession-scan` operation.

**This operation uses a different protocol from the other four.** The 4-phase event-collection protocol in `../SKILL.md` is for industry / idea / site / customer-pains operations. `profession-scan` uses a **3-phase regulatory-profile protocol** described below, because the source-of-truth for professional pains is regulation + daily reality, not court filings.

**Native-only.** No CLI path. No Perplexity. Always WebSearch + WebFetch.

**Country-aware.** Works for any country - the SCOPE queries and the pain generator both accept `{country}` as a parameter. The shipped US dataset (`data/professions/us/`) is a reference example. Quality is highest for countries with strong online regulatory presence (US, UK, DE, FR, CA, AU, NL, etc.) and degrades gracefully for low-online-coverage countries (mark as `coverage_gap` in SCOPE).

## Input parsing

From the user request extract:
- `profession` (required): free-text occupation name. Examples: "lawyers", "accountants", "construction managers", "Rechtsanwalt", "адвокат", "comptable", "Steuerberater". Ask if missing.
- `country` (required): ISO 2-letter or full name. Examples: US, DE, KZ, UK, FR, AU, BR, IN. Default to user-stated country; ask only if truly ambiguous.
- `profession_id` (derived): `<cc>-<slug>`. Examples: "lawyers" + US -> `us-lawyers`, "Elektriker" + DE -> `de-elektriker`, "civil engineers" + UK -> `uk-civil-engineers`.

Before starting, check the dataset: `ls data/professions/<cc>/pains/<profession_id>.json`. If a pain bundle already exists, ask the user whether they want to (a) re-generate fresh, (b) load and display the existing bundle, or (c) compare existing vs fresh. Default to (b) - faster, no fetches needed.

Currently the shipped dataset covers 130 US profiles + 25 US pain bundles. Other countries: empty - the operation will generate fresh.

## Phase 1 - SCOPE Builder

Use the prompt in `prompts/profession-scan/01-scope-builder.md`. Run 7 WebSearch queries (one per SCOPE domain). For each query: 1-2 searches + 2-4 fetches on the highest-scoring local-regulator / `.gov` analog / professional-association sites.

**Country adaptation** (mandatory pre-step in the prompt):
- Pick primary retrieval languages (e.g., US -> [en], DE -> [de, en], KZ -> [ru, kk], FR -> [fr], JP -> [ja])
- Identify country-specific regulators (IRS/OSHA for US; HMRC/HSE for UK; Finanzamt/BAuA for DE; URSSAF/Inspection du travail for FR; КГД/Минтруда for KZ; etc.)
- Identify country-specific statutory citation style (USC/CFR for US; BGB/StGB for DE; НК РК/КоАП РК for KZ; Code général des impôts for FR; etc.)
- Identify country-specific trusted domains (.gov for US, .gov.uk for UK, .gouv.fr for FR, .gov.kz for KZ, .europa.eu for EU-level, etc.)

**Fetch budget: 21 fetches total** (3 per domain x 7 domains). Hard cap; if exhausted with thin sections, emit `coverage_gap` notes and proceed to Phase 2.

**Trusted domain priority** (apply when scoring search results):
- `+3` for `.gov` analog of target country (.gov US, .gov.uk UK, .gouv.fr FR, .gov.kz KZ, .bund.de DE, etc.)
- `+3` for canonical legal databases (law.cornell.edu, legislation.gov.uk, legifrance.gouv.fr, gesetze-im-internet.de, adilet.zan.kz)
- `+2` for primary professional associations of the country (ABA/AICPA US, ICAEW UK, BStBK/IDW DE, Ordres FR, Палаты KZ)
- `+1` for industry trade press with named professionals/firms
- `0` for general career-advice blogs and SEO-driven listicles

**Hard rules:**
1. Each SCOPE section must include `sources[]` with `name`, `url`, `fact`, `language`
2. All amounts/rates must be specific - "EUR 5,000-50,000 per violation (~USD 5,400-54,000)" beats "fines can be significant"
3. **Native-language queries are mandatory** for non-English countries. English-only on a non-English country = fake "no evidence". Run both passes.
4. If a section is too thin to use (e.g., career_psychology returns nothing usable for a small profession in a small country), include the section with `coverage_gap: true` and proceed

Output: single JSON profile with `country_code`, `country_name`, `currency`, `primary_languages` fields plus 7 SCOPE sections. See schema in `01-scope-builder.md`. Save as `scope-<profession_id>-<YYYY-MM-DD>.json`.

## Phase 2 - Pain Generator

Use the prompt in `prompts/profession-scan/02-pain-generator.md` as a system prompt. Feed the Phase-1 SCOPE JSON to it. The prompt is a country-aware version of the validated US generator.

**Constraints (non-negotiable):**
- 8-15 pains per profession
- At least 3 distinct `skill_type` values (calculator + checklist + one other minimum)
- Priority by value: calculator > checklist > template > reference > advisor
- Each `skill_spec` must match the per-type schema (see prompt + `data/professions/us/_FORMAT.md`)
- No duplicate pains
- `money_risk_usd` mandatory (for cross-country comparison); `money_risk_local` + `currency_local` optional but recommended
- Pain content language = country's working language (English for US/UK/AU, German for DE, French for FR, Russian for KZ/RU, etc.) - unless user explicitly requests English output
- `regulatory_refs` use country-native citation format (USC/CFR for US, BGB for DE, НК РК for KZ, etc.) - never paste US citations into a non-US output

Output: JSON array of pain objects (no wrapper).

## Phase 3 - Bundle Formatter

Use `prompts/profession-scan/03-bundle-formatter.md` for validation + cleanup. Most runs skip this phase entirely if Phase 2 output is already clean.

Save as `pains-<profession_id>-<YYYY-MM-DD>.json` in the current working directory.

If you want to contribute back to the dataset, copy the file into `data/professions/<cc>/pains/<profession_id>.json` (no date suffix) and open a PR. New countries: create `data/professions/<cc>/profiles/` + `pains/` + `README.md` + `_FORMAT.md` (port from US).

## Good-run example (US)

Input: `profession-scan lawyers in US`

Expected output shape:
- 7 SCOPE sections filled, each with 3-6 sources, all `.gov` / `.edu`
- 12-15 pains in English, mix: 4-5 calculators (filing fees, billable hours, IOLTA reconciliation, malpractice premium), 3-4 checklists (case intake, conflict check, deposition prep, trust account audit), 2-3 templates (engagement letter, demand letter), 1-2 references (CLE deadlines by state), 1-2 advisors (UPL questions, bar complaint response)

Reference: `data/professions/us/pains/us-lawyers.json` shipped with the repo.

## Good-run example (non-US)

Input: `profession-scan Steuerberater in DE`

Expected output shape:
- 7 SCOPE sections sourced from `.bund.de` / `bstbk.de` / `gesetze-im-internet.de` plus German-language professional forums
- 12-15 pains in German, mix similar to US but using EStG / AO / StBerG references
- `money_risk_usd` populated via EUR conversion; `money_risk_local` in EUR

## Sparse-run example (expected failure mode)

Input: `profession-scan dental hygienists in Tajikistan`

Expected:
- SCOPE may be thin in `community`, `terminology`, `tools` (small profession, small country, low online footprint)
- Pains likely 6-9 instead of 12-15
- Honest `coverage_gap` notes in multiple SCOPE sections
- DO NOT pad to hit 8-pain minimum with vague "consult a specialist" advisor entries

## Re-use the existing dataset

If the user asks for a profession+country that's already in `data/professions/<cc>/pains/`, prefer reading and presenting the existing bundle over re-running the pipeline. Re-run only when:
- The user explicitly asks for fresh data
- The shipped pain file is older than 6 months
- A skill_spec is missing fields or has obvious errors

The 25 pre-shipped US bundles cover the top US occupations (lawyers, accountants, nurses, engineers, etc.). All other countries currently empty.
