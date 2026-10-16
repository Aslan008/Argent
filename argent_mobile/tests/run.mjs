// Запуск тестов: каждый tests/test_*.{mjs,ts} собирается esbuild в один файл (импорты без расширений,
// типы TypeScript) и выполняется в Node. Код выхода не 0, если упал хоть один.
import { build } from 'esbuild';
import { readdirSync, mkdirSync, rmSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, '.build');
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

const files = readdirSync(here).filter((f) => /^test_.*\.(mjs|ts)$/.test(f)).sort();
let failed = 0;
for (const file of files) {
  const target = join(out, file.replace(/\.(ts|mjs)$/, '.bundle.mjs'));
  try {
    await build({
      entryPoints: [join(here, file)],
      bundle: true,
      platform: 'node',
      format: 'esm',
      outfile: target,
      logLevel: 'silent',
      external: ['@mlc-ai/web-llm'],
    });
    await import(pathToFileURL(target).href);
    console.log(`✓ ${file}`);
  } catch (err) {
    failed++;
    console.error(`✗ ${file}\n${err?.stack || err}`);
  }
}
rmSync(out, { recursive: true, force: true });
console.log(failed ? `\n${failed} из ${files.length} файлов тестов упали` : `\nВсе ${files.length} файлов тестов прошли`);
process.exit(failed ? 1 : 0);
