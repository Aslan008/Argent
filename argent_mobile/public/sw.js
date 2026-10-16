// Service Worker Argent Mobile (PWA): приложение открывается и без сети.
//
// Прошлая версия не работала вовсе: в обычном .js стояло TypeScript-выражение `as Promise<Response>`,
// воркер не разбирался и не регистрировался. И её стратегия была опасной: «сначала кэш» для index.html
// навсегда закрепляла старую версию после обновления, а любые чужие GET (веса моделей WebLLM с
// Hugging Face, API в домашней сети) копировались в кэш воркера.
//
// Теперь:
// - только запросы к своему адресу; всё чужое (API, модели, шрифты) идёт мимо воркера;
// - страницы — сначала сеть, кэш только без сети (обновление видно сразу);
// - /assets/* — сначала кэш: у этих файлов хэш в имени, их содержимое не меняется;
// - прочие свои файлы — из кэша с обновлением в фоне.

const CACHE_NAME = 'argent-mobile-v2';
const APP_SHELL = ['/', '/index.html', '/manifest.json', '/icon-192.png', '/icon-512.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

function isPage(request) {
  return request.mode === 'navigate' || (request.headers.get('accept') || '').includes('text/html');
}

async function networkFirstPage(request) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const response = await fetch(request);
    if (response && response.ok) cache.put('/index.html', response.clone());
    return response;
  } catch (err) {
    const cached = (await cache.match(request)) || (await cache.match('/index.html'));
    if (cached) return cached;
    return new Response('Нет сети, и приложение ещё не сохранено для работы без неё.', {
      status: 503,
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    });
  }
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response && response.ok) cache.put(request, response.clone());
  return response;
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  const refresh = fetch(request)
    .then((response) => {
      if (response && response.ok) cache.put(request, response.clone());
      return response;
    })
    .catch(() => cached || new Response('Offline', { status: 503, statusText: 'Offline' }));
  return cached || refresh;
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (isPage(request)) {
    event.respondWith(networkFirstPage(request));
  } else if (url.pathname.startsWith('/assets/')) {
    event.respondWith(cacheFirst(request));
  } else {
    event.respondWith(staleWhileRevalidate(request));
  }
});
