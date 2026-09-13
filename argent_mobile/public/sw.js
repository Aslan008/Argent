// Service Worker для автономной работы Argent Mobile без ПК и интернета
const CACHE_NAME = 'argent-mobile-v1';

const STATIC_PRECACHE = [
  '/',
  '/index.html',
  '/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_PRECACHE);
    }).then(() => {
      return self.skipWaiting();
    })
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    }).then(() => {
      return self.clients.claim();
    })
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Игнорируем не-GET запросы и обращения к внешним API нейросетей
  if (event.request.method !== 'GET') return;
  if (
    url.pathname.includes('/chat/completions') ||
    url.hostname.includes('api.deepseek.com') ||
    url.hostname.includes('openrouter.ai') ||
    url.hostname.includes('api.openai.com') ||
    url.hostname.includes('api.groq.com')
  ) {
    return;
  }

  // Для локальных файлов и библиотек: Cache First с динамическим кэшированием
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }

      return fetch(event.request).then((networkResponse) => {
        if (!networkResponse || networkResponse.status !== 200) {
          return networkResponse;
        }

        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });

        return networkResponse;
      }).catch(() => {
        // Если нет сети и файл не найден, возвращаем главную страницу
        if (event.request.headers.get('accept')?.includes('text/html')) {
          return caches.match('/index.html') as Promise<Response>;
        }
        return new Response('Offline', { status: 503, statusText: 'Offline' });
      });
    })
  );
});
