const CACHE_NAME = 'wild-hunt-v1';

const APP_FILES = [
  './',
  './index.html',
  './game.js',
  './style.css',
  './manifest.json',
  './icon-192.png',
  './icon-512.png'
];

// Встановлення та кешування основних файлів
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(APP_FILES);
    })
  );

  self.skipWaiting();
});

// Видалення старого кешу
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(key => key !== CACHE_NAME)
          .map(key => caches.delete(key))
      )
    )
  );

  self.clients.claim();
});

// Завантаження сторінок і файлів
self.addEventListener('fetch', event => {
  const request = event.request;

  if (request.method !== 'GET') return;

  const url = new URL(request.url);

  // Не перехоплюємо сторонні сайти та CDN
  if (url.origin !== self.location.origin) return;

  // Для сторінок спочатку пробуємо інтернет,
  // а при його відсутності відкриваємо кешовану гру
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then(response => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then(cache =>
            cache.put('./index.html', copy)
          );
          return response;
        })
        .catch(() => caches.match('./index.html'))
    );
    return;
  }

  // Для інших файлів використовуємо кеш
  // і оновлюємо його після мережевого запиту
  event.respondWith(
    caches.match(request).then(cached => {
      const network = fetch(request).then(response => {
        if (response && response.ok) {
          const copy = response.clone();
          caches.open(CACHE_NAME).then(cache =>
            cache.put(request, copy)
          );
        }
        return response;
      });

      return cached || network;
    })
  );
});
