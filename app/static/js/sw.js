const CACHE = 'mr-v1';
const SHELL = [
  '/', '/offline',
  '/static/css/main.css',
  '/static/js/app.js',
  '/static/js/chartjs.min.js',
  '/static/manifest.webmanifest',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  '/static/icons/icon-maskable-512.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

function networkFirst(req, fallback) {
  return fetch(req)
    .then((res) => {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(req, copy));
      return res;
    })
    .catch(() => caches.match(req).then((hit) => hit || (fallback ? caches.match(fallback) : undefined)));
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  const url = new URL(req.url);

  if (url.origin !== self.location.origin) return; // cross-origin: pass through

  if (req.mode === 'navigate') {
    e.respondWith(networkFirst(req, '/offline'));
    return;
  }

  if (url.pathname.startsWith('/api/')) {
    e.respondWith(networkFirst(req));
    return;
  }

  if (url.pathname.startsWith('/static/')) {
    e.respondWith(
      caches.match(req).then((hit) => {
        const net = fetch(req).then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
          return res;
        }).catch(() => hit);
        return hit || net;
      })
    );
  }
});
