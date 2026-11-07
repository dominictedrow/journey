function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('overlay');
  const opening = !sidebar.classList.contains('open');
  sidebar.classList.toggle('open');
  overlay.classList.toggle('visible');
  if (opening) animateSidebar(sidebar);
}

function animateSidebar(sidebar) {
  var items = Array.from(sidebar.querySelectorAll('.nav-item, .submenu-item, .sidebar-logo, .sidebar-close'));
  items.forEach(function (el) {
    el.classList.remove('anim-rise');
    el.style.animationDelay = '';
    // Force reflow so removing+re-adding the class restarts the animation
    void el.offsetWidth;
  });
  var step = Math.min(0.04, 0.3 / Math.max(items.length - 1, 1));
  items.forEach(function (el, i) {
    el.classList.add('anim-rise');
    el.style.animationDelay = (i * step).toFixed(3) + 's';
  });
}

document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.nav-group').forEach(function (group) {
    const parent = group.querySelector('.nav-parent');
    if (!parent) return;
    parent.addEventListener('click', function () {
      group.classList.toggle('open');
    });
  });
});

function showModal(id) {
  var el = document.getElementById(id);
  if (el) el.classList.add('visible');
}

function hideModal(id) {
  var el = document.getElementById(id);
  if (el) el.classList.remove('visible');
}

// Page-in animation — stagger rise-in for every visible element
(function () {
  var SELECTORS = [
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'p', 'span', 'a', 'li', 'label',
    'canvas', 'img', 'svg',
    'table', 'tr', 'hr',
    '.card', '.page-title',
    '.tier-node', '.tier-connector',
    '.btn', '.badge',
    'textarea', 'input', 'select',
    'pre', 'code'
  ].join(',');

  function animatePageIn() {
    if (sessionStorage.getItem('skip-anim')) {
      sessionStorage.removeItem('skip-anim');
      return;
    }
    var content = document.querySelector('.content');
    if (!content) return;

    var candidates = Array.from(content.querySelectorAll(SELECTORS));

    // Deduplicate: skip element if any ancestor is already in the candidate set
    var set = new Set(candidates);
    var elements = candidates.filter(function (el) {
      var p = el.parentElement;
      while (p) {
        if (set.has(p)) return false;
        p = p.parentElement;
      }
      return true;
    });

    // Sort top-to-bottom; left-to-right within the same row (within 12 px)
    elements.sort(function (a, b) {
      var ra = a.getBoundingClientRect();
      var rb = b.getBoundingClientRect();
      var dy = ra.top - rb.top;
      if (Math.abs(dy) < 12) return ra.left - rb.left;
      return dy;
    });

    // Spread delays across a max of 0.8 s regardless of element count
    var n = elements.length;
    var spread = Math.min(0.04 * (n - 1), 0.8);
    var step = n > 1 ? spread / (n - 1) : 0;

    elements.forEach(function (el, i) {
      el.classList.add('anim-rise');
      el.style.animationDelay = (i * step).toFixed(3) + 's';
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', animatePageIn);
  } else {
    animatePageIn();
  }
}());
