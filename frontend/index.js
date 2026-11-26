// html router relative path detactive
function resolvePage(input) {
    const name = input
        .split('/').pop()
        .replace(/\.html$/, '') || 'home';

    const base = `/pages/${name}`;
    return {
        name,
        htmlPath: `${base}/${name}.html`,
        cssPath:  `${base}/${name}.css`,
        jsPath:   `${base}/${name}.js`,
    };
}

async function loadPage(input) {
    const { htmlPath, cssPath, jsPath } = resolvePage(input);

    updateActive(input);

    const html = await fetch(htmlPath).then(r => r.text());
    document.getElementById('view').innerHTML = html;

    document.getElementById('page-css').href = cssPath;

    const mod = await import(jsPath);
    mod.init();
}

document.addEventListener('click', e => {
    const a = e.target.closest('a[data-link]');
    if (!a) return;
    e.preventDefault();
    const path = a.getAttribute('href');
    history.pushState({}, '', path);
    loadPage(path);
});

// setting theme
function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    themeBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
    try { localStorage.setItem('theme', theme); } catch {}
}

function toggleTheme() {
    const cur = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    setTheme(cur);
}

const themeBtn = document.getElementById('theme-toggle');
const saved = localStorage.getItem('theme');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;


document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

// navi activate style
function updateActive(path) {
    const current = resolvePage(path).name;

    document.querySelectorAll('.navi a').forEach(a => {
        const isActive = resolvePage(a.getAttribute('href')).name === current;
        a.classList.toggle('active', isActive);
        a.setAttribute('aria-current', isActive ? 'page' : 'false');
    });
}

setTheme(saved || (prefersDark ? 'dark' : 'light')); 
window.addEventListener('popstate', () => loadPage(location.pathname));
loadPage(location.pathname);