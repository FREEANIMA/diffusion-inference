export const API_BASE = '';

/* common.js — txt2img / inpaint 공용
 * 왼쪽 트리 + 가운데 입력(common.html) + sigma/cfg/eta 곡선 + 프롬프트/결과 번호 */

// ===================== 곡선 (sigma / cfg / eta) =====================
// x = linspace(0, 1, steps + 1), 점 개수 = steps. 유저는 함수식으로만 입력 (점 직접 편집 없음)
const MATH = {
    sin: Math.sin, cos: Math.cos, tan: Math.tan, acos: Math.acos, exp: Math.exp, log: Math.log,
    sqrt: Math.sqrt, abs: Math.abs, pow: Math.pow, min: Math.min, max: Math.max,
    floor: Math.floor, pi: Math.PI, e: Math.E,
};
const MATH_NAMES = Object.keys(MATH);
const ALLOWED = new Set(['t', ...MATH_NAMES]);

// isPdf : 입력식을 밀도함수로 보고 출력은 CDF 기반 (sigma)
// eta 기본값
const curves = {
    sigma: {
        expr: 'sin(pi*(0.7 + t*3)/5)', isPdf: true, ymin: 0, ymax: 1, values: [], pdf: [],
        hint: 'Enter a density (pdf). Output is its CDF: σ = 1 − CDF. Var: t',
    },
    cfg: {
        expr: 'max(0, sin(pi*(0.5 + 2*t)/3))', ymin: 0, ymax: 1, values: [],
        hint: 'Plot range is 0–1. Final CFG = cfg × plot. Var: t',
    },
    eta: {
        expr: 'max(0, sin(pi*(0.5 + 2*t)/3)) * 0.1', ymin: 0, ymax: 0.5, values: [],
        hint: 'Used as-is per step (0–0.5). Var: t',
    },
};
let active = 'sigma';
const ui = { mini: {} };

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

function compile(expr) {
    if (!/^[\w\s+\-*/%^().,]*$/.test(expr)) throw new Error('허용되지 않는 문자가 있습니다');
    for (const id of expr.match(/[A-Za-z_]\w*/g) || []) {
        if (!ALLOWED.has(id)) throw new Error(`알 수 없는 이름: ${id}`);
    }
    const fn = new Function('t', ...MATH_NAMES,
        `"use strict"; return (${expr.replace(/\^/g, '**')});`);
    const consts = MATH_NAMES.map(k => MATH[k]);
    return t => fn(t, ...consts);
}

function currentSteps() {
    return Math.max(1, Math.round(Number(document.getElementById('t2i-steps').value) || 1));
}

// 식을 linspace(0, 1, steps) 위에서 계산
function evalRaw(c, steps) {
    const f = compile(c.expr);
    return Array.from({ length: steps }, (_, i) => {
        const t = steps > 1 ? i / (steps - 1) : 0;
        const v = Number(f(t));
        if (!Number.isFinite(v)) throw new Error(`step ${i} 에서 값이 유효하지 않습니다`);
        return v;
    });
}

// 역CDF: F는 0→1로 증가하는 배열, F(s)=q 인 s(0~1)를 선형보간으로 찾음
function invertCdf(F, q) {
    let lo = 0, hi = F.length - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (F[mid] < q) lo = mid; else hi = mid;
    }
    const d = F[hi] - F[lo];
    const f = d > 0 ? (q - F[lo]) / d : 0;
    return (lo + f) / (F.length - 1);
}

// 밀도함수(pdf) → CDF F → f(x) = F⁻¹(x). 서버 sigma_cdf(u)와 같은 결과 (0 → 1). 전송할 때 뒤집음
// 서버 sigma_cdf(u)[::-1] 와 같은 결과 (첫 값 1, 마지막 값 0)
function pdfToSigma(pdf, n) {
    const m = pdf.length;
    if (m < 2 || n < 2) return [1];
    const dx = 1 / (m - 1);
    const F = [0];
    for (let k = 1; k < m; k++) F.push(F[k - 1] + 0.5 * (pdf[k] + pdf[k - 1]) * dx);
    const total = F[m - 1];
    if (!(total > 0)) throw new Error('밀도함수의 적분이 0입니다 (0보다 큰 구간이 필요합니다)');
    for (let k = 0; k < m; k++) F[k] /= total;
    return Array.from({ length: n }, (_, i) => invertCdf(F, i / (n - 1)));
}

function setMsg(text, isError = false) {
    ui.msg.textContent = text;
    ui.msg.classList.toggle('error', isError);
}

function applyExpr(name) {
    const c = curves[name];
    try {
        const raw = evalRaw(c, currentSteps() + 1);
        if (c.isPdf) {
            const fine = evalRaw(c, 1001).map(v => Math.max(0, v));    // σ축을 촘촘하게 계산한 밀도
            c.values = pdfToSigma(fine, raw.length);
            c.dens = fine;                                             // 차트에서 볼록하게 그릴 밀도
        } else {
            c.values = raw.map(v => clamp(v, c.ymin, c.ymax));
        }
        setMsg('');
    } catch (err) {
        setMsg(err.message, true);      // 실패하면 기존 곡선 유지
    }
    redraw(name);
}

function viewOf(c) {
    if (!c.isPdf) return c.values;
    const m = Math.max(...c.pdf) || 1;
    return c.pdf.map(v => v / m).reverse();      // step 0 = σ 1 쪽부터
}

function applyAll() {
    for (const name of Object.keys(curves)) applyExpr(name);
}

const yRange = c => [c.ymin, c.ymax];

// 밀도함수를 0~1로 맞춰 점선으로 같이 보여줌
function ghostOf(c) {
    if (!c.isPdf || !c.pdf.length) return null;
    const m = Math.max(...c.pdf) || 1;
    return c.pdf.map(v => v / m);
}

// sigma 전용: σ축(0.001~0.999) 한 줄 위에 스텝별 σ를 점으로 찍고,
// 확률적으로 스텝이 쏠리는 구간(입력한 밀도)은 그 위로 볼록하게 올림
const S_LO = 0.001, S_HI = 0.999;

function drawDensity(cv, dens, values, o) {
    const dpr = window.devicePixelRatio || 1;
    const r = cv.getBoundingClientRect();
    if (!r.width || !r.height || !values.length || !dens || !dens.length) return;   // 닫힌 dialog 등
    cv.width = r.width * dpr;
    cv.height = r.height * dpr;
    const g = cv.getContext('2d');
    g.scale(dpr, dpr);

    const W = r.width, H = r.height;
    const css = getComputedStyle(cv);
    const muted = css.getPropertyValue('--muted').trim() || '#888';
    const accent = css.getPropertyValue('--accent').trim() || '#4a7';
    const fs = o.compact ? 9 : 11;

    // 축 라벨 / 범례 자리를 위해 여백 확대
    const p = o.compact
        ? { l: 24, r: 12, t: 16, b: 32 }
        : { l: 46, r: 30, t: 28, b: 48 };
    const X = v => p.l + (W - p.l - p.r) * (clamp(v, S_LO, S_HI) - S_LO) / (S_HI - S_LO);
    const y0 = H - p.b;                                   // σ축(기준선) 위치
    const hmax = H - p.t - p.b;                           // 볼록 부분의 최대 높이
    const m = Math.max(...dens) || 1;
    const D = v => dens[Math.round(clamp(v, 0, 1) * (dens.length - 1))] / m;   // σ에서의 밀도 (0~1)

    // 밀도 곡선
    const N = o.compact ? 60 : 200;
    g.beginPath();
    for (let k = 0; k <= N; k++) {
        const v = S_LO + (S_HI - S_LO) * k / N;
        const x = X(v), y = y0 - hmax * D(v);
        if (k) g.lineTo(x, y); else g.moveTo(x, y);
    }
    g.strokeStyle = accent;
    g.lineWidth = 2;
    g.stroke();
    g.lineTo(X(S_HI), y0);
    g.lineTo(X(S_LO), y0);
    g.closePath();
    g.globalAlpha = 0.18;
    g.fillStyle = accent;
    g.fill();
    g.globalAlpha = 1;

    // x축(σ) 한 줄 + 눈금
    g.strokeStyle = muted;
    g.fillStyle = muted;
    g.lineWidth = 1.5;
    g.font = `${fs}px sans-serif`;
    g.textAlign = 'center';
    g.beginPath(); g.moveTo(X(S_LO), y0); g.lineTo(X(S_HI), y0); g.stroke();
    const ticks = o.compact ? [S_LO, 0.5, S_HI] : [S_LO, 0.25, 0.5, 0.75, S_HI];
    const tl = o.compact ? 3 : 5;
    g.lineWidth = 1;
    for (const v of ticks) {
        g.beginPath(); g.moveTo(X(v), y0 - tl); g.lineTo(X(v), y0 + tl); g.stroke();
        g.fillText(String(v), X(v), y0 + tl + (o.compact ? 11 : 15));
    }
    // x축 라벨
    g.fillText('sigma', (X(S_LO) + X(S_HI)) / 2, H - (o.compact ? 3 : 6));

    // y축(확률) 선 + 눈금 (피크 기준 0~1로 정규화된 값)
    const xl = X(S_LO) - (o.compact ? 4 : 8);
    g.beginPath(); g.moveTo(xl, y0); g.lineTo(xl, y0 - hmax); g.stroke();
    g.textAlign = 'right';
    g.fillText('0', xl - 3, y0 + 3);
    g.fillText('1', xl - 3, y0 - hmax + 3);
    // y축 라벨 (세로)
    g.save();
    g.translate(o.compact ? 8 : 12, y0 - hmax / 2);
    g.rotate(-Math.PI / 2);
    g.textAlign = 'center';
    g.fillText('probability', 0, 0);
    g.restore();

    // 스텝별 σ 점
    g.fillStyle = accent;
    const rad = o.compact ? 1.8 : 3.5;
    for (const v of values) {
        g.beginPath(); g.arc(X(v), y0, rad, 0, Math.PI * 2); g.fill();
    }

    // 범례: ● = steps 수
    const label = `steps: ${values.length - 1}`;
    g.font = `${fs}px sans-serif`;
    g.textAlign = 'right';
    g.fillStyle = muted;
    const lx = W - p.r;
    const ly = o.compact ? 9 : 14;
    g.fillText(label, lx, ly + 3);
    const tw = g.measureText(label).width;
    g.fillStyle = accent;
    g.beginPath(); g.arc(lx - tw - 8, ly, rad + 1, 0, Math.PI * 2); g.fill();
}

// o: { ymin, ymax, compact, ghost }
function drawCurve(cv, values, o) {
    const dpr = window.devicePixelRatio || 1;
    const r = cv.getBoundingClientRect();
    if (!r.width || !r.height || !values.length) return;   // 닫힌 dialog 등
    cv.width = r.width * dpr;
    cv.height = r.height * dpr;
    const g = cv.getContext('2d');
    g.scale(dpr, dpr);

    const p = o.compact ? { l: 26, r: 8, t: 8, b: 22 } : { l: 38, r: 14, t: 12, b: 32 };
    const W = r.width, H = r.height, n = values.length;
    const css = getComputedStyle(cv);
    const border = css.getPropertyValue('--border').trim() || '#888';
    const accent = css.getPropertyValue('--accent').trim() || '#4a7';
    const muted = css.getPropertyValue('--muted').trim() || '#888';

    const X = i => p.l + (W - p.l - p.r) * (n > 1 ? i / (n - 1) : 0);
    const Y = v => H - p.b - (H - p.t - p.b) * (v - o.ymin) / (o.ymax - o.ymin);

    // 격자 + y축 라벨
    const ticks = o.compact ? 2 : 4;
    g.strokeStyle = border;
    g.fillStyle = muted;
    g.font = o.compact ? '9px sans-serif' : '10px sans-serif';
    g.lineWidth = 1;
    g.textAlign = 'left';
    for (let k = 0; k <= ticks; k++) {
        const v = o.ymin + (o.ymax - o.ymin) * k / ticks;
        g.globalAlpha = 0.5;
        g.beginPath(); g.moveTo(p.l, Y(v)); g.lineTo(W - p.r, Y(v)); g.stroke();
        g.globalAlpha = 1;
        g.fillText(v.toFixed(o.compact ? 1 : 2), 2, Y(v) + 3);
    }

    // x축 라벨: 0 / steps / 마지막 스텝
    const ly = H - p.b + (o.compact ? 13 : 16);
    g.textAlign = 'left';   g.fillText('0', p.l, ly);
    g.textAlign = 'center'; g.fillText('steps', (p.l + W - p.r) / 2, ly);
    g.textAlign = 'right';  g.fillText(String(n - 1), W - p.r, ly);
    g.textAlign = 'left';

    // 보조선 (sigma: 입력한 밀도함수)
    if (o.ghost) {
        g.strokeStyle = muted;
        g.lineWidth = 1.5;
        g.setLineDash([5, 4]);
        g.beginPath();
        o.ghost.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
        g.stroke();
        g.setLineDash([]);
    }

    // 곡선
    g.strokeStyle = accent;
    g.lineWidth = 2;
    g.beginPath();
    values.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
    g.stroke();

    // 스텝별 점 (큰 차트에서만, 표시 전용)
    if (!o.compact) {
        g.fillStyle = accent;
        values.forEach((v, i) => {
            g.beginPath(); g.arc(X(i), Y(v), 3, 0, Math.PI * 2); g.fill();
        });
    }
}

function drawMini(name) {
    const c = curves[name];
    if (c.isPdf) return drawDensity(ui.mini[name], c.dens, c.values, { compact: true });   // drawMini
    const [ymin, ymax] = yRange(c);
    drawCurve(ui.mini[name], c.values, { ymin, ymax, compact: true });
}

function drawEditor() {
    const c = curves[active];
    if (c.isPdf) return drawDensity(ui.canvas, c.dens, c.values, {});     
    const [ymin, ymax] = yRange(c);
    drawCurve(ui.canvas, c.values, { ymin, ymax });
}

function redraw(name) {
    drawMini(name);
    if (name === active) drawEditor();
}

function selectTab(name) {
    active = name;
    ui.tabs.forEach(b => b.classList.toggle('active', b.dataset.name === name));
    ui.input.value = curves[name].expr;
    ui.hint.textContent = curves[name].hint;
    setMsg('');
    drawEditor();
}

function initCurves() {
    const root = document.querySelector('.page-txt2img');

    // 미니 차트 3개 (보기 전용)
    for (const name of Object.keys(curves)) {
        const cv = root.querySelector(`.t2i-chart[data-name="${name}"] canvas`);
        ui.mini[name] = cv;
        new ResizeObserver(() => drawMini(name)).observe(cv);
    }

    // 오버레이
    ui.dialog = document.getElementById('t2i-rev');
    ui.input = document.getElementById('t2i-rev-expr');
    ui.canvas = document.getElementById('t2i-rev-canvas');
    ui.msg = document.getElementById('t2i-rev-msg');
    ui.hint = document.getElementById('t2i-rev-hint');
    ui.tabs = [...ui.dialog.querySelectorAll('.t2i-rev-tab')];

    ui.tabs.forEach(b => b.addEventListener('click', () => selectTab(b.dataset.name)));

    const apply = () => {
        curves[active].expr = ui.input.value;
        applyExpr(active);
    };
    document.getElementById('t2i-rev-apply').addEventListener('click', apply);
    ui.input.addEventListener('keydown', e => { if (e.key === 'Enter') apply(); });

    document.getElementById('t2i-revision').addEventListener('click', () => {
        ui.dialog.showModal();
        selectTab(active);
    });
    document.getElementById('t2i-rev-close').addEventListener('click', () => ui.dialog.close());
    ui.dialog.addEventListener('click', e => {          // 바깥(backdrop) 클릭 시 닫기
        if (e.target === ui.dialog) ui.dialog.close();
    });

    new ResizeObserver(() => drawEditor()).observe(ui.canvas);

    // 초기값: 식으로 계산
    applyAll();

    // Steps 값이 바뀌면 식으로 다시 계산
    document.getElementById('t2i-steps').addEventListener('change', applyAll);
}

function getCurveParams(cfg) {
    return {
        sigmas: curves.sigma.values.slice().reverse(),   // step 0 = σ 1 → 마지막 = σ 0
        cfgs: curves.cfg.values.map(v => v * cfg),       // step 0 → step N (차트 순서 그대로)
        etas: curves.eta.values.slice(),                 // step 0 → step N (차트 순서 그대로)
    };
}
// ===================== 프롬프트 / 결과 번호 =====================
const images = [];      // images[i] = i번째 prompt의 결과 이미지 URL (없으면 undefined)
let current = 0;        // 현재 선택된 번호 (0부터)

function promptCount() {
    return document.querySelectorAll('#t2i-prompts .t2i-prompt').length;
}

function addPrompt() {
    const item = document.createElement('div');
    item.className = 't2i-prompt-item';
    item.innerHTML = `
        <textarea class="t2i-prompt" rows="4"></textarea>
        <button type="button" class="t2i-remove" aria-label="칸 삭제">×</button>
    `;
    document.getElementById('t2i-prompts').appendChild(item);
    syncViewer();
    item.querySelector('textarea').focus();
    savePrompts();
}

// prompt 개수에 맞춰 placeholder, 번호 버튼, 이미지 표시를 한 번에 갱신
function syncViewer() {
    const n = promptCount();
    if (current >= n) current = n - 1;
    if (current < 0) current = 0;

    document.querySelectorAll('#t2i-prompts .t2i-prompt').forEach((ta, i) => {
        ta.placeholder = `prompt ${i + 1}`;
    });

    const nums = document.getElementById('t2i-nums');
    nums.innerHTML = '';
    for (let i = 0; i < n; i++) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 't2i-num' + (i === current ? ' active' : '');
        b.dataset.index = i;
        b.textContent = i + 1;
        nums.appendChild(b);
    }

    showImage(current);
}

function showImage(i) {
    const box = document.getElementById('t2i-image');
    const src = images[i];
    box.innerHTML = src
        ? `<img src="${src}" alt="prompt ${i + 1} result">`
        : '<span class="t2i-placeholder">No image</span>';
}

// ===================== 공개 API =====================
let htmlCache = null;

// 페이지 init()에서 한 번 호출.
// 페이지 HTML에 #t2i-nums, #t2i-image, #t2i-generate 가 있어야 함.
// size: false 면 Width/Height 칸을 뺌 (inpaint는 crop이 크기를 정함)
export async function initCommon({ size = true, inpaint = false, onGenerate } = {}) {
    pageKey = inpaint ? 'inpaint' : 'sampling';
    images.length = 0;
    current = 0;

    if (!htmlCache) {
        const res = await fetch(new URL('./common.html', import.meta.url), { cache: 'no-store' });
        if (!res.ok) throw new Error(`common.html 로드 실패 (${res.status})`);
        htmlCache = await res.text();
        console.log('common.html loaded:', res.url, 'has tag-tree =', htmlCache.includes('t2i-tag-tree'));
    }
    // common.css를 매번 새로 받음: 캐시된 옛 CSS가 적용되는 것을 막고, common.html과 같은 폴더의 파일을 확실히 씀
    if (!document.getElementById('t2i-common-css')) {
        const link = document.createElement('link');
        link.id = 't2i-common-css';
        link.rel = 'stylesheet';
        link.href = new URL(`./common.css?v=${Date.now()}`, import.meta.url).href;
        await new Promise(res => { link.onload = link.onerror = res; document.head.appendChild(link); });
    }
    const root = document.querySelector('.page-txt2img');
    root.insertAdjacentHTML('afterbegin', htmlCache);      // 왼쪽/가운데를 루트의 직계 자식으로
    if (!size) root.querySelectorAll('[data-size]').forEach(el => el.remove());
    if (!inpaint) root.querySelectorAll('[data-inpaint]').forEach(el => el.remove());
    initLayout(root);          // Tag Browser 접기 / 폭·높이 조절

    restorePrompts();   

    document.getElementById('t2i-add-prompt').addEventListener('click', addPrompt);
    document.getElementById('t2i-generate').addEventListener('click', onGenerate);
    document.getElementById('t2i-prompts').addEventListener('input', savePrompts);

    // 삭제 버튼은 동적으로 생기므로 부모에서 이벤트 위임
    document.getElementById('t2i-prompts').addEventListener('click', e => {
        const btn = e.target.closest('.t2i-remove');
        if (!btn) return;
        const item = btn.closest('.t2i-prompt-item');
        const items = [...document.querySelectorAll('#t2i-prompts .t2i-prompt-item')];
        images.splice(items.indexOf(item), 1);   // 해당 번호의 이미지도 같이 제거
        item.remove();
        savePrompts();
        syncViewer();
    });

    // 번호 클릭 → 이미지 전환 (이벤트 위임)
    document.getElementById('t2i-nums').addEventListener('click', e => {
        const b = e.target.closest('.t2i-num');
        if (!b) return;
        current = Number(b.dataset.index);
        syncViewer();
    });

    initTagBrowser();
    initAutocomplete();

    initCurves();
    syncViewer();
}

// 백엔드 응답을 넣을 때: setResult(i, url)
export function setResult(i, url) {
    images[i] = url;
    if (i === current) showImage(current);
}

// 공통 파라미터 수집 (Width/Height 칸이 있으면 포함)
export function collectCommon() {
    const val = id => document.getElementById(id).value;
    const cfg = Number(val('t2i-cfg'));
    const params = {
        prompts: [...document.querySelectorAll('.t2i-prompt')].map(t => t.value.trim()),
        steps: Number(val('t2i-steps')),
        cfg,
        seed: Number(val('t2i-seed')),
        calculateType: val('t2i-calculate-type'),   // ← 추가
        promptStatic: document.getElementById('t2i-prompt-static').checked,
        frameSave: document.getElementById('t2i-frame-save').checked,
        nsfw: document.getElementById('t2i-nsfw').checked,
        ...getCurveParams(cfg),
        tags: [...checkedTags],
    };
    if (document.getElementById('t2i-wiidth')) {
        params.width = Number(val('t2i-width'));
        params.height = Number(val('t2i-height'));
    }
    const st = document.getElementById('t2i-start-t');
    if (st) {
        const v = Number(st.value);
        params.startT = Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 1;   // 0~1, 비었거나 잘못된 값이면 1
    }
    return params;
}

// ===================== 서버 전송용 =====================
// collectCommon() 결과 + prompt 하나 → 백엔드 SamplingRequest 모양
//   sigmas/cfgs/etas → sigma_curve/cfg_curve/eta_curve  (cfgs는 이미 cfg 입력값이 곱해진 최종값)
//   prompts(여러 개) → prompt(하나): prompt마다 요청을 따로 보냄
//   size: { width, height } 를 주면 그 값을 씀 (inpainting은 crop 크기)
//   cfg, styles 는 SamplingRequest에 없어서 보내지 않음
export function toSamplingRequest(p, prompts = p.prompts, size = {}) {
    const steps = Math.max(1, Math.round(p.steps) || 1);
    for (const [name, arr] of [['Sigma', p.sigmas], ['CFG', p.cfgs], ['Eta', p.etas]]) {
        if (arr.length !== steps + 1) {
            throw new Error(`${name} 곡선 길이(${arr.length})가 steps+1(${steps + 1})과 다릅니다. Revision에서 식을 확인하세요`);
        }
    }
    return {
        prompts: Array.isArray(prompts) ? prompts : [prompts],
        sigma_curve: p.sigmas,
        cfg_curve: p.cfgs,
        eta_curve: p.etas,
        calculate_type: p.calculateType,            // ← 추가
        seed: p.seed,
        steps,
        width: size.width ?? p.width,
        height: size.height ?? p.height,
        frame_save: p.frameSave,
        prompt_static: p.promptStatic,
        nsfw: p.nsfw
    };
}

// JSON POST. 응답은 아직 처리하지 않음 (상태 코드만 콘솔에 찍고 Response를 그대로 돌려줌)
export async function postJSON(url, body) {
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.json();   // 204일 땐 없었던 부분
}

// ===================== 프롬프트 보존 (페이지 이동 시) =====================
const savedPrompts = {};      // { sampling: [...], inpaint: [...] }
let pageKey = 'default';

function savePrompts() {
    const list = [...document.querySelectorAll('#t2i-prompts .t2i-prompt')].map(t => t.value);
    if (list.length) savedPrompts[pageKey] = list;
}

function restorePrompts() {
    const list = savedPrompts[pageKey] ?? [''];
    const box = document.getElementById('t2i-prompts');
    box.innerHTML = '';
    list.forEach((text, i) => {
        const item = document.createElement('div');
        item.className = 't2i-prompt-item';
        item.innerHTML = `<textarea class="t2i-prompt" rows="4"></textarea>` +
            (i > 0 ? `<button type="button" class="t2i-remove" aria-label="칸 삭제">×</button>` : '');
        item.querySelector('textarea').value = text;
        box.appendChild(item);
    });
}

// ===================== Tag Browser + 자동완성 =====================
let tagCache = null;                 // [{tag, type, dup, count}, ...] (서버에서 count 내림차순)
const checkedTags = new Set();       // 페이지를 옮겨도 체크 상태 유지
let lastPrompt = null;               // 마지막으로 포커스한 프롬프트 칸

// ---- 프롬프트에 태그 넣기/빼기 ----
function targetPrompt() {
    if (lastPrompt && document.body.contains(lastPrompt)) return lastPrompt;
    return document.querySelector('#t2i-prompts .t2i-prompt');
}

function toggleTagInPrompt(tag, on) {
    const ta = targetPrompt();
    if (!ta) return;
    const parts = ta.value.split(',').map(s => s.trim()).filter(Boolean);
    const next = on ? (parts.includes(tag) ? parts : [...parts, tag]) : parts.filter(p => p !== tag);
    ta.value = next.join(', ');
    ta.dispatchEvent(new Event('input', { bubbles: true }));   // savePrompts 연동
}

// ---- 목록 그리기 (많으면 100개씩) ----
function fillList(ul, list, limit = 100) {
    ul.innerHTML = '';
    for (const tag of list.slice(0, limit)) {
        const li = document.createElement('li');
        const label = document.createElement('label');
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.value = tag;
        input.checked = checkedTags.has(tag);
        label.append(input, ' ', tag);
        li.appendChild(label);
        ul.appendChild(li);
    }
    if (list.length > limit) {
        const li = document.createElement('li');
        const more = document.createElement('button');
        more.type = 'button';
        more.className = 't2i-tag-more';
        more.textContent = `more (${list.length - limit})`;
        more.addEventListener('click', () => fillList(ul, list, limit + 200));
        li.appendChild(more);
        ul.appendChild(li);
    }
}

function renderTagBrowser(tags, query = '') {
    const box = document.getElementById('t2i-tag-tree');
    const q = query.trim().toLowerCase().replace(/\s+/g, '_');
    const groups = new Map();                          // type → [tag, ...]
    for (const t of tags) {
        if (q && !t.tag.toLowerCase().includes(q)) continue;
        if (!groups.has(t.type)) groups.set(t.type, []);
        groups.get(t.type).push(t.tag);
    }
    box.innerHTML = '';
    if (!groups.size) { box.textContent = 'No tags'; return; }

    for (const [type, list] of groups) {
        const d = document.createElement('details');
        const s = document.createElement('summary');
        s.textContent = `${type} (${list.length})`;
        const ul = document.createElement('ul');
        d.append(s, ul);
        const open = () => { if (!ul.childElementCount) fillList(ul, list); };   // 펼칠 때만 그림
        d.addEventListener('toggle', () => { if (d.open) open(); });
        if (q) { d.open = true; open(); }              // 검색 중엔 자동으로 펼침
        box.appendChild(d);
    }
}

function initTagBrowser() {
    const box = document.getElementById('t2i-tag-tree');
    const search = document.getElementById('t2i-tag-search');

    box.addEventListener('change', e => {
        const cb = e.target.closest('input[type="checkbox"]');
        if (!cb) return;
        if (cb.checked) checkedTags.add(cb.value); else checkedTags.delete(cb.value);
        toggleTagInPrompt(cb.value, cb.checked);
    });

    let timer;
    search.addEventListener('input', () => {
        clearTimeout(timer);
        timer = setTimeout(() => tagCache && renderTagBrowser(tagCache, search.value), 150);
    });

    loadTags()
        .then(tags => renderTagBrowser(tags))
        .catch(err => { box.textContent = err.message; });
}

// ---- 프롬프트 자동완성 ----
const ac = { box: null, items: [], idx: -1, ta: null };

function suggest(q) {
    const key = q.toLowerCase().replace(/\s+/g, '_');
    const starts = [], inc = [];
    for (const { tag } of tagCache) {
        const t = tag.toLowerCase();
        if (t.startsWith(key)) { starts.push(tag); if (starts.length >= 8) break; }
        else if (inc.length < 8 && t.includes(key)) inc.push(tag);
    }
    return starts.concat(inc).slice(0, 8);
}

function hideAc() {
    ac.items = []; ac.idx = -1;
    if (ac.box) ac.box.hidden = true;
}

function showAc(ta, items) {
    ac.ta = ta; ac.items = items; ac.idx = 0;
    if (!ac.box) {
        ac.box = document.createElement('div');
        ac.box.className = 't2i-ac';
        ac.box.addEventListener('mousedown', e => {          // 포커스 유지
            e.preventDefault();
            const li = e.target.closest('[data-i]');
            if (li) acceptAc(Number(li.dataset.i));
        });
        document.body.appendChild(ac.box);
    }
    const r = ta.getBoundingClientRect();
    ac.box.style.left = `${r.left}px`;
    ac.box.style.top = `${r.bottom + 2}px`;
    ac.box.style.width = `${r.width}px`;
    renderAc();
    ac.box.hidden = false;
}

function renderAc() {
    ac.box.innerHTML = '';
    ac.items.forEach((tag, i) => {
        const div = document.createElement('div');
        div.dataset.i = i;
        div.className = 't2i-ac-item' + (i === ac.idx ? ' active' : '');
        div.textContent = tag;
        ac.box.appendChild(div);
    });
}

function acceptAc(i) {
    const ta = ac.ta, tag = ac.items[i];
    if (!ta || !tag) return;
    const pos = ta.selectionStart;
    const before = ta.value.slice(0, pos);
    const start = before.lastIndexOf(',') + 1;
    const head = ta.value.slice(0, start).replace(/\s*$/, '');
    const left = (head ? head + ' ' : '') + tag + ', ';
    ta.value = left + ta.value.slice(pos).replace(/^\s+/, '');
    ta.selectionStart = ta.selectionEnd = left.length;
    hideAc();
    ta.dispatchEvent(new Event('input', { bubbles: true }));   // 저장 (자동완성은 아래에서 다시 뜰 수 있음)
    hideAc();
}

function initAutocomplete() {
    const prompts = document.getElementById('t2i-prompts');

    prompts.addEventListener('focusin', e => {
        if (e.target.classList.contains('t2i-prompt')) lastPrompt = e.target;
    });
    prompts.addEventListener('focusout', hideAc);

    prompts.addEventListener('input', e => {
        const ta = e.target;
        if (!ta.classList.contains('t2i-prompt') || !tagCache) return;
        const before = ta.value.slice(0, ta.selectionStart);
        const token = before.slice(before.lastIndexOf(',') + 1).trim();
        if (token.length < 2) return hideAc();
        const items = suggest(token);
        items.length ? showAc(ta, items) : hideAc();
    });

    prompts.addEventListener('keydown', e => {
        if (!ac.items.length || ac.box.hidden) return;
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            const n = ac.items.length;
            ac.idx = (ac.idx + (e.key === 'ArrowDown' ? 1 : n - 1)) % n;
            renderAc();
        } else if (e.key === 'Enter' || e.key === 'Tab') {
            e.preventDefault();
            acceptAc(ac.idx);
        } else if (e.key === 'Escape') {
            hideAc();
        }
    });
}


// ===================== 레이아웃: 접기 / 폭·높이 조절 =====================
const LAYOUT_KEY = 't2i-layout';

function loadLayout() {
    try { return JSON.parse(localStorage.getItem(LAYOUT_KEY)) || {}; } catch { return {}; }
}

function saveLayout(patch) {
    try { localStorage.setItem(LAYOUT_KEY, JSON.stringify({ ...loadLayout(), ...patch })); } catch { /* 저장 불가면 무시 */ }
}

function initLayout(root) {
    const ctrl = root.querySelector('.t2i-controls');
    const top = ctrl.querySelector('.t2i-pane-prompts');
    const toggle = document.getElementById('t2i-tree-toggle');
    const lay = loadLayout();
    const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

    // --- Tag Browser 열기 / 닫기 (왼쪽으로 접힘) ---
    const applyTree = open => {
        root.classList.toggle('tree-collapsed', !open);
        toggle.textContent = open ? '«' : '»';
        toggle.setAttribute('aria-expanded', String(open));
    };
    applyTree(lay.treeOpen !== false);
    toggle.addEventListener('click', () => {
        const open = root.classList.contains('tree-collapsed');
        applyTree(open);
        saveLayout({ treeOpen: open });
    });

    // --- 조절바 만들기: 입력 | 이미지판 (세로), 프롬프트 / 조건 (가로) ---
    const vBar = document.createElement('div');
    vBar.className = 't2i-resizer t2i-resizer-v';
    vBar.setAttribute('role', 'separator');
    vBar.setAttribute('aria-orientation', 'vertical');
    ctrl.after(vBar);

    const hBar = document.createElement('div');
    hBar.className = 't2i-resizer t2i-resizer-h';
    hBar.setAttribute('role', 'separator');
    hBar.setAttribute('aria-orientation', 'horizontal');
    top.after(hBar);

    const setCtrlW = w => root.style.setProperty('--ctrl-w', `${w}px`);
    const setTopH = h => ctrl.style.setProperty('--rows', `${h}px 1rem minmax(0, 1fr)`);

    if (lay.ctrlW) setCtrlW(lay.ctrlW);
    if (lay.topH) setTopH(lay.topH);

    // 드래그 공통 처리
    const makeDraggable = (bar, onMove, onEnd, onReset) => {
        bar.addEventListener('pointerdown', e => {
            if (e.button !== 0) return;
            e.preventDefault();
            bar.setPointerCapture(e.pointerId);
            bar.classList.add('dragging');
            document.documentElement.classList.add('t2i-resizing');
        });
        bar.addEventListener('pointermove', e => {
            if (bar.hasPointerCapture(e.pointerId)) onMove(e);
        });
        const end = e => {
            if (!bar.hasPointerCapture(e.pointerId)) return;
            bar.releasePointerCapture(e.pointerId);
            bar.classList.remove('dragging');
            document.documentElement.classList.remove('t2i-resizing');
            onEnd();
        };
        bar.addEventListener('pointerup', end);
        bar.addEventListener('pointercancel', end);
        bar.addEventListener('dblclick', onReset); 
    };

    // 입력 열 폭: 이미지판은 남는 폭을 차지함
    let ctrlW = lay.ctrlW;
    makeDraggable(vBar, e => {
        const left = ctrl.getBoundingClientRect().left;
        const maxW = root.getBoundingClientRect().right - left - vBar.offsetWidth - 240;
        ctrlW = Math.round(clamp(e.clientX - left, 320, Math.max(320, maxW)));
        setCtrlW(ctrlW);
    }, () => saveLayout({ ctrlW }), () => {
        root.style.removeProperty('--ctrl-w');
        ctrlW = undefined;
        saveLayout({ ctrlW: null });
    });

    // 프롬프트 높이: 조건 영역은 남는 높이를 차지함
    let topH = lay.topH;
    makeDraggable(hBar, e => {
        const t = ctrl.getBoundingClientRect().top;
        const maxH = ctrl.clientHeight - hBar.offsetHeight - 140;
        topH = Math.round(clamp(e.clientY - t, 90, Math.max(90, maxH)));
        setTopH(topH);
    }, () => saveLayout({ topH }), () => {
        ctrl.style.removeProperty('--rows');
        topH = undefined;
        saveLayout({ topH: null });
    });
}

// ---- 태그 로딩 (진행 중인 요청은 하나만 공유) ----
let tagPromise = null;

function loadTags() {
    if (tagCache) return Promise.resolve(tagCache);
    tagPromise ??= fetch(`${API_BASE}/api/tags`)
        .then(res => {
            if (!res.ok) throw new Error(`tags 로드 실패 (${res.status})`);
            return res.json();
        })
        .then(tags => (tagCache = tags))
        .catch(err => { tagPromise = null; throw err; });   // 실패하면 다음에 다시 시도
    return tagPromise;
}

// 스피너 + 경과 시간. 반환값은 타이머 정리 함수
function showTagLoading(box) {
    box.innerHTML =
        '<div class="t2i-loading"><span class="t2i-spinner"></span>' +
        '<span class="t2i-loading-text">Loading tags...</span></div>';
    const text = box.querySelector('.t2i-loading-text');
    const t0 = performance.now();
    const timer = setInterval(() => {
        text.textContent = `Loading tags... ${Math.floor((performance.now() - t0) / 1000)}s`;
    }, 500);
    return () => clearInterval(timer);
}

function initTagBrowser() {
    const box = document.getElementById('t2i-tag-tree');
    const search = document.getElementById('t2i-tag-search');

    box.addEventListener('change', e => {
        const cb = e.target.closest('input[type="checkbox"]');
        if (!cb) return;
        if (cb.checked) checkedTags.add(cb.value); else checkedTags.delete(cb.value);
        toggleTagInPrompt(cb.value, cb.checked);
    });

    let timer;
    search.addEventListener('input', () => {
        clearTimeout(timer);
        timer = setTimeout(() => tagCache && renderTagBrowser(tagCache, search.value), 150);
    });

    const load = () => {
        const stop = tagCache ? null : showTagLoading(box);   // 캐시가 있으면 깜빡이지 않게
        search.disabled = !tagCache;

        loadTags()
            .then(tags => renderTagBrowser(tags, search.value))
            .catch(err => {
                box.innerHTML = '';
                const msg = document.createElement('div');
                msg.className = 't2i-curve-msg error';
                msg.textContent = err.message;
                const retry = document.createElement('button');
                retry.type = 'button';
                retry.className = 't2i-tag-more';
                retry.textContent = 'Retry';
                retry.addEventListener('click', load);
                box.append(msg, retry);
            })
            .finally(() => {
                stop?.();
                search.disabled = false;
            });
    };
    load();
}