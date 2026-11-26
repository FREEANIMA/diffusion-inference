/* inpainting.js — inpainting 전용
 * Mask: 좌클릭 = 수정(빨강), 우클릭 = 보존(파랑)   Eraser: 좌/우클릭 드래그로 지움 (Mask 상태에서 Alt+드래그도 지우개)
 * 서버로 보내는 마스크: 빨강(수정)이 있으면 빨강만 수정 / 빨강이 없으면 (크롭 있으면 크롭, 없으면 이미지) 전체를 수정하되 파랑(보존)만 제외
 * Crop(선택): W, H는 32의 배수. 크롭하면 그 영역으로 확대됨. 안 해도 전송 가능
 * Generate 후: 이미지판이 수정된 이미지로 바뀜. 위쪽 번호(prompt)로 결과 전환, Compare로 원본과 비교
 * 확대/이동: 휠 = 확대/축소, 가운데 버튼(또는 Space + 드래그) = 이동
 */
import { initCommon, collectCommon, toSamplingRequest, postJSON } from '../common/common.js';

const GRID = 32;                 // W, H 모두 32의 배수
const MAX_CROP = 1024;           // 크롭 최대 크기 (W, H 각각)
const COLOR = { edit: '#ff3b30', keep: '#2f6bff' };
import { API_BASE } from '../common/common.js';
const API_URL = `${API_BASE}/api/inpainting`;
const USE_MOCK = false;          // true: 전송 후 가짜 결과를 이미지판에 띄워 화면(번호 전환/Compare) 테스트

// W, H가 모두 32의 배수여야 함
// 예) 32×64, 64×96, 64×192 ✔ / 50×64 ✘
const validSize = (w, h) =>
    w > 0 && h > 0 && w % GRID === 0 && h % GRID === 0;

// 상태는 window에 보관: 탭을 옮겨 다니며 페이지 DOM이 새로 만들어져도(모듈이 다시 로드돼도) 유지됨
const S = (window.__inpaintState ??= {
    img: null,                  // 원본 이미지 (항상 원본 그대로)
    mask: null,                 // 원본 크기의 마스크 캔버스 (빨강/파랑)
    mctx: null,
    tool: null,                 // 'mask' | 'eraser' | 'crop' | null
    crop: null,                 // { x, y, w, h } 원본 픽셀 기준
    view: { s: 1, ox: 0, oy: 0 },   // 원본 좌표 → 화면 좌표: screen = img * s + o
    brush: 40,                  // 원본 픽셀 기준 굵기
    drag: null,
    hover: null,
    space: false,
    results: [],                // results[i] = { img, rect } — i번째 prompt의 결과
    cur: 0,                     // 선택된 번호 (0부터)
    showRes: false,             // true: 수정된 결과를 보여줌 / false: 원본 + 마스크 편집
    compare: false,             // true: 원본 | 결과 분할 비교
    cmpX: 0.5,                  // 분할선 위치 (결과 영역 폭 기준 0~1)
    cleanup: null,
});

let cv, ctx, stage, cw = 0, ch = 0, dpr = 1;
const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export async function init() {
    // 크기는 crop이 정하므로 Width/Height 칸은 뺌
    await initCommon({ size: false, inpaint: true, onGenerate });
    bind();
}

// ===================== 화면 / 좌표 =====================
const fullRect = () => ({ x: 0, y: 0, w: S.img.naturalWidth, h: S.img.naturalHeight });
const curResult = () => S.results[S.cur] || null;
const shownResult = () => (S.showRes ? curResult() : null);

function resize() {
    const r = stage.getBoundingClientRect();
    if (!r.width || !r.height) return;          // 아직 안 보이는 상태
    dpr = window.devicePixelRatio || 1;
    cw = r.width;
    ch = r.height;
    cv.width = Math.round(cw * dpr);
    cv.height = Math.round(ch * dpr);
    fit();
}

// 결과 보기: 결과 영역 / 편집: 크롭 영역(없으면 이미지 전체)이 꽉 차게 = 확대
function fit() {
    if (!S.img) return draw();
    const res = shownResult();
    const r = res ? res.rect : (S.crop || fullRect());
    const pad = 16;
    const s = Math.min((cw - pad * 2) / r.w, (ch - pad * 2) / r.h);
    S.view = {
        s,
        ox: (cw - r.w * s) / 2 - r.x * s,
        oy: (ch - r.h * s) / 2 - r.y * s,
    };
    draw();
}

function toImg(e) {
    const b = cv.getBoundingClientRect();
    const { s, ox, oy } = S.view;
    return { x: (e.clientX - b.left - ox) / s, y: (e.clientY - b.top - oy) / s };
}

// ===================== 그리기 =====================
// 결과 이미지를 crop 위치에 덧그림. 서버가 크롭 크기 이미지를 주든, 전체 크기 이미지를 주든 처리
function drawPatch(c, res) {
    const { img, rect: r } = res;
    const isFull = img.naturalWidth === S.img.naturalWidth && img.naturalHeight === S.img.naturalHeight
        && (r.w !== S.img.naturalWidth || r.h !== S.img.naturalHeight);
    if (isFull) c.drawImage(img, r.x, r.y, r.w, r.h, r.x, r.y, r.w, r.h);
    else c.drawImage(img, r.x, r.y, r.w, r.h);
}

function pill(text, x, y) {
    ctx.font = '12px sans-serif';
    const tw = ctx.measureText(text).width + 12;
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(x, y, tw, 20);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, x + 6, y + 14);
    return tw;
}

function draw() {
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    if (!S.img) return;

    const { s, ox, oy } = S.view;
    const res = shownResult();

    ctx.save();
    ctx.translate(ox, oy);
    ctx.scale(s, s);
    ctx.imageSmoothingEnabled = s < 3;          // 많이 확대하면 픽셀이 보이게
    ctx.drawImage(S.img, 0, 0);                 // 원본

    if (res) {
        ctx.save();
        if (S.compare) {                        // 분할선 오른쪽만 결과, 왼쪽은 원본
            const dx = res.rect.x + S.cmpX * res.rect.w;
            ctx.beginPath();
            ctx.rect(dx, res.rect.y, res.rect.x + res.rect.w - dx, res.rect.h);
            ctx.clip();
        }
        drawPatch(ctx, res);
        ctx.restore();
    } else {
        ctx.globalAlpha = 0.5;
        ctx.drawImage(S.mask, 0, 0);
    }
    ctx.restore();

    // 비교 분할선
    if (res && S.compare) {
        const r = res.rect;
        const x = (r.x + S.cmpX * r.w) * s + ox;
        const y0 = r.y * s + oy, y1 = (r.y + r.h) * s + oy;
        ctx.save();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.shadowColor = 'rgba(0,0,0,0.6)';
        ctx.shadowBlur = 4;
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
        ctx.beginPath(); ctx.arc(x, (y0 + y1) / 2, 9, 0, Math.PI * 2);
        ctx.fillStyle = '#fff'; ctx.fill();
        ctx.restore();
        pill('Original', r.x * s + ox + 6, y0 + 6);
        const tw = ctx.measureText('Edited').width + 12;
        pill('Edited', (r.x + r.w) * s + ox - tw - 6, y0 + 6);
    }

    // 크롭 영역: 바깥을 어둡게 + 테두리 + 크기 표시 (편집 보기에서만)
    const r = S.drag && S.drag.type === 'crop' ? S.drag.rect : (res ? null : S.crop);
    if (r) {
        const x = r.x * s + ox, y = r.y * s + oy, w = r.w * s, h = r.h * s;
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.55)';
        ctx.beginPath();
        ctx.rect(0, 0, cw, ch);
        ctx.rect(x, y, w, h);
        ctx.fill('evenodd');
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
        ctx.restore();
        const label = `${r.w} × ${r.h}`;
        ctx.font = '12px sans-serif';
        const tw = ctx.measureText(label).width + 12;
        pill(label, clamp(x, 0, cw - tw), clamp(y - 20, 0, ch - 20));
    }

    // 브러시 미리보기 원
    if (S.hover && !res && (S.tool === 'mask' || S.tool === 'eraser') && !S.drag) {
        ctx.save();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(S.hover.x, S.hover.y, (S.brush * s) / 2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
    }
}

// ===================== 마스크 =====================
function paintLine(a, b, kind, erase) {
    const c = S.mctx;
    c.save();
    c.lineCap = c.lineJoin = 'round';
    c.lineWidth = S.brush;
    c.strokeStyle = COLOR[kind];
    if (erase) c.globalCompositeOperation = 'destination-out';
    c.beginPath();
    c.moveTo(a.x, a.y);
    c.lineTo(b.x + 0.01, b.y);                   // 클릭만 해도 점이 찍히도록
    c.stroke();
    c.restore();
}

// 마스크 픽셀 계산 (수정 = 255, 그 외 = 0). data: RGBA, r: data가 담고 있는 영역, area: 수정 대상 영역
//  · area 안에 빨강(수정)이 있으면 → 빨강만 흰색
//  · 빨강이 하나도 없으면 → area 전체를 흰색으로 하고 파랑(보존)만 검정으로 뺌
//    (크롭 일부만 수정하면 주변과 안 맞는 문제를 피하려고, 크롭 전체를 새로 생성하되 보존 영역만 유지)
//  · area 밖은 항상 검정
function buildMaskPixels(data, r, area) {
    const n = r.w * r.h;
    const isRed = k => data[k * 4 + 3] > 127 && data[k * 4] > data[k * 4 + 2];
    const isBlue = k => data[k * 4 + 3] > 127 && data[k * 4 + 2] > data[k * 4];
    const inArea = k => {
        const x = r.x + (k % r.w), y = r.y + ((k / r.w) | 0);
        return x >= area.x && x < area.x + area.w && y >= area.y && y < area.y + area.h;
    };

    let hasRed = false;
    for (let k = 0; k < n && !hasRed; k++) if (inArea(k) && isRed(k)) hasRed = true;

    const out = new Uint8ClampedArray(n * 4);
    for (let k = 0; k < n; k++) {
        const white = inArea(k) && (hasRed ? isRed(k) : !isBlue(k));
        out[k * 4] = out[k * 4 + 1] = out[k * 4 + 2] = white ? 255 : 0;
        out[k * 4 + 3] = 255;
    }
    return { out, hasRed };
}

// r 영역의 마스크를 PNG로. area = 수정 대상 영역 (크롭, 없으면 전체)
function maskDataURL(r, area) {
    const src = S.mctx.getImageData(r.x, r.y, r.w, r.h);
    const { out, hasRed } = buildMaskPixels(src.data, r, area);
    const c = document.createElement('canvas');
    c.width = r.w;
    c.height = r.h;
    c.getContext('2d').putImageData(new ImageData(out, r.w, r.h), 0, 0);
    return { url: c.toDataURL('image/png'), hasRed };
}

function cropDataURL(r) {
    const c = document.createElement('canvas');
    c.width = r.w;
    c.height = r.h;
    c.getContext('2d').drawImage(S.img, r.x, r.y, r.w, r.h, 0, 0, r.w, r.h);
    return c.toDataURL('image/png');
}

// ===================== 크롭 (W, H: 32 배수) =====================
let sizeCache = { key: '', list: [] };


// 이미지 안에 들어가는 유효한 (w, h) 조합 전체 (이미지마다 한 번만 계산)
function sizeOptions(W, H) {
    const key = `${W}x${H}`;
    if (sizeCache.key === key) return sizeCache.list;
    const maxW = Math.min(W, MAX_CROP), maxH = Math.min(H, MAX_CROP);
    const list = [];
    for (let w = GRID; w <= maxW; w += GRID) {
        for (let h = GRID; h <= maxH; h += GRID) {
            if (validSize(w, h)) list.push([w, h]);
        }
    }
    sizeCache = { key, list };
    return list;
}

function cropRect(p0, p1) {
    const W = S.img.naturalWidth, H = S.img.naturalHeight;
    const options = sizeOptions(W, H);
    if (!options.length) return null;                 // 이미지가 32보다 작음

    // 드래그한 크기에 가장 가까운 유효 조합을 고름
    const dw = Math.abs(p1.x - p0.x), dh = Math.abs(p1.y - p0.y);
    let best = options[0], bestD = Infinity;
    for (const o of options) {
        const d = (o[0] - dw) ** 2 + (o[1] - dh) ** 2;
        if (d < bestD) { bestD = d; best = o; }
    }
    const [w, h] = best;

    // 드래그 방향 쪽으로 늘리고, 이미지 밖으로 나가면 안으로 밀어 넣음
    const x = clamp(Math.round(p1.x >= p0.x ? p0.x : p0.x - w), 0, W - w);
    const y = clamp(Math.round(p1.y >= p0.y ? p0.y : p0.y - h), 0, H - h);
    return { x, y, w, h };
}

// ===================== 이미지 불러오기 =====================
const loadImage = url => new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error('이미지를 불러오지 못했습니다'));
    im.src = url;
});

async function loadFile(file) {
    if (!file || !file.type.startsWith('image/')) return setStatus('이미지 파일만 열 수 있습니다', true);
    let img;
    try {
        img = await loadImage(URL.createObjectURL(file));
    } catch (err) {
        return setStatus(err.message, true);
    }
    S.img = img;
    S.mask = document.createElement('canvas');
    S.mask.width = img.naturalWidth;
    S.mask.height = img.naturalHeight;
    S.mctx = S.mask.getContext('2d');
    S.crop = null;
    S.results = [];                                   // 새 이미지면 이전 결과는 버림
    S.showRes = false;
    S.compare = false;
    $('inp-drop').hidden = true;
    refreshUI();
    resize();
    const ok = validSize(img.naturalWidth, img.naturalHeight);
    setStatus(`${img.naturalWidth} × ${img.naturalHeight}` + (ok ? '' : ' — 크기가 32의 배수가 아님. 필요하면 Crop 사용'));
}

// ===================== UI 상태 =====================
function setStatus(text, isError = false) {
    const el = $('inp-status');
    if (!el) return;                                  // 다른 탭에 가 있는 경우
    el.textContent = text;
    el.classList.toggle('error', isError);
}

const TOOL_BTN = { mask: 'inp-mask', eraser: 'inp-eraser', crop: 'inp-crop' };

// 버튼 눌림 표시 / 커서를 상태에 맞춤
function refreshUI() {
    for (const [t, id] of Object.entries(TOOL_BTN)) $(id).setAttribute('aria-pressed', String(t === S.tool));
    $('inp-compare').setAttribute('aria-pressed', String(S.showRes && S.compare));
    cv.style.cursor = S.showRes
        ? (S.compare ? 'ew-resize' : 'default')
        : S.tool === 'crop' ? 'crosshair'
        : (S.tool === 'mask' || S.tool === 'eraser') ? 'none' : 'default';
}

// name: 'mask' | 'eraser' | 'crop' | null  (한 번에 하나만 켜짐). 도구를 켜면 편집 보기로 돌아감
function setTool(name) {
    S.tool = name;
    const leaveResult = !!name && S.showRes;
    if (name) { S.showRes = false; S.compare = false; }
    refreshUI();
    if (leaveResult) fit(); else draw();
}

const toggleTool = name => setTool(S.tool === name ? null : name);

// 결과 보기 ↔ 편집 보기
function setResultView(on) {
    S.showRes = on && !!curResult();
    if (S.showRes) S.tool = null; else S.compare = false;
    refreshUI();
    fit();
}

// 번호 버튼(common.js가 그림)에서 현재 선택된 번호 읽기
function syncCur() {
    const a = document.querySelector('#t2i-nums .t2i-num.active');
    const idx = a ? Number(a.dataset.index) : 0;
    const changed = idx !== S.cur;
    S.cur = idx;
    return changed;
}

function onNumsChange() {
    if (syncCur() && curResult()) setResultView(true);   // 번호를 눌러 바꾸면 그 결과를 보여줌
    else if (S.showRes && !curResult()) setResultView(false);
    else draw();
}

// ===================== 바인딩 =====================
function bind() {
    S.cleanup?.();                                    // 이전 페이지 인스턴스의 리스너 정리
    stage = $('inp-stage');
    cv = $('inp-canvas');
    ctx = cv.getContext('2d');

    const ro = new ResizeObserver(resize);
    ro.observe(stage);

    // 번호 전환 / prompt 추가·삭제 감지 (common.js가 번호 버튼을 매번 다시 그림)
    const mo = new MutationObserver(onNumsChange);
    mo.observe($('t2i-nums'), { childList: true });

    // prompt 삭제 시 해당 번호의 결과도 같이 제거 (common.js 처리보다 먼저 실행되도록 capture)
    const onRemove = e => {
        const btn = e.target.closest('.t2i-remove');
        if (!btn) return;
        const items = [...document.querySelectorAll('#t2i-prompts .t2i-prompt-item')];
        const i = items.indexOf(btn.closest('.t2i-prompt-item'));
        if (i >= 0) S.results.splice(i, 1);
    };
    $('t2i-prompts').addEventListener('click', onRemove, true);

    // --- 열기 (버튼 / 클릭 / 드롭) ---
    const file = $('inp-file');
    $('inp-open').addEventListener('click', () => file.click());
    $('inp-drop').addEventListener('click', () => file.click());
    file.addEventListener('change', () => { loadFile(file.files[0]); file.value = ''; });
    stage.addEventListener('dragover', e => { e.preventDefault(); stage.classList.add('dragover'); });
    stage.addEventListener('dragleave', () => stage.classList.remove('dragover'));
    stage.addEventListener('drop', e => {
        e.preventDefault();
        stage.classList.remove('dragover');
        loadFile(e.dataTransfer.files[0]);
    });

    // --- 도구 버튼 ---
    for (const t of Object.keys(TOOL_BTN)) {
        $(TOOL_BTN[t]).addEventListener('click', () => {
            if (!S.img) return setStatus('먼저 이미지를 여세요', true);
            toggleTool(t);
        });
    }
    $('inp-upscale').addEventListener('click', onUpscale);
    $('inp-reset').addEventListener('click', () => {
        S.crop = null;                                  // 크롭 해제 + 전체 보기
        if (S.mctx) S.mctx.clearRect(0, 0, S.mask.width, S.mask.height);   // 마스크 전부 삭제
        S.tool = null;
        S.showRes = false;                              // 편집 보기로 (결과는 번호를 눌러 다시 볼 수 있음)
        S.compare = false;
        refreshUI();
        fit();
        setStatus('');
    });
    $('inp-compare').addEventListener('click', () => {
        if (!curResult()) return setStatus('Generate 후에 원본과 비교할 수 있습니다', true);
        const turnOn = !(S.showRes && S.compare);
        setResultView(true);
        S.compare = turnOn;
        S.cmpX = 0.5;
        refreshUI();
        draw();
    });
    $('inp-brush').value = S.brush;
    $('inp-brush').addEventListener('input', e => { S.brush = Number(e.target.value); draw(); });

    // --- 캔버스 입력 ---
    cv.addEventListener('contextmenu', e => e.preventDefault());   // 우클릭 = 보존 마스크

    const setCmp = (p, r) => { S.cmpX = clamp((p.x - r.x) / r.w, 0, 1); };

    cv.addEventListener('pointerdown', e => {
        if (!S.img) return;
        cv.setPointerCapture(e.pointerId);
        const p = toImg(e);
        const res = shownResult();

        if (e.button === 1 || (e.button === 0 && S.space)) {
            S.drag = { type: 'pan', lx: e.clientX, ly: e.clientY };
            cv.classList.add('panning');
        } else if (res) {
            if (S.compare && e.button === 0) {          // 비교 분할선 이동
                S.drag = { type: 'cmp', rect: res.rect };
                setCmp(p, res.rect);
            }
        } else if (S.tool === 'crop') {
            if (e.button === 0) S.drag = { type: 'crop', p0: p, rect: null };
        } else if ((S.tool === 'mask' || S.tool === 'eraser') && (e.button === 0 || e.button === 2)) {
            // Mask: 좌클릭 = 수정(빨강) / 우클릭 = 보존(파랑). Eraser: 좌·우클릭 모두 지움
            const kind = e.button === 0 ? 'edit' : 'keep';
            const erase = S.tool === 'eraser' || e.altKey;
            S.drag = { type: 'paint', kind, erase, last: p };
            paintLine(p, p, kind, erase);
        }
        draw();
    });

    cv.addEventListener('pointermove', e => {
        const b = cv.getBoundingClientRect();
        S.hover = { x: e.clientX - b.left, y: e.clientY - b.top };
        const d = S.drag;
        if (d) {
            if (d.type === 'pan') {
                S.view.ox += e.clientX - d.lx;
                S.view.oy += e.clientY - d.ly;
                d.lx = e.clientX;
                d.ly = e.clientY;
            } else if (d.type === 'paint') {
                const p = toImg(e);
                paintLine(d.last, p, d.kind, d.erase);
                d.last = p;
            } else if (d.type === 'crop') {
                d.rect = cropRect(d.p0, toImg(e));
            } else if (d.type === 'cmp') {
                setCmp(toImg(e), d.rect);
            }
        }
        draw();
    });

    const end = () => {
        const d = S.drag;
        S.drag = null;
        cv.classList.remove('panning');
        if (d && d.type === 'crop') {
            if (d.rect) {
                S.crop = d.rect;
                setTool(null);
                fit();                                  // 크롭 영역으로 확대
                setStatus(`Crop ${S.crop.w} × ${S.crop.h}`);
            } else if (!sizeOptions(S.img.naturalWidth, S.img.naturalHeight).length) {
                setStatus(`이미지가 ${GRID}px보다 작아 크롭할 수 없습니다`, true);
            }
        }
        draw();
    };
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
    cv.addEventListener('pointerleave', () => { S.hover = null; draw(); });

    // 휠 확대/축소 (커서 위치 기준)
    cv.addEventListener('wheel', e => {
        if (!S.img) return;
        e.preventDefault();
        const b = cv.getBoundingClientRect();
        const mx = e.clientX - b.left, my = e.clientY - b.top;
        const { s, ox, oy } = S.view;
        const ns = clamp(s * (e.deltaY < 0 ? 1.15 : 1 / 1.15), 0.05, 40);
        S.view = { s: ns, ox: mx - (mx - ox) * ns / s, oy: my - (my - oy) * ns / s };
        draw();
    }, { passive: false });

    // Space = 이동 모드, Esc = 크롭 도구 취소 (window 리스너라 정리 필요)
    const typing = e => /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    const onKeyDown = e => {
        if (typing(e)) return;
        if (e.code === 'Space' && S.img) { S.space = true; e.preventDefault(); }
        if (e.key === 'Escape' && S.tool === 'crop') setTool(null);
    };
    const onKeyUp = e => { if (e.code === 'Space') S.space = false; };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    S.cleanup = () => {
        window.removeEventListener('keydown', onKeyDown);
        window.removeEventListener('keyup', onKeyUp);
        ro.disconnect();
        mo.disconnect();
    };

    // --- 탭을 옮겨 왔을 때 이전 상태 복원 ---
    S.drag = null;
    S.hover = null;
    S.space = false;
    syncCur();                                        // common.js가 번호를 1번으로 되돌렸을 수 있음
    if (S.img) $('inp-drop').hidden = true;           // 이미지가 있으면 "Click or drop" 안내 숨김
    if (S.showRes && !curResult()) S.showRes = false;
    refreshUI();
    resize();
}

// ===================== 생성 / 업스케일 =====================
// 크롭이 있으면 크롭 영역, 없으면 원본 전체 (크기 검증은 크롭 도구가 이미 보장하므로 여기선 막지 않음)
function currentRect() {
    return S.crop || fullRect();
}

// 개발용: 백엔드 없이 화면 확인 (prompt마다 색조만 다르게)
async function mockInpaint(params) {
    const r = params.crop;
    await new Promise(res => setTimeout(res, 400));
    return params.prompts.map((_, i) => {
        const c = document.createElement('canvas');
        c.width = r.w;
        c.height = r.h;
        const g = c.getContext('2d');
        g.filter = `hue-rotate(${70 * (i + 1)}deg) saturate(1.6)`;
        g.drawImage(S.img, r.x, r.y, r.w, r.h, 0, 0, r.w, r.h);
        return c.toDataURL('image/png');
    });
}

const b64 = url => url.split(',')[1];     // "data:image/png;base64," 접두사 제거

// InPaintingRequest (prompt와 무관하므로 한 번만 만들어 재사용)
//   image_b64 : 원본 전체
//   mask_b64  : 원본과 같은 크기. 수정 = 흰색(255), 그 외 = 검정(0)  (빨강이 없으면 크롭 전체가 수정, 파랑만 제외)
//   crop_b64  : 크롭 영역만 잘라낸 이미지 (크롭 없으면 null)
//   crop_box  : [x, y, w, h] 원본 좌표 (크롭 없으면 null)
//   start_t   : Start-t 입력값 (inpainting 전용 칸)
function buildInpainting(params) {
    const full = fullRect();
    const mask = maskDataURL(full, S.crop || full);
    return {
        hasRed: mask.hasRed,
        inpainting: {
            image_b64: b64(cropDataURL(full)),
            mask_b64: b64(mask.url),
            crop_b64: S.crop ? b64(cropDataURL(S.crop)) : null,
            crop_box: S.crop ? [S.crop.x, S.crop.y, S.crop.w, S.crop.h] : null,
            start_t: params.startT,
        },
    };
}

// 요청만 보냄 (응답 처리는 아직 안 함). prompt 하나당 요청 하나, 순서대로
// body = { sampling: SamplingRequest, inpainting: InPaintingRequest }  — 파라미터와 이미지/마스크를 분리
// sampling.width/height = 생성 크기 (크롭이 있으면 크롭 크기, 없으면 원본 크기)
async function sendInpaint(params, r, inpainting) {
    const body = {
        sampling: toSamplingRequest(params, params.prompts, { width: r.w, height: r.h }),
        inpainting,
    };
    await postJSON(API_URL, body);
}

// 서버 응답 이미지(URL/data URL 배열, prompt 순서)를 이미지판에 반영. 응답 처리를 붙일 때 호출
async function showResults(urls, r) {
    const imgs = await Promise.all(urls.map(loadImage));
    S.results = imgs.map(img => ({ img, rect: { ...r } }));   // 결과마다 어느 영역의 결과인지 같이 저장
    S.compare = false;
    if (!cv || !cv.isConnected) return;       // 다른 탭으로 갔다면 상태만 저장
    syncCur();
    setResultView(true);                      // 이미지판을 수정된 이미지로 교체
    setStatus(`${imgs.length}개 생성됨`);
}

async function onGenerate() {
    if (!S.img) return setStatus('먼저 이미지를 여세요', true);
    const r = currentRect();
    const sizeWarn = validSize(r.w, r.h) ? '' : ` — ${r.w}×${r.h}는 32의 배수가 아님 (서버에서 처리 필요)`;

    const params = collectCommon();     // prompts, steps, cfg, seed, sigmas, cfgs, etas, styles
    const btn = $('t2i-generate');
    btn.disabled = true;                // 보내는 동안 중복 클릭 방지
    setStatus('Sending...');
    try {
        const { inpainting, hasRed } = buildInpainting(params);
        await sendInpaint(params, r, inpainting);
        setStatus(`${params.prompts.length}개 요청 전송됨` +
            (hasRed ? '' : ` — 빨강 마스크 없음: ${S.crop ? '크롭' : '이미지'} 전체 수정, 파랑 영역만 보존`) +
            sizeWarn);
        // TODO: 응답 이미지가 오면 showResults(urls, r) 호출
        if (USE_MOCK) await showResults(await mockInpaint({ prompts: params.prompts, crop: r }), r);
    } catch (err) {
        setStatus(err.message, true);
    } finally {
        btn.disabled = false;
    }
}

async function onUpscale() {
    if (!S.img) return setStatus('먼저 이미지를 여세요', true);
    const r = S.crop || fullRect();
    console.log('upscale', r);
    // TODO: cropDataURL(r)을 백엔드로 보내고, 결과로 S.img 교체
}