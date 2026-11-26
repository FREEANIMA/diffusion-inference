/* sampling.js — txt2img 전용 */
import { initCommon, collectCommon, toSamplingRequest, postJSON, API_BASE } from '../common/common.js';

const API_URL = `${API_BASE}/api/sampling`;
const POLL_INTERVAL_MS = 300;

let results = [];   // data URL 배열

// Generate 버튼이 생성 중에는 Stop 버튼이 됨 (클릭 핸들러는 onGenerate 하나)
let running = false;            // 생성 중인지
let activeJob = null;           // 실행 중인 job_id (요청 응답 전에는 null)
let cancelRequested = false;    // job_id를 받기 전에 Stop을 눌렀는지

export async function init() {
    await initCommon({ size: true, onGenerate });
    if (running) setGenButton('running');         // 생성 중에 돌아온 경우
}

/* ===== 결과 표시 ===== */

function showResult(i) {
    const box = document.getElementById('t2i-image');
    box.innerHTML = '';
    const img = document.createElement('img');
    img.src = results[i];
    img.style.maxWidth = '100%';
    box.appendChild(img);

    document.querySelectorAll('#t2i-nums button').forEach((b, idx) => {
        b.classList.toggle('active', idx === i);
    });
}

function setResults(urls) {
    results = urls;
    const nums = document.getElementById('t2i-nums');
    nums.innerHTML = '';
    urls.forEach((_, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = i + 1;
        b.addEventListener('click', () => showResult(i));
        nums.appendChild(b);
    });
    if (urls.length) showResult(0);
}

/* ===== 미리보기 (x0_hat) ===== */

function showPreview(b64) {
    const box = document.getElementById('t2i-image');
    if (!box) return;
    let img = box.querySelector('img.preview');
    if (!img) {
        box.innerHTML = '';
        img = document.createElement('img');
        img.className = 'preview';
        img.style.maxWidth = '100%';
        box.appendChild(img);
    }
    img.src = `data:image/png;base64,${b64}`;
}

/* ===== 진행률 ===== */

function setProgress(step, total) {
    const bar = document.getElementById('t2i-progress');
    const text = document.getElementById('t2i-progress-text');
    if (!bar || !text) return;
    bar.max = total || 1;
    bar.value = step;
    text.textContent = total ? `${step} / ${total}` : '';
}

function showProgress(visible) {
    document.querySelector('.t2i-progress-wrap')?.classList.toggle('active', visible);
}

/* ===== 정지 ===== */

function setGenButton(state) {  // 'idle' | 'running' | 'stopping'
    const btn = document.getElementById('t2i-generate');
    if (!btn) return;
    btn.disabled = state === 'stopping';
    btn.textContent = state === 'running' ? 'Stop' : state === 'stopping' ? 'Stopping...' : 'Generate';
}

async function sendCancel() {
    if (!activeJob) return;     // 아직 job_id가 없음 → 받는 즉시 보냄
    try {
        await fetch(`${API_URL}/${activeJob}/cancel`, { method: 'POST' });
    } catch (err) {
        console.error(err);
    }
}

function requestCancel() {
    cancelRequested = true;
    setGenButton('stopping');
    return sendCancel();
}

/* ===== 폴링 ===== */

async function pollJob(jobId) {
    let seen = 0;                               // 이미 받은 preview_seq (같은 프리뷰 재전송 방지)
    while (true) {
        const res = await fetch(`${API_URL}/${jobId}?seen=${seen}`);
        if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
        const job = await res.json();

        setProgress(job.step, job.total);
        if (job.preview) {
            seen = job.preview_seq;
            showPreview(job.preview);
        }

        if (job.status === 'done' || job.status === 'cancelled') return job;
        if (job.status === 'error') throw new Error(job.error);
        await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
    }
}

/* ===== 생성 ===== */

async function onGenerate() {
    if (running) return requestCancel();          // 생성 중에 누르면 Stop

    running = true;
    activeJob = null;
    cancelRequested = false;
    setGenButton('running');
    setProgress(0, 0);
    showProgress(true);
    try {
        const params = collectCommon();
        const body = toSamplingRequest(params);     // prompts: ["prompt1", "prompt2", ...]
        console.log('send', body);

        const { job_id } = await postJSON(API_URL, body);
        activeJob = job_id;
        if (cancelRequested) sendCancel();          // 응답 전에 Stop을 눌렀다면 지금 보냄

        const data = await pollJob(job_id);
        if (data.status === 'cancelled') return;    // 마지막 미리보기는 화면에 그대로 남김
        setResults(data.images.map(b64 => `data:image/png;base64,${b64}`));
    } catch (err) {
        console.error(err);
        alert(err.message);
    } finally {
        running = false;
        activeJob = null;
        cancelRequested = false;
        setGenButton('idle');
        showProgress(false);
    }
}