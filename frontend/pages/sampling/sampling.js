/* sampling.js — txt2img 전용 */
import { initCommon, collectCommon, toSamplingRequest, postJSON, API_BASE } from '../common/common.js';

const API_URL = `${API_BASE}/api/sampling`;
const POLL_INTERVAL_MS = 300;

let results = [];   // data URL 배열

export async function init() {
    await initCommon({ size: true, onGenerate });
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

/* ===== 진행률 ===== */

function setProgress(step, total) {
    const bar = document.getElementById('t2i-progress');
    const text = document.getElementById('t2i-progress-text');
    bar.max = total || 1;
    bar.value = step;
    text.textContent = total ? `${step} / ${total}` : '';
}

function showProgress(visible) {
    document.querySelector('.t2i-progress-wrap').classList.toggle('active', visible);
}

async function pollJob(jobId) {
    while (true) {
        const res = await fetch(`${API_URL}/${jobId}`);
        if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
        const job = await res.json();

        setProgress(job.step, job.total);

        if (job.status === 'done') return job;
        if (job.status === 'error') throw new Error(job.error);
        await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
    }
}

/* ===== 생성 ===== */

async function onGenerate() {
    const btn = document.getElementById('t2i-generate');
    btn.disabled = true;                // 보내는 동안 중복 클릭 방지
    setProgress(0, 0);
    showProgress(true);
    try {
        const params = collectCommon();
        const body = toSamplingRequest(params);     // prompts: ["prompt1", "prompt2", ...]
        console.log('send', body);

        const { job_id } = await postJSON(API_URL, body);
        const data = await pollJob(job_id);
        setResults(data.images.map(b64 => `data:image/png;base64,${b64}`));
    } catch (err) {
        console.error(err);
        alert(err.message);
    } finally {
        btn.disabled = false;
        showProgress(false);
    }
}