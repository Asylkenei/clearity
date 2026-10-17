const form = document.querySelector('#upload-form');
const fileInput = document.querySelector('#video-file');
const keyInput = document.querySelector('#api-key');
const button = document.querySelector('#submit-button');
const status = document.querySelector('#status');
const result = document.querySelector('#result');
const empty = document.querySelector('#empty-state');

function setStatus(text, state = '') {
  status.textContent = text;
  status.dataset.state = state;
}

fileInput.addEventListener('change', () => {
  document.querySelector('#file-name').textContent = fileInput.files[0]?.name || 'Файл пока не выбран';
});

async function request(url, options, key) {
  const response = await fetch(url, {
    ...options, headers: { 'X-API-Key': key }, signal: AbortSignal.timeout(60000)
  });
  const payload = await response.json();
  if (!response.ok) {
    const detail = payload.detail;
    throw new Error(typeof detail === 'string' ? detail : detail?.message || 'Ошибка запроса. Проверьте файл и ключ доступа.');
  }
  return payload;
}

function showResult(data, filename) {
  const metadata = document.querySelector('#metadata');
  metadata.replaceChildren();
  const values = [
    ['Длительность ≈', `${data.duration_sec.toFixed(1)} сек`],
    ['Размер кадра', `${data.width} × ${data.height}`],
    ['Частота кадров', `${data.fps} FPS`]
  ];
  for (const [label, value] of values) {
    const metric = document.createElement('div');
    metric.className = 'metric';
    const dt = document.createElement('dt'); dt.textContent = label;
    const dd = document.createElement('dd'); dd.textContent = value;
    metric.append(dt, dd); metadata.append(metric);
  }
  const frames = document.querySelector('#frames');
  frames.replaceChildren();
  for (const frame of data.frames) {
    const figure = document.createElement('figure');
    const image = document.createElement('img');
    image.src = `data:image/jpeg;base64,${frame.image_base64}`;
    image.alt = `Обзорный кадр на ${frame.timestamp_sec.toFixed(2)} секунде`;
    const caption = document.createElement('figcaption');
    caption.textContent = `${frame.timestamp_sec.toFixed(2)} сек от начала`;
    figure.append(image, caption); frames.append(figure);
  }
  empty.hidden = true; result.hidden = false;
  setStatus(`Видео «${filename}» прочитано. Проверка нарушений не выполнялась.`, 'done');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.mov')) return setStatus('Выберите файл .mov.', 'error');
  if (file.size > 100 * 1024 * 1024) return setStatus('Файл больше 100 МБ.', 'error');
  const key = keyInput.value.trim();
  button.disabled = true; fileInput.disabled = true;
  result.hidden = true; empty.hidden = false;
  setStatus('Загружаем видео на локальный сервер…');
  try {
    const body = new FormData(); body.append('file', file);
    const submitted = await request('/api/v1/preview', { method: 'POST', body }, key);
    setStatus('Видео получено. Читаем параметры и обзорные кадры…');
    const deadline = Date.now() + 5 * 60 * 1000;
    while (Date.now() < deadline) {
      const job = await request(`/api/v1/jobs/${submitted.job_id}`, {}, key);
      if (job.status === 'done') { showResult(job.result, file.name); return; }
      if (job.status === 'failed') throw new Error(job.error?.message || 'Не удалось прочитать видео.');
      await new Promise(resolve => setTimeout(resolve, 700));
    }
    throw new Error('Подготовка заняла больше пяти минут. Проверьте сервер.');
  } catch (error) {
    setStatus(error.name === 'TimeoutError' ? 'Сервер не ответил вовремя.' : error.message, 'error');
  } finally {
    button.disabled = false; fileInput.disabled = false;
  }
});
