const form = document.querySelector('#upload-form');
const fileInput = document.querySelector('#video-file');
const keyInput = document.querySelector('#api-key');
const button = document.querySelector('#submit-button');
const status = document.querySelector('#status');
const result = document.querySelector('#result');
const empty = document.querySelector('#empty-state');
const mode = document.querySelector('#mode');

async function refreshModelStatus() {
  const label = document.querySelector('#model-status');
  try {
    const response = await fetch('/health', { signal: AbortSignal.timeout(5000) });
    const data = await response.json();
    label.textContent = data.analysis_ready ? 'Бинарный анализ готов. MOV до 3 минут. Результат требует проверки человеком.' : data.inspection_ready
      ? 'Модель подключена. Видео до 3 минут; обработка может занять до 4 минут.'
      : data.public_mode ? 'Модель выключена или загружается. Попросите владельца запустить её. Обзорные кадры доступны.'
        : 'Модель выключена или загружается. Запустите start-model.ps1. Обзорные кадры доступны без модели.';
  } catch {
    label.textContent = 'Нет связи с сервером. Проверьте, что он запущен.';
  }
}
refreshModelStatus();

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
  document.querySelector('#observations').replaceChildren();
  document.querySelector('#result-note').hidden = true;
  document.querySelector('#preview-result').hidden = false;
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

function showInspection(data, filename) {
  showResult({ ...data, frames: [] }, filename);
  document.querySelector('#preview-result').hidden = true;
  const note = document.querySelector('#result-note');
  note.hidden = false;
  note.textContent = data.message + ' Просмотрено кадров: ' + data.sampled_frames + '. ' + data.sampling;
  const holder = document.querySelector('#observations');
  const labels = { possible_concern: 'Возможная проблема · проверить вручную',
    observation: 'Наблюдение модели', unclear: 'Не удалось уверенно рассмотреть' };
  const categories = { cash_handling: 'Действия с наличными / похожими предметами',
    valuables_storage: 'Хранение ценностей', workspace_access: 'Рабочая зона',
    collection: 'Инкассация', other: 'Другое действие', unclear: 'Неясно' };
  const concerns = data.windows.flatMap(w => w.events).filter(e => e.status === 'possible_concern').length;
  const filterLabel = document.createElement('label'); filterLabel.className = 'filter-row';
  const filter = document.createElement('input'); filter.type = 'checkbox';
  filterLabel.append(filter, 'Показать только возможные проблемы (' + concerns + ')');
  holder.append(filterLabel);
  for (const window of data.windows) {
    const section = document.createElement('section');
    section.dataset.concerns = String(window.events.some(e => e.status === 'possible_concern'));
    const heading = document.createElement('h3'); heading.className = 'window-title';
    heading.textContent = 'Выбранные кадры: ' + window.start_sec.toFixed(1) + '–' + window.end_sec.toFixed(1) + ' сек';
    section.append(heading);
    if (window.events.length === 0) {
      const text = document.createElement('p'); text.className = 'hint';
      text.textContent = 'Модель не выделила действий на этих кадрах. Это не подтверждает отсутствие нарушений.';
      section.append(text);
    }
    for (const event of window.events) {
      const article = document.createElement('article'); article.className = 'observation';
      article.dataset.state = event.status;
      const title = document.createElement('h3'); title.textContent = labels[event.status];
      const category = document.createElement('p'); category.textContent = categories[event.category];
      const observation = document.createElement('p'); observation.textContent = event.observation;
      const reason = document.createElement('p'); reason.textContent = event.reason;
      const frames = document.createElement('div'); frames.className = 'frames';
      for (const frame of event.evidence_frames) {
        const figure = document.createElement('figure');
        const image = document.createElement('img');
        image.src = 'data:image/jpeg;base64,' + frame.image_base64;
        image.alt = 'Кадр на ' + frame.timestamp_sec.toFixed(2) + ' секунде';
        const caption = document.createElement('figcaption');
        caption.textContent = frame.timestamp_sec.toFixed(2) + ' сек · кадр для ручной проверки';
        figure.append(image, caption); frames.append(figure);
      }
      article.append(title, category, observation, reason, frames); section.append(article);
    }
    holder.append(section);
  }
  filter.addEventListener('change', () => {
    for (const section of holder.querySelectorAll('section')) {
      section.hidden = filter.checked && section.dataset.concerns !== 'true';
      for (const article of section.querySelectorAll('article')) {
        article.hidden = filter.checked && article.dataset.state !== 'possible_concern';
      }
    }
  });
  setStatus('Модель закончила просмотр за ' + data.elapsed_sec.toFixed(0)
    + ' сек. Возможных проблем: ' + concerns + '. Нужна ручная проверка.');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.mov')) return setStatus('Выберите файл .mov.', 'error');
  if (file.size > 100 * 1024 * 1024) return setStatus('Файл больше 100 МБ.', 'error');
  const key = keyInput.value.trim();
  const inspection = mode.value === 'inspect';
  const analysis = mode.value === 'analyze';
  button.disabled = true; fileInput.disabled = true; mode.disabled = true;
  result.hidden = true; empty.hidden = false;
  setStatus('Загружаем видео на локальный сервер…');
  try {
    const body = new FormData(); body.append('file', file);
    const submitted = await request(analysis ? '/api/v1/analyze' : inspection ? '/api/v1/inspect' : '/api/v1/preview', { method: 'POST', body }, key);
    setStatus(inspection ? 'Видео получено. Локальная модель рассматривает кадры. Это может занять несколько минут…'
      : 'Видео получено. Читаем параметры и обзорные кадры…');
    const deadline = Date.now() + 5 * 60 * 1000;
    while (Date.now() < deadline) {
      const job = await request(`/api/v1/jobs/${submitted.job_id}`, {}, key);
      if (job.status === 'done') {
        if (analysis) showAnalysis(job); else if (inspection) showInspection(job.result, file.name); else showResult(job.result, file.name);
        return;
      }
      if (job.progress) setStatus('Локальная модель: обработано частей ' + job.progress.completed_windows
        + ' из ' + job.progress.total_windows + '. Продолжайте ждать…');
      if (job.status === 'failed') throw new Error(job.error?.message || 'Не удалось прочитать видео.');
      await new Promise(resolve => setTimeout(resolve, 700));
    }
    throw new Error('Подготовка заняла больше пяти минут. Проверьте сервер.');
  } catch (error) {
    setStatus(error.name === 'TimeoutError' ? 'Сервер не ответил вовремя.' : error.message, 'error');
  } finally {
    button.disabled = false; fileInput.disabled = false; mode.disabled = false;
    refreshModelStatus();
  }
});

function showAnalysis(job) {
  document.querySelector('#metadata').replaceChildren();
  document.querySelector('#preview-result').hidden = true;
  const holder = document.querySelector('#observations');
  holder.replaceChildren();
  const note = document.querySelector('#result-note');
  note.hidden = false;
  note.textContent = job.result.message;
  for (const event of job.violations) {
    const article = document.createElement('article'); article.className = 'observation';
    const heading = document.createElement('h3');
    heading.textContent = event.type + ' · ' + event.timecode_sec.toFixed(2) + ' сек';
    const description = document.createElement('p'); description.textContent = event.description;
    const source = document.createElement('a'); source.href = event.normative_source.url;
    source.target = '_blank'; source.rel = 'noopener noreferrer';
    source.textContent = event.normative_source.document + ' · ' + event.normative_source.section;
    const context = document.createElement('p'); context.textContent = event.normative_source.note;
    const frames = document.createElement('div'); frames.className = 'frames';
    for (const frame of event.evidence_frames) {
      const figure = document.createElement('figure'); const image = document.createElement('img');
      image.src = 'data:image/jpeg;base64,' + frame.image_base64;
      image.alt = 'Кадр для проверки на ' + frame.timestamp_sec + ' секунде';
      const caption = document.createElement('figcaption'); caption.textContent = frame.timestamp_sec.toFixed(2) + ' сек';
      figure.append(image, caption); frames.append(figure);
    }
    article.append(heading, description, source, context, frames); holder.append(article);
  }
  empty.hidden = true; result.hidden = false;
  setStatus(job.verdict === 'violation' ? 'Обнаружены признаки возможных нарушений' : 'Нарушение не обнаружено', 'done');
}
