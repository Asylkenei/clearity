const byId = id => document.getElementById(id);
const video = byId('review-video');
let accessKey = '';
let catalog = null;
let categories = [];
let selected = null;
let videoURL = null;
let dirty = false;

const stateNames = { not_started: 'Не просмотрено', draft: 'Черновик', reviewed: 'Просмотрено', needs_review: 'Нужно уточнить' };

function statusMessage(text, state = '') {
  byId('review-status').textContent = text;
  byId('review-status').dataset.state = state;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options, headers: { 'X-API-Key': accessKey, ...(options.headers || {}) },
    signal: AbortSignal.timeout(60000)
  });
  if (!response.ok) {
    let error = {};
    try { error = await response.json(); } catch {}
    const detail = error.detail;
    const text = typeof detail === 'string' ? detail
      : Array.isArray(detail) ? detail.map(item => item.msg).join('; ')
      : detail?.message || 'Запрос не выполнен. Проверьте сервер и ключ.';
    throw new Error(text);
  }
  return response;
}

function labelText(record) {
  return record.label === 'violation' ? 'По разметке: нарушение' : 'По разметке: без нарушения';
}

function renderList() {
  const filter = byId('record-filter').value;
  const list = byId('record-list');
  list.replaceChildren();
  for (const record of catalog.records) {
    if (['violation', 'no_violation'].includes(filter) && record.label !== filter) continue;
    if (filter === 'unfinished' && record.annotation.review_state === 'reviewed') continue;
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'record-button';
    button.setAttribute('aria-current', String(record.video_id === selected?.video_id));
    const title = document.createElement('span'); title.textContent = record.filename;
    const subtitle = document.createElement('span'); subtitle.className = 'record-subtitle';
    const time = record.label_timecode_sec === null ? 'без отметки' : `${record.label_timecode_sec} сек`;
    subtitle.textContent = `${record.label === 'violation' ? 'Нарушение' : 'Без нарушения'} · ${time} · ${stateNames[record.annotation.review_state]}`;
    button.append(title, subtitle);
    button.addEventListener('click', () => selectRecord(record));
    list.append(button);
  }
  byId('dataset-summary').textContent = `${catalog.summary.videos} записей · ${catalog.summary.violation} с нарушением · ${catalog.summary.no_violation} без`;
  const reviewed = catalog.records.filter(r => r.annotation.review_state === 'reviewed').length;
  byId('review-progress').textContent = `Просмотрено: ${reviewed} из ${catalog.records.length}`;
}

function renderCategories(value = '') {
  const select = byId('category'); select.replaceChildren();
  select.add(new Option('Пока не определена', ''));
  for (const category of categories) select.add(new Option(category.name, category.id));
  select.value = value || '';
  describeCategory();
}

function describeCategory() {
  const category = categories.find(c => c.id === byId('category').value);
  byId('category-definition').textContent = category
    ? `${category.definition || 'Определение пока не заполнено.'} Нормативное основание не проверено.`
    : 'Если неясно, оставьте категорию пустой и выберите статус «Нужно уточнить».';
}

function resetPlayer() {
  video.pause(); video.removeAttribute('src'); video.load(); video.style.display = 'none';
  byId('video-placeholder').hidden = false;
  byId('current-time').textContent = '0.00 сек';
  if (videoURL) URL.revokeObjectURL(videoURL);
  videoURL = null;
  for (const id of ['jump-context', 'jump-mark', 'back-second', 'forward-second', 'set-start', 'set-end']) byId(id).disabled = true;
}

function selectRecord(record) {
  if (dirty && !window.confirm('Есть несохранённые наблюдения. Перейти к другой записи без сохранения?')) return;
  selected = record; dirty = false; resetPlayer();
  byId('selected-title').textContent = record.filename;
  byId('source-label').textContent = `${labelText(record)}${record.label_timecode_sec === null ? '. Отметки времени нет.' : ` · отметка ${record.label_timecode_sec} сек.`}`;
  const twins = record.same_content_ids;
  byId('duplicate-note').hidden = !twins.length;
  if (twins.length) {
    const descriptions = twins.map(id => {
      const twin = catalog.records.find(r => r.video_id === id);
      return `${twin.filename}${twin.label_timecode_sec === null ? '' : ` — ${twin.label_timecode_sec} сек`}`;
    });
    byId('duplicate-note').textContent = `Те же байты видео: ${descriptions.join(', ')}. Исходные отметки сохраняем отдельно. Проверьте, относятся ли они к одному действию.`;
  }
  const annotation = record.annotation;
  byId('observation').value = annotation.observation || '';
  byId('notes').value = annotation.notes || '';
  byId('visibility').value = annotation.visibility || 'clear';
  byId('review-state').value = annotation.review_state === 'not_started' ? 'draft' : annotation.review_state;
  byId('event-start').value = annotation.event_start_sec ?? '';
  byId('event-end').value = annotation.event_end_sec ?? '';
  renderCategories(annotation.category_id);
  renderList();
  statusMessage(`Открыта разметка ${record.filename}. Видео загрузится после нажатия «Открыть видео».`);
}

byId('unlock-form').addEventListener('submit', async event => {
  event.preventDefault(); accessKey = byId('review-key').value.trim();
  try {
    const [datasetResponse, categoriesResponse] = await Promise.all([
      api('/api/v1/review/dataset'), api('/api/v1/review/categories')
    ]);
    catalog = await datasetResponse.json(); categories = await categoriesResponse.json();
    if (!catalog.records.length) throw new Error(catalog.warnings.join(' ') || 'Каталог пуст.');
    byId('unlock-panel').hidden = true; byId('review-workspace').hidden = false;
    selectRecord(catalog.records[0]);
  } catch (error) { statusMessage(error.message, 'error'); }
});

byId('record-filter').addEventListener('change', renderList);
byId('category').addEventListener('change', describeCategory);
byId('annotation-form').addEventListener('input', () => { dirty = true; });

byId('load-video').addEventListener('click', async () => {
  if (!selected) return;
  const id = selected.video_id;
  byId('load-video').disabled = true;
  statusMessage('Читаем видео с локального диска…');
  try {
    const response = await api(`/api/v1/review/videos/${encodeURIComponent(id)}`);
    const blob = await response.blob();
    if (selected.video_id !== id) return;
    resetPlayer(); videoURL = URL.createObjectURL(blob);
    video.src = videoURL; video.style.display = 'block'; byId('video-placeholder').hidden = true;
  } catch (error) { statusMessage(error.message, 'error'); }
  finally { byId('load-video').disabled = false; }
});

video.addEventListener('loadedmetadata', () => {
  if (!selected) return;
  video.playbackRate = Number(byId('playback-speed').value);
  video.currentTime = Math.max(0, (selected.label_timecode_sec || 0) - 3);
  for (const id of ['back-second', 'forward-second', 'set-start', 'set-end']) byId(id).disabled = false;
  for (const id of ['jump-context', 'jump-mark']) byId(id).disabled = selected.label_timecode_sec === null;
  statusMessage('Видео готово. Нажмите ▶ в плеере и опишите видимое действие.');
});
video.addEventListener('error', () => {
  if (video.getAttribute('src')) statusMessage('Браузер не смог прочитать видео. Откройте этот MOV локальным проигрывателем; наблюдение можно заполнить здесь.', 'error');
});
video.addEventListener('timeupdate', () => { byId('current-time').textContent = `${video.currentTime.toFixed(2)} сек`; });
byId('playback-speed').addEventListener('change', event => { video.playbackRate = Number(event.target.value); });

function seek(seconds) { video.pause(); video.currentTime = Math.max(0, Math.min(video.duration || 0, seconds)); }
byId('jump-context').addEventListener('click', () => seek(selected.label_timecode_sec - 3));
byId('jump-mark').addEventListener('click', () => seek(selected.label_timecode_sec));
byId('back-second').addEventListener('click', () => seek(video.currentTime - 1));
byId('forward-second').addEventListener('click', () => seek(video.currentTime + 1));
for (const side of ['start', 'end']) byId(`set-${side}`).addEventListener('click', () => {
  byId(`event-${side}`).value = video.currentTime.toFixed(2); dirty = true;
});

byId('annotation-form').addEventListener('submit', async event => {
  event.preventDefault(); if (!selected) return;
  const record = selected;
  const numberOrNull = id => byId(id).value === '' ? null : Number(byId(id).value);
  const body = {
    observation: byId('observation').value.trim(), category_id: byId('category').value || null,
    visibility: byId('visibility').value, review_state: byId('review-state').value,
    event_start_sec: numberOrNull('event-start'), event_end_sec: numberOrNull('event-end'),
    notes: byId('notes').value.trim()
  };
  byId('save-annotation').disabled = true;
  try {
    const response = await api(`/api/v1/review/annotations/${encodeURIComponent(record.video_id)}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    record.annotation = await response.json();
    if (selected.video_id === record.video_id) dirty = false;
    renderList(); statusMessage(`Наблюдение для ${record.filename} сохранено на этом компьютере.`, 'done');
  } catch (error) { statusMessage(error.message, 'error'); }
  finally { byId('save-annotation').disabled = false; }
});

byId('category-form').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    const response = await api('/api/v1/review/categories', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: byId('new-category-name').value, definition: byId('new-category-definition').value })
    });
    const category = await response.json(); categories.push(category);
    renderCategories(category.id); dirty = true; byId('category-form').reset();
    statusMessage('Рабочая категория добавлена. Нажмите «Сохранить наблюдение», чтобы связать её с записью.');
  } catch (error) { statusMessage(error.message, 'error'); }
});

byId('export-button').addEventListener('click', async () => {
  if (dirty) { statusMessage('Сначала сохраните наблюдение текущей записи.', 'error'); return; }
  try {
    const response = await api('/api/v1/review/export');
    const data = await response.json();
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'forte-observations.json';
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    statusMessage('Экспорт подготовлен. В нём сохранены исходные метки и ваши отдельные наблюдения.', 'done');
  } catch (error) { statusMessage(error.message, 'error'); }
});
window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
