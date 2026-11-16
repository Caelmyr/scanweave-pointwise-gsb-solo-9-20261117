'use strict';
/* ScanWeave scratch restoration studio.
   Marks are drawn on the composite preview, saved into the project document and
   stay editable (and undoable) after reload. Repairs are stored as separate
   result records; the original scan and composite are never overwritten. */

const SCRATCH = {
  open: false, canvas: null, context: null, base: null, baseSize: null,
  states: {marked: true, repaired: false}, tool: 'paint',
  draft: [], drawing: false, busy: false, report: null, view: 'marked',
  restorations: [], confirmed: null, fractions: .5,
};

function scratchActive(){ return SCRATCH.open && SCRATCH.canvas && S.project; }

/* Called from app.js: viewer tab button toggles between composite and marks. */
async function toggleScratchStudio(){
  if(SCRATCH.open){ SCRATCH.open = false; closeDialog(); await refreshHero(); return; }
  if(S.project?.current_result) await refreshCurrent();
  await openScratchStudio();
}

function scratchMarkup(){
  return `<div class="scratch-studio">
    <div class="scratch-head">
      <div class="scratch-tools">
        <button data-scratch-tool="paint" class="active">🖌 标记划痕</button>
        <button data-scratch-tool="erase">◌ 擦掉误画</button>
        <button id="scratchUndoStroke">↶ 撤销一笔</button>
        <button id="scratchClear">清除全部标记</button>
      </div>
      <div class="scratch-views">
        <button data-scratch-view="marked" class="active">显示标记</button>
        <button data-scratch-view="original">原扫描</button>
        <button data-scratch-view="repaired">修复结果</button>
        <button data-scratch-view="split">左右对比</button>
        <button data-scratch-view="difference">差异</button>
      </div>
    </div>
    <div class="scratch-stage">
      <canvas id="scratchCanvas" aria-label="划痕标记画布"></canvas>
      <div class="scratch-hint" id="scratchHint">在图上涂出要修的划痕；可切换擦除工具去掉误画。</div>
    </div>
    <div class="scratch-controls">
      <label class="field">笔刷大小<input id="scratchRadius" type="range" min="0.15" max="8" step="0.05" value="0.4"><span id="scratchRadiusLabel">0.4%</span></label>
      <label class="field">修复范围外扩<input id="scratchExpand" type="range" min="0" max="2.5" step="0.05" value="0.15"><span id="scratchExpandLabel">0.15%</span></label>
      <label class="field">边缘融合<input id="scratchFeather" type="range" min="0" max="2" step="0.05" value="0.3"><span id="scratchFeatherLabel">0.3%</span></label>
      <label class="field">修复方式<select id="scratchMethod"><option value="texture">纹理接续（保留纸纹/细字）</option><option value="smooth">平滑补全</option></select></label>
      <label class="field scratch-split">对比位置<input id="scratchFraction" type="range" min="0" max="1" step="0.01" value="0.5"></label>
    </div>
    <div class="scratch-foot">
      <div id="scratchStatus" class="scratch-status">尚未标记。</div>
      <div class="row">
        <button id="scratchPreview">↻ 更新预览</button>
        <button id="scratchConfirm" class="primary">✓ 确认修复结果</button>
        <button id="scratchExport">↗ 导出修复图</button>
        <button id="scratchHistory">修复记录</button>
      </div>
    </div>
    <div id="scratchRecords" class="scratch-records"></div>
  </div>`;
}

/* Reset the studio when the dialog is dismissed so a stale canvas is never
   reused after switching projects. */
$('dialog').addEventListener('close', () => {
  SCRATCH.open = false;
  SCRATCH.canvas = null;
  SCRATCH.base = null;
  SCRATCH.draft = [];
});

async function openScratchStudio(){
  const p = project();
  if(!p.current_result) throw Error('请先完成合成，再标记划痕');
  const data = await api(path('/scratch-state'));
  SCRATCH.open = true;
  SCRATCH.canvas = null;
  SCRATCH.base = null;
  openDialog('划痕修复 · ' + p.name, scratchMarkup());
  SCRATCH.restorations = data.restorations || [];
  SCRATCH.confirmed = data.confirmed;
  SCRATCH.settings = data.settings;
  SCRATCH.state = data;
  SCRATCH.report = null;
  SCRATCH.view = 'marked';
  bindScratchControls(data);
  await scratchLoad();
  renderScratchRecords();
  scratchStatus();
}

function bindScratchControls(data){
  const settings = data.settings;
  const radius = $('scratchRadius'); radius.value = String(0.4);
  $('scratchRadiusLabel').textContent = '0.4%';
  radius.oninput = () => { $('scratchRadiusLabel').textContent = Number(radius.value).toFixed(1) + '%'; };
  const expand = $('scratchExpand'); expand.value = String(settings.expand * 100);
  $('scratchExpandLabel').textContent = (settings.expand * 100).toFixed(1) + '%';
  expand.oninput = () => { $('scratchExpandLabel').textContent = Number(expand.value).toFixed(1) + '%'; };
  expand.onchange = () => saveScratchSettingsSoon();
  const feather = $('scratchFeather'); feather.value = String(settings.feather * 100);
  $('scratchFeatherLabel').textContent = (settings.feather * 100).toFixed(1) + '%';
  feather.oninput = () => { $('scratchFeatherLabel').textContent = Number(feather.value).toFixed(1) + '%'; };
  feather.onchange = () => saveScratchSettingsSoon();
  $('scratchMethod').value = settings.method;
  $('scratchMethod').onchange = () => saveScratchSettingsSoon();
  const fraction = $('scratchFraction');
  fraction.value = String(SCRATCH.fractions);
  fraction.oninput = safe(async () => {
    SCRATCH.fractions = Number(fraction.value);
    if(SCRATCH.view === 'split') await scratchRefreshView();
  });
  document.querySelectorAll('[data-scratch-tool]').forEach(el => {
    el.onclick = () => {
      SCRATCH.tool = el.dataset.scratchTool;
      document.querySelectorAll('[data-scratch-tool]').forEach(b => b.classList.toggle('active', b === el));
    };
  });
  document.querySelectorAll('[data-scratch-view]').forEach(el => {
    el.onclick = safe(async () => {
      SCRATCH.view = el.dataset.scratchView;
      document.querySelectorAll('[data-scratch-view]').forEach(b => b.classList.toggle('active', b === el));
      $('scratchFraction').closest('.field').style.display = SCRATCH.view === 'split' ? '' : 'none';
      await scratchRefreshView();
    });
  });
  $('scratchFraction').closest('.field').style.display = 'none';
  bind('scratchUndoStroke', async () => {
    const state = SCRATCH.state;
    if(!state.strokes.length) throw Error('没有可撤销的标记');
    const last = state.strokes[state.strokes.length - 1];
    await mutate('/scratch-marks/' + last.id + '/remove', 'POST', {});
    await scratchReload();
  });
  bind('scratchClear', async () => {
    if(!confirm('清除全部划痕标记？修复范围会一起清空。')) return;
    await mutate('/scratch-clear', 'POST', {});
    await scratchReload();
  });
  bind('scratchPreview', () => scratchReload());
  bind('scratchConfirm', () => scratchConfirm());
  bind('scratchExport', () => scratchExportDialog());
  bind('scratchHistory', () => renderScratchRecords(true));
}

async function scratchSaveSettings(){
  const body = {
    expand: Number($('scratchExpand').value) / 100,
    feather: Number($('scratchFeather').value) / 100,
    method: $('scratchMethod').value,
  };
  await mutate('/scratch-settings', 'PATCH', {settings: body});
  await scratchReload();
}

/* Sliders fire while another save may still be in flight; coalesce them into a
   single save, and retry briefly instead of colliding with the revision lock. */
const saveScratchSettingsNow = safe(async (attempt = 0) => {
  if(S.busy){
    if(attempt > 20) return;
    setTimeout(() => saveScratchSettingsNow(attempt + 1), 200);
    return;
  }
  await scratchSaveSettings();
});

function saveScratchSettingsSoon(){
  clearTimeout(SCRATCH.settingsTimer);
  SCRATCH.settingsTimer = setTimeout(() => saveScratchSettingsNow(), 350);
}

async function scratchLoad(){
  const key = project().id;
  const blob = await api('/projects/' + key + '/scratch-preview?view=original');
  SCRATCH.base = await createImageBitmap(blob);
  SCRATCH.baseSize = {width: SCRATCH.base.width, height: SCRATCH.base.height};
  const canvas = $('scratchCanvas');
  canvas.width = SCRATCH.base.width;
  canvas.height = SCRATCH.base.height;
  SCRATCH.context = canvas.getContext('2d');
  const stage = canvas.parentElement;
  stage.style.aspectRatio = SCRATCH.base.width + ' / ' + SCRATCH.base.height;
  bindScratchCanvas();
  await scratchRefreshView();
}

function scratchPoint(event){
  const canvas = SCRATCH.canvas;
  const rect = canvas.getBoundingClientRect();
  return [
    Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  ];
}

function scratchDrawDraft(){
  const ctx = SCRATCH.context, canvas = SCRATCH.canvas;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(SCRATCH.base, 0, 0);
  const scale = Math.min(canvas.width, canvas.height);
  // Marks are composited on their own layer, so an eraser stroke removes
  // earlier marks instead of punching a hole through the scan.
  const layer = document.createElement('canvas');
  layer.width = canvas.width;
  layer.height = canvas.height;
  const paint = layer.getContext('2d');
  if(!paint) return;
  for(const stroke of [...(SCRATCH.state?.strokes || []), ...SCRATCH.draft]){
    const points = stroke.points.map(p => [p[0] * (canvas.width - 1), p[1] * (canvas.height - 1)]);
    paint.save();
    paint.globalCompositeOperation = stroke.mode === 'erase' ? 'destination-out' : 'source-over';
    paint.strokeStyle = 'rgba(233,56,36,1)';
    paint.fillStyle = 'rgba(233,56,36,1)';
    paint.lineWidth = Math.max(1, stroke.radius * scale * 2);
    paint.lineCap = 'round';
    paint.lineJoin = 'round';
    if(points.length > 1){
      paint.beginPath();
      paint.moveTo(points[0][0], points[0][1]);
      for(const point of points.slice(1)) paint.lineTo(point[0], point[1]);
      paint.stroke();
    }
    const last = points[points.length - 1];
    paint.beginPath();
    paint.arc(last[0], last[1], Math.max(.6, stroke.radius * scale), 0, Math.PI * 2);
    paint.fill();
    paint.restore();
  }
  ctx.save();
  ctx.globalAlpha = .55;
  ctx.drawImage(layer, 0, 0);
  ctx.restore();
}

function bindScratchCanvas(){
  const canvas = $('scratchCanvas');
  SCRATCH.canvas = canvas;
  SCRATCH.context = canvas.getContext('2d');
  canvas.onpointerdown = event => {
    if(SCRATCH.busy) return;
    SCRATCH.drawing = true;
    SCRATCH.draft = [{
      id: 'draft',
      mode: SCRATCH.tool,
      radius: Number($('scratchRadius').value) / 100,
      points: [scratchPoint(event)],
    }];
    canvas.setPointerCapture(event.pointerId);
    scratchDrawDraft();
  };
  canvas.onpointermove = event => {
    if(!SCRATCH.drawing) return;
    const point = scratchPoint(event);
    const stroke = SCRATCH.draft[0];
    const last = stroke.points[stroke.points.length - 1];
    if(Math.hypot(point[0] - last[0], point[1] - last[1]) < .002) return;
    stroke.points.push(point);
    scratchDrawDraft();
  };
  canvas.onpointerup = safe(async () => {
    if(!SCRATCH.drawing) return;
    SCRATCH.drawing = false;
    if(!SCRATCH.draft.length) return;
    const stroke = SCRATCH.draft[0];
    SCRATCH.draft = [];
    if(stroke.points.length < 1) { await scratchRefreshView(); return; }
    await scratchCommitStroke(stroke);
  });
  canvas.onpointercancel = safe(async () => {
    SCRATCH.drawing = false;
    SCRATCH.draft = [];
    await scratchRefreshView();
  });
}

async function scratchCommitStroke(stroke){
  if(SCRATCH.busy) return;
  SCRATCH.busy = true;
  scratchBusy(true, '正在保存标记…');
  try{
    await mutate('/scratch-marks', 'POST', {strokes: [{
      mode: stroke.mode, radius: stroke.radius, points: stroke.points,
    }]});
    await scratchReload();
  } finally {
    SCRATCH.busy = false;
    scratchBusy(false);
  }
}

async function scratchRefreshView(){
  if(!SCRATCH.open || !SCRATCH.base) return;
  if(SCRATCH.view === 'marked' || SCRATCH.view === 'original'){
    scratchDrawDraft();
    if(SCRATCH.view === 'original') SCRATCH.context.drawImage(SCRATCH.base, 0, 0);
    return;
  }  const query = '?view=' + SCRATCH.view + '&fraction=' + SCRATCH.fractions;
  const url = '/api/projects/' + project().id + '/scratch-preview' + query;
  const response = await fetch(url);
  if(!response.ok){ let message = '预览生成失败'; try{ message = (await response.json()).error; }catch{} throw Error(message); }
  const report = response.headers.get('X-Scratch-Report');
  if(report) SCRATCH.report = JSON.parse(report);
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);
  SCRATCH.context.drawImage(bitmap, 0, 0, SCRATCH.canvas.width, SCRATCH.canvas.height);
  bitmap.close();
  scratchStatus();
}

async function scratchReload(){
  const data = await api(path('/scratch-state'));
  SCRATCH.state = data;
  SCRATCH.settings = data.settings;
  SCRATCH.restorations = data.restorations || [];
  SCRATCH.confirmed = data.confirmed;
  renderScratchRecords();
  await scratchRefreshView();
  scratchStatus();
}

async function scratchConfirm(){
  if(!SCRATCH.state?.strokes.length) throw Error('请先在图上标记需要修复的划痕');
  const note = prompt('给这次修复留一句备注（可留空）：', '') ?? '';
  SCRATCH.busy = true;
  scratchBusy(true, '正在生成修复结果…');
  try{
    const output = await mutate('/scratch-restore', 'POST', {note});
    SCRATCH.confirmed = output.restoration.id;
    SCRATCH.restorations = [output.restoration, ...SCRATCH.restorations.filter(r => r.id !== output.restoration.id)];
    await scratchReload();
    toast('修复结果已保存；原始扫描与合成结果保持不变');
  } finally {
    SCRATCH.busy = false;
    scratchBusy(false);
  }
}

async function scratchExportDialog(){
  if(!SCRATCH.confirmed){
    const current = SCRATCH.restorations.find(r => r.current) || SCRATCH.restorations[0];
    if(!current) throw Error('请先确认一次修复结果');
    SCRATCH.confirmed = current.id;
  }
  openDialog('导出修复结果', `<form id="scratchExportForm"><div class="form-grid">
    ${field('格式','scratchFormat','png',[['png','PNG 无损'],['jpeg','JPEG'],['tiff','16 位 TIFF']])}
    ${field('PNG 位深','scratchBits',8,[8,16])}
  </div><p class="muted">导出的是当前确认的修复结果；原始扫描与合成结果不会被覆盖，随时可以继续修改标记后重新导出。</p>
  <div class="dialog-actions"><button class="primary">下载</button></div></form>`);
  $('scratchExportForm').onsubmit = safe(async event => {
    event.preventDefault();
    const format = $('scratchFormat').value;
    const blob = await api(path('/scratch-restorations/' + SCRATCH.confirmed + '/export'), 'POST',
      {format, bits: Number($('scratchBits').value)});
    downloadBlob(blob, format === 'jpeg' ? 'scratch-restored.jpg' : format === 'tiff' ? 'scratch-restored-16bit.tiff' : 'scratch-restored.png');
    toast('修复图已下载');
  });
}

function scratchBusy(flag, message){
  document.querySelectorAll('.scratch-studio button').forEach(button => { button.disabled = !!flag; });
  const status = $('scratchStatus');
  if(status && message) status.textContent = message;
}

function scratchStatus(){
  const status = $('scratchStatus');
  if(!status) return;
  const state = SCRATCH.state || {strokes: [], settings: {}};
  const report = SCRATCH.report;
  const parts = [`标记 ${state.strokes?.length || 0} 笔`];
  if(report && report.marked_pixels !== undefined){
    parts.push(`覆盖 ${report.marked_pixels} 像素`);
    parts.push(report.method === 'texture' ? '纹理接续' : report.method === 'smooth' ? '平滑补全' : '未修复');
    if(report.outside_changed !== undefined) parts.push(`标记外改动 ${report.outside_changed} 像素`);
  }
  parts.push(SCRATCH.confirmed ? '已保存修复结果' : '尚未确认修复');
  status.textContent = parts.join(' · ');
}

function scratchTime(value){
  const date = new Date(value);
  if(Number.isNaN(date.getTime())) return value;
  const pad = n => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function renderScratchRecords(expand){
  const box = $('scratchRecords');
  if(!box) return;
  const rows = SCRATCH.restorations || [];
  box.innerHTML = `<div class="section-title">修复记录 ${rows.length ? '(' + rows.length + ')' : ''}</div>` +
    (rows.length ? rows.map(row => `<div class="job-entry ${row.current ? 'scratch-current' : ''}">
      <div class="row"><span>${esc(scratchTime(row.created_at))} · ${row.width}×${row.height} · ${row.strokes} 笔 · ${row.mask_pixels} 像素</span>
      <span>${row.current ? '当前修复' : '历史'}${row.matches_marks ? '' : ' · 标记已变'}${row.base_stale ? ' · 合成结果已更新' : ''}</span></div>
      ${row.note ? `<div class="muted">${esc(row.note)}</div>` : ''}
      <div class="row">
        <button data-scratch-record-view="${row.id}">查看</button>
        <button data-scratch-record-export="${row.id}">导出</button>
        <button data-scratch-record-remove="${row.id}">删除</button>
      </div></div>`).join('') : '<p class="muted">确认修复后会在这里留下记录；每次都是单独的结果，不覆盖原扫描。</p>');
  box.querySelectorAll('[data-scratch-record-view]').forEach(el => el.onclick = safe(async () => {
    const blob = await api(path('/scratch-restorations/' + el.dataset.scratchRecordView + '/image'));
    if(S.dialogUrl) URL.revokeObjectURL(S.dialogUrl);
    S.dialogUrl = URL.createObjectURL(blob);
    SCRATCH.open = false;
    $('dialogTitle').textContent = '已保存的修复结果';
    $('dialogBody').innerHTML = `<img class="dialog-image" src="${S.dialogUrl}" alt="已保存的修复结果"><div class="dialog-actions"><button id="scratchBack">返回修复工作室</button></div>`;
    bind('scratchBack', () => openScratchStudio());
  }));
  box.querySelectorAll('[data-scratch-record-export]').forEach(el => el.onclick = safe(async () => {
    SCRATCH.confirmed = el.dataset.scratchRecordExport;
    await scratchExportDialog();
  }));
  box.querySelectorAll('[data-scratch-record-remove]').forEach(el => el.onclick = safe(async () => {
    if(!confirm('删除这条修复记录？原扫描和标记都会保留，可撤销。')) return;
    await mutate('/scratch-restorations/' + el.dataset.scratchRecordRemove + '/remove', 'POST', {});
    await scratchReload();
  }));
  if(expand) box.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}
