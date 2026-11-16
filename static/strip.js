'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const BASE = '/api/strip-sheets';
const S = { sheets: [], sheet: null, selected: null, zoom: 1, view: 'layout', busy: false, previewUrl: null };
let toastTimer;

function toast(msg) { $('toast').textContent = msg; $('toast').classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').classList.remove('show'), 4600); }
async function api(path, method = 'GET', data) {
  const opt = { method, headers: {} };
  if (data !== undefined) { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(data); }
  const r = await fetch(BASE + path, opt);
  if (!r.ok) { let m; try { m = (await r.json()).error; } catch { m = '请求失败 ' + r.status; } throw Error(m); }
  return r.headers.get('Content-Type')?.includes('application/json') ? r.json() : r.blob();
}
function safe(fn) { return async (...a) => { try { await fn(...a); } catch (e) { toast(e.message); console.error(e); } }; }
function bind(id, fn) { $(id).onclick = safe(fn); }
function rev(extra = {}) { return { revision: S.sheet.revision, ...extra }; }
function openDialog(title, html) { $('dialogTitle').textContent = title; $('dialogBody').innerHTML = html; if (!$('dialog').open) $('dialog').showModal(); }
function closeDialog() { $('dialog').close(); }

// ---------------- 布局列表 ----------------
async function refreshSheets() {
  S.sheets = await api('?query=' + encodeURIComponent($('sheetSearch').value));
  $('sheetCount').textContent = S.sheets.length;
  $('sheetList').innerHTML = S.sheets.map(s => {
    const cls = s.confirmed ? 'ok' : s.open_issues ? 'err' : s.strip_count ? 'warn' : 'idle';
    return `<div class="sheet-entry ${s.id === S.sheet?.id ? 'active' : ''}" data-id="${s.id}">
      <strong><span class="dot ${cls}"></span>${esc(s.name)}</strong>
      <small>${s.strip_count} 条带 · ${s.confirmed ? '已确认' : s.open_issues ? s.open_issues + ' 疑点待处理' : '未确认'} · REV ${s.revision}</small></div>`;
  }).join('') || '<div class="empty-note">还没有布局。</div>';
  $('sheetList').querySelectorAll('[data-id]').forEach(el => el.onclick = safe(() => loadSheet(el.dataset.id)));
}

async function loadSheet(id) {
  S.sheet = await api('/' + id);
  S.selected = S.sheet.strips[0]?.id || null;
  $('emptyApp').hidden = true; $('workbench').hidden = false;
  $('axisSelect').value = S.sheet.axis;
  render(); await refreshSheets();
}

async function mutate(path, method, data) {
  if (S.busy) throw Error('正在保存，请稍候');
  S.busy = true;
  try { S.sheet = await api(path, method, data); render(); await refreshSheets(); return S.sheet; }
  finally { S.busy = false; }
}

// ---------------- 渲染 ----------------
function render() {
  const s = S.sheet; if (!s) return;
  $('sheetTitle').textContent = s.name;
  $('sheetNotes').textContent = s.notes || '导入有重叠的条带，按内容建议排列；疑点由你确认。';
  $('stripCount').textContent = s.strips.length + ' STRIPS';
  const openIssues = s.issues.filter(i => !i.resolved);
  const errors = openIssues.filter(i => i.severity === 'error').length;
  $('issueBadge').textContent = openIssues.length;
  $('edgeBadge').textContent = s.edges.length;
  const cs = $('confirmState');
  if (s.confirmed) { cs.textContent = '● 已确认，可导出'; cs.className = 'state ok'; }
  else if (errors) { cs.textContent = '● ' + errors + ' 个错误疑点'; cs.className = 'state err'; }
  else if (openIssues.length) { cs.textContent = '● ' + openIssues.length + ' 个待确认疑点'; cs.className = 'state warn'; }
  else { cs.textContent = '○ 未确认'; cs.className = 'state'; }
  $('undoBtn').disabled = !s.undo.length; $('redoBtn').disabled = !s.redo.length;
  renderCanvas(); renderStripList(); renderIssues(); renderEdges(); renderSelection();
}

function flaggedIds() {
  const set = new Set();
  S.sheet.issues.filter(i => !i.resolved).forEach(i => i.strips.forEach(x => set.add(x)));
  return set;
}

function renderCanvas() {
  const s = S.sheet; const wrap = $('stripCanvas');
  if (!s.strips.length) { wrap.innerHTML = '<div class="empty-note" style="position:absolute;top:40px;left:0;right:0">导入条带后开始。</div>'; wrap.style.width = '300px'; wrap.style.height = '200px'; return; }
  const dims = s.strips.map(t => {
    const swapped = t.rotation === 90 || t.rotation === 270;
    return [swapped ? t.height : t.width, swapped ? t.width : t.height];
  });
  const maxW = Math.max(...s.strips.map((t, k) => t.position[0] + dims[k][0]));
  const maxH = Math.max(...s.strips.map((t, k) => t.position[1] + dims[k][1]));
  wrap.style.width = Math.ceil(maxW) + 'px'; wrap.style.height = Math.ceil(maxH) + 'px';
  wrap.style.transform = 'scale(' + S.zoom + ')';
  const flagged = flaggedIds();
  wrap.innerHTML = s.strips.map((t, i) => {
    const swapped = t.rotation === 90 || t.rotation === 270;
    const bw = swapped ? t.height : t.width;
    const bh = swapped ? t.width : t.height;
    return `
    <div class="strip-node ${t.id === S.selected ? 'selected' : ''} ${flagged.has(t.id) ? 'flagged' : ''}"
      data-id="${t.id}" style="left:${t.position[0]}px;top:${t.position[1]}px;width:${bw}px;height:${bh}px;${t.enabled ? '' : 'opacity:.35'}">
      <img src="/api/strip-sheets/${s.id}/strips/${t.id}/thumbnail?rev=${s.revision}" alt="${esc(t.name)}" draggable="false">
      <span class="tag">${String(i + 1).padStart(2, '0')} · ${esc(t.name)}</span>
      ${t.manual ? '<span class="manual-mark">手动</span>' : ''}
    </div>`;
  }).join('');
  wrap.querySelectorAll('.strip-node').forEach(node => attachDrag(node));
}

function attachDrag(node) {
  const id = node.dataset.id;
  node.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    S.selected = id; renderStripList(); renderSelection();
    const t = S.sheet.strips.find(x => x.id === id);
    const startX = e.clientX, startY = e.clientY;
    const origX = t.position[0], origY = t.position[1];
    let moved = false;
    node.setPointerCapture(e.pointerId); node.classList.add('dragging');
    const move = ev => {
      const dx = (ev.clientX - startX) / S.zoom, dy = (ev.clientY - startY) / S.zoom;
      if (Math.abs(ev.clientX - startX) + Math.abs(ev.clientY - startY) > 3) moved = true;
      t.position = [Math.round((origX + dx) * 10) / 10, Math.round((origY + dy) * 10) / 10];
      node.style.left = t.position[0] + 'px'; node.style.top = t.position[1] + 'px';
    };
    const up = safe(async () => {
      node.removeEventListener('pointermove', move); node.classList.remove('dragging');
      if (moved && (t.position[0] !== origX || t.position[1] !== origY)) {
        await mutate('/' + S.sheet.id + '/strips/' + id, 'PATCH', rev({ position: t.position }));
      }
    });
    node.addEventListener('pointermove', move);
    node.addEventListener('pointerup', up, { once: true });
  });
  // 双击旋转：事件委托，只绑定一次
  node.addEventListener('dblclick', safe(async ev => {
    ev.preventDefault();
    const cur = S.sheet.strips.find(x => x.id === id);
    await mutate('/' + S.sheet.id + '/strips/' + id, 'PATCH', rev({ rotation: (cur.rotation + 90) % 360 }));
    toast('已旋转（仅摆放，不改原素材）');
  }));
}

function renderStripList() {
  const s = S.sheet; const flagged = flaggedIds();
  $('stripList').innerHTML = s.strips.map((t, i) => `
    <div class="strip-item ${t.id === S.selected ? 'selected' : ''} ${flagged.has(t.id) ? 'flagged' : ''}" data-id="${t.id}">
      <span class="strip-order">${String(i + 1).padStart(2, '0')}</span>
      <img src="/api/strip-sheets/${s.id}/strips/${t.id}/thumbnail?rev=${s.revision}" alt="">
      <div class="strip-meta"><strong>${esc(t.name)}</strong><small>${t.width}×${t.height} · (${t.position[0]}, ${t.position[1]})${t.manual ? ' · 手动' : ''}</small></div>
    </div>`).join('') || '<div class="empty-note">尚未导入条带。</div>';
  $('stripList').querySelectorAll('[data-id]').forEach(el => el.onclick = () => { S.selected = el.dataset.id; renderCanvas(); renderStripList(); renderSelection(); });
}

function renderSelection() {
  const t = S.sheet?.strips.find(x => x.id === S.selected);
  $('selectionInfo').textContent = t ? `选中：${t.name}（${t.width}×${t.height}）` : '点击条带查看详情与邻接。';
}

function stripName(id) { return S.sheet.strips.find(t => t.id === id)?.name || id.slice(0, 6); }

function renderIssues() {
  const issues = S.sheet.issues;
  $('issueList').innerHTML = issues.length ? issues.map(i => `
    <div class="issue-card ${i.severity} ${i.resolved ? 'resolved' : ''}">
      <div class="issue-head"><span class="sev ${i.severity}">${i.severity === 'error' ? '错误' : '警告'}</span>
        <label style="font-size:11px"><input type="checkbox" data-issue="${i.id}" ${i.resolved ? 'checked' : ''}> 已确认/解决</label></div>
      <p>${esc(i.message)}</p>
      <div class="issue-refs">关联：${i.strips.map(stripName).map(esc).join(' · ')}</div>
    </div>`).join('') : '<div class="empty-note">没有疑点。邻接关系明确时不会强行制造问题。</div>';
  $('issueList').querySelectorAll('[data-issue]').forEach(cb => cb.onchange = safe(async () => {
    await api('/' + S.sheet.id + '/issues/' + cb.dataset.issue + '/resolve', 'POST', rev({ resolved: cb.checked }));
    S.sheet = await api('/' + S.sheet.id); render();
    toast(cb.checked ? '疑点已标记为解决' : '疑点已重新打开');
  }));
}

const EDGE_LABEL = { accepted: '已接受', rejected: '已拒绝', candidate: '候选' };
function renderEdges() {
  const edges = [...S.sheet.edges].sort((a, b) => b.score - a.score);
  $('edgeList').innerHTML = edges.length ? edges.map(e => `
    <div class="edge-card ${e.status}">
      <div class="edge-row"><strong>${esc(stripName(e.a))} → ${esc(stripName(e.b))}</strong>
        <span class="edge-score">${e.source === 'manual' ? '人工指定' : 'NCC ' + e.score.toFixed(3)}</span></div>
      <div class="scorebar"><i style="width:${Math.round(e.score * 100)}%"></i></div>
      <div class="issue-refs">重叠 ${(e.overlap_fraction * 100).toFixed(1)}% · 状态：${EDGE_LABEL[e.status]}</div>
      <div class="edge-actions">
        <button data-edge-accept='${JSON.stringify({ a: e.a, b: e.b })}'>接受为相邻</button>
        <button data-edge-reject='${JSON.stringify({ a: e.a, b: e.b })}'>拒绝</button>
      </div>
    </div>`).join('') : '<div class="empty-note">尚无邻接关系。点击“分析邻接并建议排列”。</div>';
  $('edgeList').querySelectorAll('[data-edge-accept]').forEach(b => b.onclick = safe(async () => {
    const q = JSON.parse(b.dataset.edgeAccept);
    S.sheet = await api('/' + S.sheet.id + '/edges', 'POST', rev({ ...q, status: 'accepted' })); render();
    toast('已接受该邻接关系，相关疑点标记为解决');
  }));
  $('edgeList').querySelectorAll('[data-edge-reject]').forEach(b => b.onclick = safe(async () => {
    const q = JSON.parse(b.dataset.edgeReject);
    S.sheet = await api('/' + S.sheet.id + '/edges', 'POST', rev({ ...q, status: 'rejected' })); render();
    toast('已拒绝该邻接关系');
  }));
}

// ---------------- 视图切换 ----------------
$('tabLayout').onclick = () => switchView('layout');
$('tabIssues').onclick = () => switchView('issues');
$('tabEdges').onclick = () => switchView('edges');
function switchView(v) {
  S.view = v;
  [['tabLayout', 'layoutView'], ['tabIssues', 'issuesView'], ['tabEdges', 'edgesView']].forEach(([btn, view]) => {
    $(btn).classList.toggle('selected', btn === 'tab' + v[0].toUpperCase() + v.slice(1));
    $(view).hidden = view !== v + 'View';
  });
}

// 缩放
function setZoom(z) { S.zoom = Math.min(2.5, Math.max(0.2, z)); $('zoomLabel').textContent = Math.round(S.zoom * 100) + '%'; renderCanvas(); }
bind('zoomIn', () => setZoom(S.zoom + 0.15));
bind('zoomOut', () => setZoom(S.zoom - 0.15));
bind('zoomFit', () => {
  const scroll = $('canvasScroll'); const s = S.sheet;
  if (!s.strips.length) return;
  const maxW = Math.max(...s.strips.map(t => t.position[0] + t.width));
  const maxH = Math.max(...s.strips.map(t => t.position[1] + t.height));
  setZoom(Math.min((scroll.clientWidth - 60) / maxW, (scroll.clientHeight - 60) / maxH, 1));
});

// ---------------- 条带导入 ----------------
bind('uploadBtn', () => { $('fileInput').value = ''; $('fileInput').click(); });
$('fileInput').onchange = safe(async () => {
  const files = [...$('fileInput').files]; if (!files.length) return;
  // 先上传为不可变素材，再加入布局；原始文件不被改写
  const assetIds = [];
  for (const f of files) {
    const fd = new FormData(); fd.append('file', f);
    const r = await fetch('/api/assets', { method: 'POST', body: fd });
    if (!r.ok) throw Error((await r.json()).error);
    assetIds.push((await r.json()).id);
  }
  await mutate('/' + S.sheet.id + '/strips', 'POST', rev({ asset_ids: assetIds }));
  toast('已导入 ' + assetIds.length + ' 条带，原始扫描独立留存');
});

// ---------------- 分析 / 确认 / 导出 ----------------
bind('analyzeBtn', async () => {
  toast('正在分析条带内容与邻接关系…');
  await mutate('/' + S.sheet.id + '/analyze', 'POST', rev({}));
  const errs = S.sheet.issues.filter(i => i.severity === 'error' && !i.resolved).length;
  toast(errs ? '分析完成，但有 ' + errs + ' 个错误疑点需要你确认' : '分析完成，已给出排列与位置建议');
  setZoom(1);
});

bind('confirmBtn', async () => {
  await mutate('/' + S.sheet.id + '/confirm', 'POST', rev({}));
  toast('布局已确认，导出图将与此布局一致');
});

bind('exportBtn', async () => {
  openDialog('导出整幅拼接图', `
    <p class="tiny" style="font-size:12px;color:#5d6f63">仅在布局确认后可导出。导出严格按当前布局渲染，原始扫描不会被裁切或重写；未覆盖区域留空。</p>
    <div class="field">格式
      <select id="exportFormat"><option value="png">PNG（16位，无损）</option><option value="tiff">TIFF（16位）</option><option value="jpeg">JPEG</option></select></div>
    <div class="dialog-actions">
      <button type="button" id="exportManifest">下载布局清单 JSON</button>
      <button class="primary" id="doExport">导出</button></div>`);
  bind('doExport', async () => {
    const fmt = $('exportFormat').value;
    const r = await fetch(BASE + '/' + S.sheet.id + '/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ revision: S.sheet.revision, format: fmt }) });
    if (!r.ok) throw Error((await r.json()).error);
    const blob = await r.blob();
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = 'strip-layout.' + (fmt === 'jpeg' ? 'jpg' : fmt); a.click();
    setTimeout(() => URL.revokeObjectURL(url), 12000); closeDialog(); toast('整图已导出');
  });
  bind('exportManifest', async () => {
    const m = await api('/' + S.sheet.id + '/manifest');
    const blob = new Blob([JSON.stringify(m, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = 'strip-layout-manifest.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 12000);
  });
});

bind('previewBtn', async () => {
  const blob = await api('/' + S.sheet.id + '/preview');
  if (S.previewUrl) URL.revokeObjectURL(S.previewUrl);
  S.previewUrl = URL.createObjectURL(blob);
  openDialog('整幅布局预览（未覆盖区域以棋盘底标示）', `<img src="${S.previewUrl}" style="max-width:100%;max-height:66vh;object-fit:contain;display:block;margin:auto">`);
});

bind('undoBtn', () => mutate('/' + S.sheet.id + '/undo', 'POST', rev({})));
bind('redoBtn', () => mutate('/' + S.sheet.id + '/redo', 'POST', rev({})));
bind('resetBtn', async () => {
  await mutate('/' + S.sheet.id + '/reset', 'POST', rev({}));
  // 重置后重新运行分析以恢复内容建议位置
  await mutate('/' + S.sheet.id + '/analyze', 'POST', rev({}));
  toast('已清除手动调整并恢复内容建议');
});
bind('duplicateBtn', async () => { const s = await api('/' + S.sheet.id + '/duplicate', 'POST', rev({})); await loadSheet(s.id); toast('已复制布局'); });
bind('removeSel', async () => {
  if (!S.selected) return toast('请先选择条带');
  await mutate('/' + S.sheet.id + '/strips/remove', 'POST', rev({ strip_ids: [S.selected] }));
  S.selected = S.sheet.strips[0]?.id || null; render();
});
bind('rotateSel', async () => {
  if (!S.selected) return;
  const cur = S.sheet.strips.find(t => t.id === S.selected);
  await mutate('/' + S.sheet.id + '/strips/' + S.selected, 'PATCH', rev({ rotation: (cur.rotation + 90) % 360 }));
});
async function shiftSelected(delta) {
  if (!S.selected) return;
  const list = S.sheet.strips; const i = list.findIndex(t => t.id === S.selected);
  const j = i + delta; if (j < 0 || j >= list.length) return;
  const ids = list.map(t => t.id); [ids[i], ids[j]] = [ids[j], ids[i]];
  await mutate('/' + S.sheet.id + '/strips/reorder', 'POST', rev({ strip_ids: ids }));
}
bind('moveUp', () => shiftSelected(-1));
bind('moveDown', () => shiftSelected(1));

// ---------------- 布局属性 / 新建 ----------------
function sheetDialog(s) {
  openDialog(s ? '布局属性' : '新建条带布局', `
    <div class="field">布局名称<input id="dName" value="${esc(s ? s.name : '未命名长幅布局')}"></div>
    <div class="field">拼接方向<select id="dAxis">
      <option value="vertical" ${s && s.axis === 'vertical' ? 'selected' : ''}>纵向接续（条带上下排列）</option>
      <option value="horizontal" ${s && s.axis === 'horizontal' ? 'selected' : ''}>横向接续（条带左右排列）</option></select></div>
    <div class="field">说明<textarea id="dNotes">${esc(s ? s.notes : '')}</textarea></div>
    <div class="dialog-actions"><button class="primary" id="dSave">保存</button></div>`);
  bind('dSave', safe(async () => {
    const body = { name: $('dName').value.trim() || '未命名长幅布局', axis: $('dAxis').value, notes: $('dNotes').value };
    if (s) { await mutate('/' + s.id, 'PATCH', rev(body)); }
    else { const created = await api('', 'POST', body); await loadSheet(created.id); }
    closeDialog();
  }));
}
bind('newSheet', () => sheetDialog(null));
bind('emptyNew', () => sheetDialog(null));
bind('sheetSettings', () => sheetDialog(S.sheet));
$('axisSelect').onchange = safe(async () => {
  await mutate('/' + S.sheet.id, 'PATCH', rev({ axis: $('axisSelect').value }));
  toast('拼接方向已更改，请重新运行邻接分析');
});
bind('closeDialog', closeDialog);
$('dialog').addEventListener('click', e => { if (e.target === $('dialog')) closeDialog(); });
let searchTimer;
$('sheetSearch').oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(safe(refreshSheets), 200); };

// 初始化
safe(async () => {
  await refreshSheets();
  if (S.sheets.length) await loadSheet(S.sheets[0].id);
  else { $('emptyApp').hidden = false; $('workbench').hidden = true; }
})();
