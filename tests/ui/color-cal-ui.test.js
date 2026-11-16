/* 色彩校正前端逻辑的 DOM 级烟测：用 jsdom + 桩件模拟完整交互。 */
const fs = require('fs');
const {JSDOM} = require('./node_modules/jsdom');

const html = fs.readFileSync(require('path').resolve(__dirname, '../../templates/index.html'), 'utf8');
const dom = new JSDOM(html.replace(/<script[\s\S]*?<\/script>/g, ''), {pretendToBeVisual: true});
const {window} = dom;
global.window = window;
global.document = window.document;
global.HTMLElement = window.HTMLElement;
global.Image = window.Image;
global.fetch = async () => { throw new Error('no network in test'); };
global.URL.createObjectURL = () => 'blob:stub';
global.URL.revokeObjectURL = () => {};
global.createImageBitmap = async () => ({width: 720, height: 480, close() {}});

// 最小桩件
const store = {};
global.S = {project: {id: 'a'.repeat(32), revision: 3, reference: 'f1', frames: [
    {id: 'f1', name: 'scan-1', asset: 'x'.repeat(64), enabled: true},
    {id: 'f2', name: 'scan-2', asset: 'y'.repeat(64), enabled: true}], settings: {size: 768}},
  projects: [], selected: 'f1', result: null, view: 'result', url: null, dialogUrl: null, cap: null, busy: false};
let lastCall = null;
global.api = async (urlPath, method = 'GET', data) => {
  lastCall = {urlPath, method, data};
  if (urlPath.endsWith('/color-plans') && method === 'GET') {
    return store.plans || [];
  }
  if (urlPath.endsWith('/color-plans/analyze')) {
    store.analyzeCalls = (store.analyzeCalls || 0) + 1;
    return {usable: true, reliable_count: data.patches.length, patch_count: data.patches.length,
      model: data.model, issues: [], warnings: ['测试警告'],
      fit: {rank: 3, design_columns: 3, condition: 9.1,
        error_before: {mean: 15.4, maximum: 19}, error: {mean: 2.6, maximum: 5, note: 'note'},
        coefficients: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
      patches: data.patches.map((p, i) => ({index: i, name: p.name, target: p.target, reliable: true,
        flags: [], rgb: [.5, .5, .5], linear: [.2, .2, .2], std: [.01, .01, .01],
        clipped_fraction: 0, pixels: 100, delta_e_before: 10, delta_e_after: 2, delta_e_cv: 3}))};
  }
  if (urlPath.endsWith('/color-plans') && method === 'POST') {
    const plan = {id: 'b'.repeat(32), name: data.name, frame_id: data.frame_id, model: data.model,
      patches: data.patches, applied: false, coefficients: [],
      fit_report: {delta_e_mean_after: 2.6, reliable_count: data.patches.length, patch_count: data.patches.length, warnings: []}};
    store.plans = [plan];
    S.project.settings.color_plans = [plan];
    S.project.revision += 1;
    return {project: S.project, plan, analysis: {}};
  }
  if (urlPath.includes('/apply')) {
    store.plans[0].applied = true;
    S.project.settings.color_plans = store.plans;
    S.project.revision += 1;
    return {project: S.project, plan: store.plans[0]};
  }
  if (urlPath.includes('/disable')) {
    store.plans[0].applied = false;
    S.project.settings.color_plans = store.plans;
    S.project.revision += 1;
    return {project: S.project, plan: store.plans[0]};
  }
  if (urlPath.includes('/remove')) {
    store.plans = [];
    S.project.settings.color_plans = [];
    S.project.revision += 1;
    return S.project;
  }
  if (urlPath.includes('/scanner/chart-palette')) {
    return JSON.parse(fs.readFileSync(require('path').resolve(__dirname, 'palette.json'), 'utf8'));
  }
  if (urlPath.includes('/frames/f1/image')) return new window.Blob(['x']);
  throw new Error('unmocked ' + method + ' ' + urlPath);
};
global.mutate = async (suffix, method, data) => {
  const result = await api('/projects/' + S.project.id + suffix, method, {revision: S.project.revision, ...data});
  if (result.project) { S.project = result.project; }
  return result;
};
global.toast = msg => { store.toast = msg; };
global.safe = fn => async (...args) => { try { await fn(...args); } catch (e) { store.error = e.message; } };
global.bind = (id, fn) => { const el = $(id); if (el) el.onclick = fn; };
global.esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
global.openDialog = (title, html) => { document.getElementById('dialogTitle').textContent = title; document.getElementById('dialogBody').innerHTML = html; window.dialogOpened = true; };
global.closeDialog = () => { window.dialogClosed = true; };
global.imagePanel = (title, blob) => { store.imagePanel = title; };
global.confirm = () => true;
global.prompt = () => '前端测试方案';
const $ = id => document.getElementById(id);
global.$ = $;
global.path = suffix => '/projects/' + S.project.id + suffix;
global.rev = extra => ({revision: S.project.revision, ...extra});
// canvas 2D 桩
window.HTMLCanvasElement.prototype.getContext = function () {
  return new Proxy({}, {get: (t, prop) => {
    if (prop === 'canvas') return this;
    return typeof prop === 'string' ? () => {} : undefined;
  }, set: () => true});
};
window.HTMLCanvasElement.prototype.setPointerCapture = () => {};

// 加载被测脚本（color-cal.js 依赖的 app.js 工具已用桩件替代）
const vm = require('vm');
const code = fs.readFileSync(require('path').resolve(__dirname, '../../static/color-cal.js'), 'utf8');
vm.runInThisContext(code);

(async () => {
  const palette = [
    {name: '灰', target: '#7d7d7d', bounds: [.1, .1, .2, .2]},
    {name: '红', target: '#b54832', bounds: [.3, .1, .4, .2]},
    {name: '蓝', target: '#3d5a96', bounds: [.5, .1, .6, .2]}];
  fs.writeFileSync(require('path').resolve(__dirname, 'palette.json'), JSON.stringify(palette));

  // 1. 打开对话框
  await colorCalibrationDialog();
  if (!$('ccCanvas') || !$('ccSave')) throw new Error('对话框元素缺失');
  if (!$('ccSave').disabled) throw new Error('无分析结果时保存按钮应禁用');

  // 2. 载入样张色卡 -> 自动试算
  await $('ccLoadPalette').onclick();
  await new Promise(r => setTimeout(r, 350));
  if (CC.patches.length !== 3) throw new Error('色卡载入失败');
  if (store.analyzeCalls !== 1) throw new Error('应触发一次试算，实际 ' + store.analyzeCalls);
  if ($('ccSave').disabled) throw new Error('试算可靠后保存按钮应可用');
  const metrics = $('ccAnalysis').textContent;
  if (!metrics.includes('2.60') || !metrics.includes('15.40')) throw new Error('色差指标未展示: ' + metrics);
  if (!metrics.includes('可靠色块')) throw new Error('可靠性汇总未展示');

  // 3. 修改目标颜色触发重新试算
  const hex = document.querySelectorAll('.cc-hex')[0];
  hex.value = '#123456';
  hex.onchange();
  await new Promise(r => setTimeout(r, 350));
  if (CC.patches[0].target !== '#123456') throw new Error('HEX 颜色未更新');
  if (store.analyzeCalls !== 2) throw new Error('改色应重新试算');

  // 4. 非法 HEX 被拒绝
  hex.value = 'nope';
  hex.onchange();
  if (CC.patches[0].target !== '#123456') throw new Error('非法 HEX 不应生效');
  if (!store.toast.includes('#RRGGBB')) throw new Error('应提示颜色格式错误');

  // 5. 删除色块
  document.querySelector('.cc-del').onclick();
  if (CC.patches.length !== 2) throw new Error('删除色块失败');

  // 6. 保存 -> 确认应用
  await $('ccSave').onclick();
  if (!store.plans || store.plans.length !== 1) throw new Error('方案未保存');
  if (!store.plans[0].applied) throw new Error('确认后方案应处于应用状态');
  if (!window.dialogClosed) throw new Error('应用后应关闭对话框');

  // 7. 管理面板
  await colorPlansManager();
  if (!document.querySelector('[data-disable]')) throw new Error('已应用方案应显示停用按钮');
  await document.querySelector('[data-disable]').onclick();
  if (store.plans[0].applied) throw new Error('停用失败');

  // 8. 阻断状态禁用保存：让 analyze 返回不可用
  const blockedApi = async (urlPath, method = 'GET', data) => {
    if (urlPath.endsWith('/color-plans/analyze')) {
      return {usable: false, reliable_count: 0, patch_count: 1, issues: ['色块数量不足'], warnings: [], patches: []};
    }
    if (urlPath.endsWith('/color-plans') && method === 'GET') return store.plans || [];
    if (urlPath.includes('/frames/f1/image')) return new window.Blob(['x']);
    throw new Error('unmocked2 ' + method + ' ' + urlPath);
  };
  global.api = blockedApi;
  await colorCalibrationDialog();
  CC.patches = [{id: 'p1', name: 'x', bounds: [.1, .1, .2, .2], target: '#808080'},{id: 'p2', name: 'y', bounds: [.3, .1, .4, .2], target: '#909090'},{id: 'p3', name: 'z', bounds: [.5, .1, .6, .2], target: '#a0a0a0'}];
  await ccAnalyze();
  if (!$('ccSave').disabled) throw new Error('阻断状态保存按钮必须禁用');
  if (!$('ccAnalysis').textContent.includes('色块数量不足')) throw new Error('阻断问题未展示');

  console.log('前端 DOM 测试全部通过');
})().catch(e => { console.error('FAIL:', e.message); process.exit(1); });
