'use strict';
/* 批次色彩校正：圈选色块、填写已知颜色、查看拟合质量后再决定是否应用。 */
const CC = {patches: [], analysis: null, frameId: null, image: null, planId: null, model: 'matrix'};
const CC_MODELS = [['matrix', '线性矩阵（3×3，至少3个色块）'], ['affine', '仿射矩阵（含偏移，至少4个色块）'], ['polynomial', '二阶多项式（至少10个色块，覆盖非线性色偏）']];
const CC_PALETTE = ['#3a3a3a', '#7d7d7d', '#c2c2c2', '#eeeeee', '#b54832', '#c9973f', '#6f8f4e', '#3f9c8f', '#3d5a96', '#7d4f9c', '#bd5f86', '#c06f3a', '#000000', '#ffffff'];
const CC_DEBOUNCE = {timer: null};

function ccUid() {
    return 'p' + Math.random().toString(16).slice(2, 10) + Date.now().toString(16).slice(-4);
}

function ccPatchPayload() {
    return CC.patches.map(p => ({id: p.id, name: p.name, bounds: p.bounds, target: p.target}));
}

async function colorCalibrationDialog(planId) {
    if (!S.project) throw Error('请先载入扫描批次');
    const plans = await api(path('/color-plans'));
    plansCache = plans;
    let editing = planId ? plans.find(q => q.id === planId) : null;
    CC.planId = editing ? editing.id : null;
    CC.frameId = editing ? editing.frame_id : S.project.reference;
    CC.model = editing ? editing.model : 'matrix';
    CC.patches = editing ? editing.patches.map(p => ({...p})) : [];
    CC.analysis = null;
    const planOptions = plans.length ? plans.map(q => `<option value="${q.id}" ${editing && q.id === editing.id ? 'selected' : ''}>${esc(q.name)}${q.applied ? '（已应用）' : ''}</option>`).join('') : '';
    openDialog(editing ? '调整色彩校正方案 · ' + editing.name : '批次色彩校正 · 色卡拟合', `
<div class="cc-layout">
  <div class="cc-canvas-wrap">
    <canvas id="ccCanvas" class="cc-canvas" width="720" height="480"></canvas>
    <div class="cc-hint" id="ccHint">在图上拖拽圈出色块，松开后在右侧填写该色块的已知颜色</div>
  </div>
  <div class="cc-side">
    <label class="field">采样帧（色卡所在扫描）
      <select id="ccFrame">${S.project.frames.map(f => `<option value="${f.id}" ${f.id === CC.frameId ? 'selected' : ''}>${esc(f.name)}${f.id === S.project.reference ? ' ★' : ''}${f.enabled ? '' : '（未启用）'}</option>`).join('')}</select>
    </label>
    <div class="row" style="gap:6px">
      <button id="ccLoadPalette" type="button" title="载入合成样张内置12格色卡的位置与已知颜色">载入样张色卡</button>
      <button id="ccClear" type="button">清空色块</button>
    </div>
    <label class="field">校正模型<select id="ccModel">${CC_MODELS.map(([v, l]) => `<option value="${v}" ${v === CC.model ? 'selected' : ''}>${esc(l)}</option>`).join('')}</select></label>
    ${plans.length ? `<label class="field">已保存方案<select id="ccPlanSelect"><option value="">— 新建方案 —</option>${planOptions}</select></label>` : ''}
    <div class="section-title">色块清单（<span id="ccCount">0</span>）</div>
    <div id="ccPatchList" class="cc-patch-list"><p class="muted">还没有色块。</p></div>
    <div id="ccAnalysis" class="cc-analysis muted">圈好至少 3 个分散色块后自动试算。</div>
  </div>
</div>
<div class="dialog-actions">
  <button type="button" id="ccPreview">校正前后分屏预览</button>
  <button type="button" class="primary" id="ccSave">保存方案</button>
  <button type="button" id="ccClose">关闭</button>
</div>`);
    await ccLoadImage();
    ccBindCanvas();
    ccBindControls(plans);
    ccRenderPatches();
    if (CC.patches.length) ccScheduleAnalyze();
}

async function ccLoadImage() {
    const blob = await api(path('/frames/' + CC.frameId + '/image'));
    if (CC.image) CC.image.close && CC.image.close();
    CC.image = await createImageBitmap(blob);
    const canvas = $('ccCanvas');
    canvas.width = CC.image.width;
    canvas.height = CC.image.height;
    ccDraw();
}

function ccDraw() {
    const canvas = $('ccCanvas'), ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(CC.image, 0, 0, canvas.width, canvas.height);
    CC.patches.forEach((p, i) => {
        const [x0, y0, x1, y1] = p.bounds;
        const left = x0 * canvas.width, top = y0 * canvas.height;
        const w = (x1 - x0) * canvas.width, h = (y1 - y0) * canvas.height;
        const analysis = CC.analysis;
        const report = analysis && analysis.patches ? analysis.patches.find(q => q.index === i) : null;
        let color = '#f4f1e6';
        if (report) color = report.reliable ? '#74d49b' : '#ef7a5a';
        if (ccDrag && ccDrag.index === i) color = '#ffd166';
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(left, top, w, h);
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.fillRect(left, top - 18, 34, 18);
        ctx.fillStyle = '#16242a';
        ctx.font = '12px sans-serif';
        ctx.fillText(String(i + 1), left + 5, top - 5);
        ctx.fillStyle = p.target;
        ctx.fillRect(left + w - 22, top, 22, 16);
        ctx.strokeStyle = 'rgba(0,0,0,.6)';
        ctx.strokeRect(left + w - 22, top, 22, 16);
    });
    if (ccDrag && ccDrag.live) {
        const r = ccDrag.live;
        ctx.strokeStyle = '#ffd166';
        ctx.lineWidth = 2;
        ctx.strokeRect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
    }
}

let ccDrag = null;
function ccEventPoint(e) {
    const canvas = $('ccCanvas');
    const rect = canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
            Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))];
}

function ccBindCanvas() {
    const canvas = $('ccCanvas');
    canvas.onpointerdown = e => {
        const [x, y] = ccEventPoint(e);
        const hit = CC.patches.findIndex(p => x >= p.bounds[0] && x <= p.bounds[2] && y >= p.bounds[1] && y <= p.bounds[3]);
        if (hit >= 0 && e.shiftKey) {
            CC.patches.splice(hit, 1);
            ccRenderPatches();
            ccScheduleAnalyze();
            return;
        }
        ccDrag = {start: [x, y], live: null, index: CC.patches.length};
        canvas.setPointerCapture(e.pointerId);
    };
    canvas.onpointermove = e => {
        if (!ccDrag) return;
        const [x, y] = ccEventPoint(e);
        ccDrag.live = [ccDrag.start[0] * canvas.width, ccDrag.start[1] * canvas.height, x * canvas.width, y * canvas.height];
        ccDraw();
    };
    canvas.onpointerup = e => {
        if (!ccDrag) return;
        const [x, y] = ccEventPoint(e);
        const [sx, sy] = ccDrag.start;
        ccDrag = null;
        if (Math.abs(x - sx) < .01 || Math.abs(y - sy) < .01) { ccDraw(); return; }
        const bounds = [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)];
        if (CC.patches.length >= 48) { toast('最多48个色块'); ccDraw(); return; }
        CC.patches.push({id: ccUid(), name: '色块 ' + (CC.patches.length + 1), bounds, target: '#808080'});
        ccRenderPatches();
        ccScheduleAnalyze();
    };
}

function ccRenderPatches() {
    $('ccCount').textContent = CC.patches.length;
    const el = $('ccPatchList');
    if (!CC.patches.length) {
        el.innerHTML = '<p class="muted">还没有色块。</p>';
        ccDraw();
        return;
    }
    const reports = CC.analysis && CC.analysis.patches ? CC.analysis.patches : [];
    el.innerHTML = CC.patches.map((p, i) => {
        const r = reports[i];
        const status = r ? (r.reliable ? '<span class="cc-ok">可靠</span>' : '<span class="cc-bad">不可靠</span>') : '';
        const dE = r && r.delta_e_after != null ? `ΔE ${r.delta_e_before.toFixed(1)} → ${r.delta_e_after.toFixed(1)}` : (r ? `ΔE ${r.delta_e_before.toFixed(1)}` : '');
        const swatches = CC_PALETTE.map(c => `<button type="button" class="cc-swatch" data-color="${c}" data-index="${i}" style="background:${c}" title="${c}"></button>`).join('');
        return `<div class="cc-patch ${r && !r.reliable ? 'cc-patch-bad' : ''}">
          <div class="cc-patch-head"><b>${i + 1}. ${esc(p.name)}</b>${status}</div>
          <input class="cc-name" data-index="${i}" value="${esc(p.name)}" aria-label="色块名称">
          <div class="row" style="gap:6px;margin:6px 0">
            <input type="color" class="cc-target" data-index="${i}" value="${p.target}" aria-label="已知颜色">
            <input type="text" class="cc-hex" data-index="${i}" value="${p.target}" pattern="#[0-9a-fA-F]{6}">
            <button type="button" class="cc-del" data-index="${i}">删除</button>
          </div>
          <div class="cc-swatches">${swatches}</div>
          <div class="cc-de">${dE}</div>
          ${r && r.flags.length ? `<ul class="cc-flags">${r.flags.map(f => `<li>${esc(f)}</li>`).join('')}</ul>` : ''}
        </div>`;
    }).join('');
    el.querySelectorAll('.cc-name').forEach(input => input.onchange = safe(() => {
        CC.patches[Number(input.dataset.index)].name = input.value || '色块';
        ccDraw();
    }));
    el.querySelectorAll('.cc-target').forEach(input => input.oninput = () => {
        const i = Number(input.dataset.index);
        CC.patches[i].target = input.value;
        el.querySelectorAll('.cc-hex')[i].value = input.value;
        ccDraw();
        ccScheduleAnalyze();
    });
    el.querySelectorAll('.cc-hex').forEach(input => input.onchange = () => {
        if (!/^#[0-9a-fA-F]{6}$/.test(input.value)) { toast('颜色必须为 #RRGGBB'); input.value = CC.patches[Number(input.dataset.index)].target; return; }
        const i = Number(input.dataset.index);
        CC.patches[i].target = input.value.toLowerCase();
        el.querySelectorAll('.cc-target')[i].value = CC.patches[i].target;
        ccDraw();
        ccScheduleAnalyze();
    });
    el.querySelectorAll('.cc-swatch').forEach(btn => btn.onclick = () => {
        const i = Number(btn.dataset.index);
        CC.patches[i].target = btn.dataset.color;
        ccRenderPatches();
        ccScheduleAnalyze();
    });
    el.querySelectorAll('.cc-del').forEach(btn => btn.onclick = () => {
        CC.patches.splice(Number(btn.dataset.index), 1);
        ccRenderPatches();
        ccScheduleAnalyze();
    });
    ccDraw();
}

function ccScheduleAnalyze() {
    clearTimeout(CC_DEBOUNCE.timer);
    CC_DEBOUNCE.timer = setTimeout(safe(ccAnalyze), 250);
}

async function ccAnalyze() {
    if (CC.patches.length < 2) {
        CC.analysis = null;
        ccRenderPatches();
        ccRenderAnalysis();
        return;
    }
    const out = $('ccAnalysis');
    out.className = 'cc-analysis muted';
    out.textContent = '试算中…';
    try {
        CC.analysis = await api(path('/color-plans/analyze'), 'POST', {
            frame_id: CC.frameId, model: CC.model, patches: ccPatchPayload()});
    } catch (error) {
        CC.analysis = null;
        ccRenderPatches();
        out.className = 'cc-analysis cc-blocked';
        out.textContent = error.message;
        $('ccSave').disabled = true;
        $('ccPreview').disabled = true;
        ccDraw();
        return;
    }
    ccRenderPatches();
    ccRenderAnalysis();
}

function ccRenderAnalysis() {
    const out = $('ccAnalysis');
    const a = CC.analysis;
    if (!a || !a.fit) {
        out.className = 'cc-analysis ' + (a && a.issues && a.issues.length ? 'cc-blocked' : 'muted');
        out.innerHTML = (a && a.issues && a.issues.length) ? a.issues.map(f => `<div class="cc-issue">✕ ${esc(f)}</div>`).join('') : '色块不足，继续圈选。';
        $('ccSave').disabled = !(a && a.usable);
        $('ccPreview').disabled = !(a && a.usable);
        return;
    }
    const fit = a.fit;
    const issues = a.issues.map(f => `<div class="cc-issue">✕ ${esc(f)}</div>`).join('');
    const warnings = a.warnings.map(f => `<div class="cc-warn">! ${esc(f)}</div>`).join('');
    out.className = 'cc-analysis ' + (a.usable ? 'cc-ok-panel' : 'cc-blocked');
    out.innerHTML = `
      <div class="cc-metrics">
        <div><span>平均ΔE 拟合前</span><b>${fit.error_before.mean.toFixed(2)}</b></div>
        <div><span>平均ΔE 拟合后</span><b class="${fit.error.mean <= 4 ? 'cc-ok' : 'cc-bad'}">${fit.error.mean.toFixed(2)}</b></div>
        <div><span>最大ΔE 拟合后</span><b>${fit.error.maximum.toFixed(2)}</b></div>
        <div><span>可靠色块</span><b>${a.reliable_count}/${a.patch_count}</b></div>
        <div><span>条件数</span><b>${fit.condition.toExponential(1)}</b></div>
        <div><span>模型秩</span><b>${fit.rank}/${fit.design_columns}</b></div>
      </div>
      ${issues}${warnings}
      <p class="cc-note">${esc(fit.error.note)}</p>`;
    $('ccSave').disabled = !a.usable;
    $('ccPreview').disabled = !a.usable;
}

function ccBindControls(plans) {
    $('ccFrame').onchange = safe(async () => {
        CC.frameId = $('ccFrame').value;
        CC.analysis = null;
        await ccLoadImage();
        if (CC.patches.length) ccScheduleAnalyze();
    });
    $('ccModel').onchange = () => { CC.model = $('ccModel').value; ccScheduleAnalyze(); };
    $('ccClear').onclick = () => {
        CC.patches = []; CC.analysis = null; ccRenderPatches(); ccRenderAnalysis();
        $('ccSave').disabled = true; $('ccPreview').disabled = true;
    };
    $('ccClose').onclick = closeDialog;
    $('ccLoadPalette').onclick = safe(ccLoadBuiltinPalette);
    if ($('ccPlanSelect')) $('ccPlanSelect').onchange = safe(async () => {
        const id = $('ccPlanSelect').value;
        if (id) await colorCalibrationDialog(id);
    });
    $('ccSave').onclick = safe(ccSave);
    $('ccPreview').onclick = safe(ccPreview);
    $('ccSave').disabled = true;
    $('ccPreview').disabled = true;
}

async function ccLoadBuiltinPalette() {
    const palette = await api('/scanner/chart-palette');
    CC.patches = palette.map((q, i) => ({id: ccUid(), name: q.name, bounds: q.bounds, target: q.target}));
    toast('已载入12格样张色卡，可调整框选或改用自己的已知颜色');
    ccRenderPatches();
    ccScheduleAnalyze();
}

async function ccSave() {
    const a = CC.analysis;
    if (!a) { toast('请先圈选色块并完成试算'); return; }
    if (!a.usable) { toast('存在阻断性问题，不能保存看似可靠的校正：' + a.issues.join('；')); return; }
    const defaultName = S.project.name + ' · 批次色卡校正';
    const name = CC.planId ? plans_name() : prompt('为这套校正方案命名', defaultName);
    if (!name) return;
    const payload = {frame_id: CC.frameId, model: CC.model, name, patches: ccPatchPayload(), applied: false};
    if (CC.planId) payload.plan_id = CC.planId;
    const result = await mutate('/color-plans', 'POST', payload);
    CC.planId = result.plan.id;
    toast('方案已保存到工程，重新打开仍可调整；尚未应用到扫描');
    await ccApplyAfterSave(result.plan);
}

function plans_name() {
    const sel = $('ccPlanSelect');
    const editing = (CC.planId && plansCache.find(q => q.id === CC.planId)) || null;
    if (editing) return editing.name;
    return sel && sel.selectedOptions[0] ? sel.selectedOptions[0].text.replace(/（已应用）/, '') : '批次色卡校正';
}
let plansCache = [];

async function ccApplyAfterSave(plan) {
    const applyNow = confirm(`方案「${plan.name}」已保存。\n拟合后平均ΔE ${plan.fit_report.delta_e_mean_after.toFixed(2)}，可靠色块 ${plan.fit_report.reliable_count}/${plan.fit_report.patch_count}。\n\n确定应用到本批所有启用的扫描？\n（原始扫描不会被修改，应用后需重新合成与导出；点“取消”仅保存不应用）`);
    if (!applyNow) { await colorCalibrationDialog(plan.id); return; }
    await mutate('/color-plans/' + plan.id + '/apply', 'POST', {});
    toast('色彩校正已应用，重新合成后导出即使用校正结果');
    closeDialog();
}

async function ccPreview() {
    if (!CC.analysis || !CC.analysis.usable) { toast('当前试算不可靠，无法生成预览'); return; }
    const blob = await api(path('/color-plans/preview'), 'POST', {
        frame_id: CC.frameId, model: CC.model, patches: ccPatchPayload(), fraction: .5});
    imagePanel('校正前后分屏（左：原始扫描 / 右：校正结果）', blob);
}

async function colorPlansManager() {
    if (!S.project) throw Error('请先载入扫描批次');
    const plans = await api(path('/color-plans'));
    openDialog('批次色彩校正方案', `
      <p class="muted">方案保存在工程内，可随时调整。应用后所有启用扫描的合成与导出都会使用校正，原始扫描保持不变。</p>
      ${plans.length ? plans.map(p => `
        <div class="job-entry cc-managed ${p.applied ? 'cc-managed-active' : ''}">
          <div class="row"><b>${esc(p.name)}</b>${p.applied ? '<span class="cc-ok">● 已应用</span>' : '<span class="muted">未应用</span>'}</div>
          <div class="cc-de">模型 ${esc(p.model)} · ${p.patches.length} 个色块 · 平均ΔE ${p.fit_report ? p.fit_report.delta_e_mean_after.toFixed(2) : '—'}</div>
          ${p.fit_report && p.fit_report.warnings.length ? `<ul class="cc-flags">${p.fit_report.warnings.slice(0, 2).map(w => `<li>${esc(w)}</li>`).join('')}</ul>` : ''}
          <div class="row" style="margin-top:8px">
            <button data-edit="${p.id}">调整</button>
            ${p.applied ? `<button data-disable="${p.id}">停用</button>` : `<button data-apply="${p.id}">应用到本批扫描</button>`}
            <button data-remove="${p.id}">删除</button>
          </div>
        </div>`).join('') : '<p class="muted">还没有校正方案。</p>'}
      <div class="dialog-actions"><button class="primary" id="ccNew" type="button">新建校正方案</button></div>`);
    bind('ccNew', () => colorCalibrationDialog());
    $('dialogBody').querySelectorAll('[data-edit]').forEach(b => b.onclick = safe(() => colorCalibrationDialog(b.dataset.edit)));
    $('dialogBody').querySelectorAll('[data-apply]').forEach(b => b.onclick = safe(async () => {
        await mutate('/color-plans/' + b.dataset.apply + '/apply', 'POST', {});
        toast('已应用，重新合成后生效'); colorPlansManager();
    }));
    $('dialogBody').querySelectorAll('[data-disable]').forEach(b => b.onclick = safe(async () => {
        await mutate('/color-plans/' + b.dataset.disable + '/disable', 'POST', {});
        toast('已停用校正'); colorPlansManager();
    }));
    $('dialogBody').querySelectorAll('[data-remove]').forEach(b => b.onclick = safe(async () => {
        if (!confirm('删除这套校正方案？')) return;
        await mutate('/color-plans/' + b.dataset.remove + '/remove', 'POST', {});
        colorPlansManager();
    }));
}
