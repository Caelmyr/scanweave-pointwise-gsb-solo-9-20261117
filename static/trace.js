'use strict';
/* 像素贡献溯源前端：只读取现有降噪结果，不触发任何重算或写操作。 */
(function () {
  const T = { active: false, resultId: null, point: null, marker: null };

  const STATUS_LABEL = {
    contributing: '实际贡献',
    partially_excluded: '部分通道被剔除',
    excluded: '已剔除',
    covered_not_selected: '覆盖但未取中值',
    no_coverage: '无覆盖',
    covered: '已覆盖',
  };
  const REASON_LABEL = {
    sigma_outlier: 'Sigma 离群剔除',
    trimmed_outlier: '截尾范围外剔除',
    median_not_selected: '中值未取该样本',
    partial_channel_outlier: '个别通道离群',
    zero_frame_weight: '帧权重为 0',
    no_coverage: '该帧未覆盖此位置（配准/遮罩/片基外）',
  };
  const pct = (v) => (v * 100).toFixed(1) + '%';
  const rgb = (a) => a.map((v) => v.toFixed(3)).join(', ');

  function hero() { return $('hero'); }

  function setActive(on) {
    T.active = on;
    $('traceBtn').classList.toggle('trace-on', on);
    hero().classList.toggle('trace-cursor', on);
    if (!on) removeMarker();
    $('traceBtn').textContent = on ? '⊹ 溯源中…点击结果取点' : '⊹ 像素溯源';
  }

  function removeMarker() {
    if (T.marker) { T.marker.remove(); T.marker = null; }
  }

  function eventPoint(event) {
    const img = hero();
    const rect = img.getBoundingClientRect();
    // 与 CSS object-fit: contain 一致，计算图片实际内容盒。
    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
    const drawnW = img.naturalWidth * scale;
    const drawnH = img.naturalHeight * scale;
    const offX = (rect.width - drawnW) / 2;
    const offY = (rect.height - drawnH) / 2;
    const x = (event.clientX - rect.left - offX) / drawnW;
    const y = (event.clientY - rect.top - offY) / drawnH;
    if (x < 0 || x > 1 || y < 0 || y > 1) return null;
    return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
  }

  function showMarker(nx, ny) {
    removeMarker();
    const img = hero();
    const rect = img.getBoundingClientRect();
    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
    const drawnW = img.naturalWidth * scale;
    const drawnH = img.naturalHeight * scale;
    const offX = (rect.width - drawnW) / 2;
    const offY = (rect.height - drawnH) / 2;
    const dot = document.createElement('div');
    dot.className = 'trace-marker';
    dot.style.left = (offX + nx * drawnW) + 'px';
    dot.style.top = (offY + ny * drawnH) + 'px';
    img.parentElement.appendChild(dot);
    T.marker = dot;
  }

  function frameThumb(frame, point) {
    const marked = `/api${projectPath()}/explain-frame/${frame.frame_id}?result_id=${T.resultId}&marked=true&point=${point[0]},${point[1]}`;
    const original = `/api${projectPath()}/frames/${frame.frame_id}/image`;
    const shares = frame.contribution_share;
    const shareBars = ['R', 'G', 'B'].map((ch, i) =>
      `<span class="share-chip ${shares[i] > 0 ? '' : 'zero'}">${ch} ${pct(shares[i])}</span>`).join('');
    const status = STATUS_LABEL[frame.status] || frame.status;
    const reason = frame.reason ? (REASON_LABEL[frame.reason] || frame.reason) : '—';
    return `<article class="trace-frame ${frame.participates && frame.accepted_channels.some(Boolean) ? '' : 'muted'}">
      <div class="trace-frame-head"><b>${esc(frame.name)}</b><span class="trace-status s-${frame.status}">${esc(status)}</span></div>
      <div class="trace-frame-body">
        <a href="${marked}" target="_blank" rel="noopener" title="对齐帧（共同画布，含点击位置标记）"><img src="${marked}" alt="对齐帧标记"></a>
        <div class="trace-metrics">
          <div class="share-row">${shareBars}</div>
          <div>样本 RGB：${rgb(frame.sample_rgb)}</div>
          <div>覆盖率：${pct(frame.coverage)} · 帧权重：${frame.frame_weight}</div>
          <div class="trace-reason">原因：${esc(reason)}</div>
          <div class="trace-links">
            <a href="${marked}" target="_blank" rel="noopener">对齐帧+标记 ↗</a>
            <a href="${original}" target="_blank" rel="noopener">查看原帧 ↗</a>
          </div>
        </div>
      </div>
    </article>`;
  }

  function projectPath() {
    return '/projects/' + S.project.id;
  }

  async function runTrace(point) {
    if (!S.project) throw Error('请先选择工程');
    if (S.view !== 'result' || !S.project.current_result) {
      toast('请先在「合成预览」中生成多帧降噪结果');
      setActive(false);
      return;
    }
    T.point = point;
    showMarker(point[0], point[1]);
    let report;
    try {
      report = await api(projectPath() + '/pixel-explain', 'POST', { point, radius: 0 });
    } catch (error) {
      toast(error.message);
      setActive(false);
      return;
    }
    T.resultId = report.result_id;
    renderReport(report, point);
  }

  function renderReport(report, point) {
    const canvas = report.inside_canvas ? report.canvas_pixel.join(' , ') : '共同画布外';
    const out = report.output
      ? `<div>降噪输出 RGB（共同画布）：${rgb(report.output.denoise_rgb)}</div>
         <div>当前显示 RGB（裁切/旋转/后期后）：${rgb(report.output.display_rgb)}</div>`
      : '<div class="warning">该位置无任何帧覆盖，不参与计算。</div>';
    const match = report.recomputed_matches_result === true
      ? '<span class="trace-ok">重算与存档像素一致 ✓</span>'
      : report.recomputed_matches_result === false
        ? '<span class="trace-warn">重算与存档存在差异</span>' : '';
    const frames = report.frames.map((f) => frameThumb(f, point)).join('');
    openDialog('像素贡献溯源', `
      <div class="trace-summary">
        <div class="row" style="justify-content:space-between">
          <b>${esc(report.method)} · ${report.linear_space ? '线性空间' : 'sRGB 空间'} · 贡献帧 ${report.contributor_count ?? report.frames.length}</b>
          ${match}
        </div>
        <div>结果像素 (x, y)：${report.display_pixel.join(' , ')} → 共同画布：${esc(canvas)}</div>
        ${out}
        <div>各通道贡献份额合计：R ${pct(report.total_contribution_share?.[0] ?? 0)} · G ${pct(report.total_contribution_share?.[1] ?? 0)} · B ${pct(report.total_contribution_share?.[2] ?? 0)}</div>
        ${report.stale ? '<div class="warning">这是历史结果的解释：工程此后已修改。停用/启用帧并重新合成后，对新结果重新取点即可获得更新解释。</div>' : '<div class="trace-ok">解释绑定当前结果；停用异常帧并重新合成后，新结果的解释会相应更新。</div>'}
        <div class="muted tiny">${esc(report.note)}</div>
      </div>
      <div class="trace-frames">${frames || '<p class="muted">该结果没有可显示的帧。</p>'}</div>
      <div class="dialog-actions"><button id="traceResample" class="primary">在结果上重新取点</button></div>
    `);
    bind('traceResample', () => { closeDialog(); setActive(true); });
  }

  function bindTrace() {
    const btn = $('traceBtn');
    if (!btn) return;
    btn.onclick = safe(() => {
      if (S.view !== 'result' || !S.project?.current_result) {
        toast('请先打开一个多帧降噪合成结果');
        return;
      }
      setActive(!T.active);
    });
    hero().addEventListener('click', safe((event) => {
      if (!T.active || S.view !== 'result') return;
      const point = eventPoint(event);
      if (point) runTrace(point);
    }));
    // 离开结果视图、切换工程或结果刷新后，旧标记与取点状态不再适用。
    const reset = () => { if (T.active || T.marker) setActive(false); T.resultId = null; T.point = null; };
    const origLoad = window.loadProject;
    window.loadProject = async function (...args) {
      reset();
      return origLoad.apply(this, args);
    };
    document.querySelectorAll('[data-view]').forEach((el) => {
      el.addEventListener('click', () => { if (el.dataset.view !== 'result') setActive(false); });
    });
  }

  window.scanweaveTrace = { bindTrace, setActive, removeMarker, reset: () => { T.resultId = null; T.point = null; } };
})();
