'use strict';
/* Region correspondence review across historical results.
   Read-only: the server re-derives geometry from each result's saved
   snapshot and never selects, rewrites or activates a result here. */
const REVIEW={bounds:[.3,.3,.7,.7],frameId:null,first:null,second:null,gain:4,payload:null,results:[],bitmap:null};

function reviewBoundsText(bounds){
  if(!bounds)return '尚未框选';
  const scale=REVIEW.bitmap?{w:REVIEW.bitmap.width,h:REVIEW.bitmap.height}:{w:1,h:1};
  const px=[Math.floor(bounds[0]*scale.w),Math.floor(bounds[1]*scale.h),Math.ceil(bounds[2]*scale.w),Math.ceil(bounds[3]*scale.h)];
  return `x ${bounds[0].toFixed(3)}–${bounds[2].toFixed(3)} · y ${bounds[1].toFixed(3)}–${bounds[3].toFixed(3)}（素材像素 ${px.join(', ')}）`;
}
function reviewVersionLabel(entry){
  const when=String(entry.created_at||'').replace('T',' ').slice(0,16);
  return `${entry.id.slice(0,8)} · ${entry.mode||'?'} · REV ${entry.source_revision} · ${when}${entry.current?' · 当前':''}${entry.stale?' · 设置已变更':''}`;
}
function reviewInvalidate(){
  REVIEW.payload=null;
  const output=$('reviewOutput');
  if(output)output.innerHTML='<p class="muted">区域或版本已改变，点击“生成对比”更新结果。</p>';
}
function reviewStatusBadge(status){
  const labels={ok:['已定位','ok'],partial:['部分覆盖','warn'],approximate:['近似对应','warn'],no_overlap:['无重叠','bad'],outside:['区域已被裁掉','bad'],no_valid:['无有效像素','bad'],
    missing_frame:['缺少对应素材','bad'],disabled_frame:['对应帧已停用','bad'],ambiguous_frame:['对应不唯一','bad'],
    geometry_mismatch:['几何无法对齐','bad'],unreadable:['无法读取','bad'],skipped:['未计算','bad']};
  const [text,kind]=labels[status]||[status||'未知','bad'];
  return `<span class="review-badge ${kind}">${esc(text)}</span>`;
}

async function reviewLoadFrame(frameId){
  const frame=project().frames.find(f=>f.id===frameId);
  if(!frame)throw Error('找不到参考扫描帧');
  const blob=await api('/assets/'+frame.asset+'/image');
  const bitmap=await createImageBitmap(blob);
  if(REVIEW.bitmap)REVIEW.bitmap.close();
  REVIEW.bitmap=bitmap;
  REVIEW.frameId=frameId;
}

function reviewPaintCanvas(){
  const canvas=$('reviewCanvas');
  if(!canvas||!REVIEW.bitmap)return;
  canvas.width=REVIEW.bitmap.width;canvas.height=REVIEW.bitmap.height;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(REVIEW.bitmap,0,0);
  const b=REVIEW.bounds;
  const x0=Math.floor(b[0]*canvas.width),y0=Math.floor(b[1]*canvas.height);
  const x1=Math.ceil(b[2]*canvas.width),y1=Math.ceil(b[3]*canvas.height);
  ctx.fillStyle='rgba(38,91,89,.18)';ctx.fillRect(x0,y0,x1-x0,y1-y0);
  ctx.strokeStyle='#e0b77a';ctx.lineWidth=2;ctx.strokeRect(x0,y0,x1-x0,y1-y0);
  const label=$('reviewBounds');
  if(label)label.textContent=reviewBoundsText(b);
}

async function reviewDialog(){
  project();
  const p=project();
  const frames=p.frames.filter(f=>f.enabled);
  if(!frames.length)throw Error('请先启用参考扫描帧');
  const frameId=p.reference&&frames.some(f=>f.id===p.reference)?p.reference:frames[0].id;
  await reviewLoadFrame(frameId);
  const listing=await api(path('/review/results'));
  REVIEW.results=listing.results||[];
  if(!REVIEW.results.length)throw Error('还没有历史合成结果，请先合成一次');
  const options=REVIEW.results.map(item=>[item.id,reviewVersionLabel(item)]);
  const defaultFirst=REVIEW.results[0].id;
  const defaultSecond=(REVIEW.results.find(item=>item.id!==defaultFirst)||REVIEW.results[0]).id;
  REVIEW.first=REVIEW.first&&REVIEW.results.some(r=>r.id===REVIEW.first)?REVIEW.first:defaultFirst;
  REVIEW.second=REVIEW.second&&REVIEW.results.some(r=>r.id===REVIEW.second)?REVIEW.second:defaultSecond;
  REVIEW.payload=null;
  openDialog('区域复核 · 跨版本对照',`
    <p class="muted">在原始参考扫描上框选一块内容，再选择两份历史结果。系统按各版本保存的处理几何定位同一内容，并只比较双方都有效的像素；被裁掉或没有对应的版本会明确标出。</p>
    <div class="review-top">
      <div class="review-canvas-wrap">
        <canvas id="reviewCanvas" class="review-canvas"></canvas>
        <div class="tiny" id="reviewBounds"></div>
      </div>
      <div class="review-controls">
        <label class="field">参考扫描帧<select id="reviewFrame">${frames.map(f=>`<option value="${f.id}" ${f.id===frameId?'selected':''}>${esc(f.name)}</option>`).join('')}</select></label>
        <label class="field">历史结果 A<select id="reviewFirst">${options.map(([v,l])=>`<option value="${v}" ${v===REVIEW.first?'selected':''}>${esc(l)}</option>`).join('')}</select></label>
        <label class="field">历史结果 B<select id="reviewSecond">${options.map(([v,l])=>`<option value="${v}" ${v===REVIEW.second?'selected':''}>${esc(l)}</option>`).join('')}</select></label>
        <label class="field">差异增益<input id="reviewGain" type="number" min="1" max="16" step="1" value="${REVIEW.gain}"></label>
        <div class="row"><button id="reviewRun" class="primary">生成对比</button><button id="reviewDefault">恢复默认区域</button></div>
        <p class="tiny">拖动画布框选区域。复核只读，不会切换当前结果或改写历史文件。</p>
      </div>
    </div>
    <div id="reviewOutput" class="review-output"><p class="muted">框选区域后点击“生成对比”。</p></div>`);
  reviewPaintCanvas();
  bindCanvasSelection();
  $('reviewFrame').onchange=safe(async()=>{
    await reviewLoadFrame($('reviewFrame').value);
    reviewPaintCanvas();
    reviewInvalidate();
  });
  $('reviewFirst').onchange=()=>{REVIEW.first=$('reviewFirst').value;reviewInvalidate();};
  $('reviewSecond').onchange=()=>{REVIEW.second=$('reviewSecond').value;reviewInvalidate();};
  $('reviewGain').onchange=()=>{REVIEW.gain=Number($('reviewGain').value)||4;reviewInvalidate();};
  bind('reviewDefault',()=>{REVIEW.bounds=[.3,.3,.7,.7];reviewPaintCanvas();reviewInvalidate();});
  bind('reviewRun',runReview);
}

function bindCanvasSelection(){
  const canvas=$('reviewCanvas');
  if(!canvas)return;
  let start=null,dragging=false;
  const point=e=>{
    const r=canvas.getBoundingClientRect();
    return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),
            Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];
  };
  canvas.onpointerdown=e=>{
    if(!REVIEW.bitmap)return;
    dragging=true;start=point(e);
    canvas.setPointerCapture(e.pointerId);
  };
  canvas.onpointermove=e=>{
    if(!dragging||!start)return;
    const q=point(e);
    REVIEW.bounds=[Math.min(start[0],q[0]),Math.min(start[1],q[1]),Math.max(start[0],q[0]),Math.max(start[1],q[1])];
    reviewPaintCanvas();
  };
  canvas.onpointerup=()=>{
    if(!dragging)return;
    dragging=false;
    const b=REVIEW.bounds;
    if(b[2]-b[0]<.01||b[3]-b[1]<.01){REVIEW.bounds=[.3,.3,.7,.7];}
    reviewPaintCanvas();
    reviewInvalidate();
  };
  canvas.onpointercancel=()=>{dragging=false;};
}

function reviewPatchCard(title,body,foot){
  return `<article class="review-card"><header><b>${esc(title)}</b>${foot||''}</header>${body}</article>`;
}
function reviewRender(){
  const data=REVIEW.payload;
  if(!data)return;
  const entries=data.results;
  const diff=data.difference;
  const metrics=diff.metrics;
  const metricRows=metrics?[['比较像素',metrics.pixels],['有效重叠',(metrics.overlap_fraction*100).toFixed(1)+'%'],['平均绝对差异',metrics.mae.toFixed(5)],['均方根差异',metrics.rmse.toFixed(5)],['差异 P95',metrics.p95_abs.toFixed(5)],['最大差异',metrics.maximum_abs.toFixed(5)],['变化像素比例',(metrics.changed_fraction*100).toFixed(1)+'%'],['亮度相关系数',metrics.luminance_correlation==null?'—':metrics.luminance_correlation.toFixed(4)]]
    .map(([k,v])=>`<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join(''):'';
  const cards=entries.map((entry,index)=>{
    const geometry=entry.correspondence;
    const details=geometry?`<details><summary>输出几何</summary><pre>${esc(JSON.stringify(geometry,null,2))}</pre></details>`:'';
    const warnings=(entry.warnings&&entry.warnings.length)?`<div class="warning">${esc(entry.warnings.join('；'))}</div>`:'';
    const body=entry.patch?`<img src="${entry.patch}" alt="版本 ${esc(entry.id.slice(0,8))} 对应区域">`:`<div class="review-missing">该版本无法提供所选区域，因此未生成对应图。</div>`;
    const vs=entry.vs_reference&&entry.vs_reference.mae!=null?`<p class="tiny">与原始扫描显示差异（参考）：MAE ${entry.vs_reference.mae.toFixed(4)}</p>`:'';
    return reviewPatchCard(`版本 ${index===0?'A':'B'} · ${entry.id.slice(0,12)}`,body,reviewStatusBadge(entry.status))
      +`<div class="review-card-foot"><p class="tiny">${esc(entry.message)}</p>${vs}${warnings}${details}</div>`;
  }).join('');
  const diffCard=reviewPatchCard('差异 A ↔ B',diff.image?`<img src="${diff.image}" alt="差异对照图">`:`<div class="review-missing">${esc(diff.message||'未计算差异')}</div>`,reviewStatusBadge(diff.status));
  $('reviewOutput').innerHTML=`
    <div class="review-grid">
      ${reviewPatchCard('原始参考扫描','<img src="'+data.source.patch+'" alt="原始参考扫描区域">')}
      ${cards}
      ${diffCard}
    </div>
    ${diff.status!=='ok'&&diff.message?`<div class="warning">${esc(diff.message)}</div>`:''}
    ${metricRows?`<div class="review-metrics">${metricRows}</div><p class="tiny">${esc(metrics.note)}</p>`:''}
    <div class="review-provenance"><span>参考帧：${esc(data.source.frame_name)}</span><span>素材摘要：${esc(data.source.asset_id.slice(0,16))}…</span><span>工程修订：${esc(data.project.revision)}</span><span>当前结果未改变</span><button id="reviewExport">↗ 导出复核报告</button></div>`;
  bind('reviewExport',reviewExport);
}

async function runReview(){
  if(!REVIEW.bounds)throw Error('请先框选区域');
  if(!REVIEW.first||!REVIEW.second)throw Error('请选择两份历史结果');
  if(REVIEW.first===REVIEW.second)throw Error('请选择两份不同的历史结果');
  $('reviewOutput').innerHTML='<p class="muted">正在读取历史结果并还原几何…</p>';
  const payload=await api(path('/review'),'POST',{frame_id:REVIEW.frameId,bounds:REVIEW.bounds,result_ids:[REVIEW.first,REVIEW.second],gain:REVIEW.gain});
  REVIEW.payload=payload;
  reviewRender();
}
async function reviewExport(){
  if(!REVIEW.payload)throw Error('请先生成对比');
  const blob=await api(path('/review/report'),'POST',{frame_id:REVIEW.frameId,bounds:REVIEW.bounds,result_ids:[REVIEW.first,REVIEW.second],gain:REVIEW.gain});
  downloadBlob(blob,'scanweave-region-review.html');
  toast('复核报告已下载（HTML，可打印为PDF）');
}
bind('reviewBtn',reviewDialog);
