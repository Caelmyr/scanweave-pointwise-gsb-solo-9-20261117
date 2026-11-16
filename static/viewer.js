'use strict';
/* Tiled browsing canvas.
   - Draws a window-sized preview first, so switching between original and
     composite never waits for the full image.
   - When zoomed, loads 256px detail tiles for the visible region only and
     composites them over the preview as they arrive.
   - Both previews and tiles live in one bounded LRU keyed by result token,
     so revisiting an image is instant without unbounded growth.
   Quality checks and exports still read the exact stored result. */
const Viewer=(()=>{
  const $=id=>document.getElementById(id);
  const canvas=$('hero'),stage=$('photoStage'),ctx=canvas.getContext('2d');
  const TILE=256,BUDGET=64*1024*1024,MAX_PARALLEL=4,PREVIEW_CAP=2048;
  const cache=new Map();            // key -> {canvas,bytes,used}
  const inflight=new Map();         // url -> Promise<canvas>
  const requested=new Set();        // tiles requested for the current token
  let cacheBytes=0,session=null,active=0,queue=[],raf=0,resizeTimer=0,pan=null;

  const lruGet=key=>{const item=cache.get(key);if(!item)return null;item.used=performance.now();cache.delete(key);cache.set(key,item);return item.canvas;};
  function lruPut(key,item){
    const bytes=item.canvas.width*item.canvas.height*4;item.bytes=bytes;item.used=performance.now();
    const old=cache.get(key);if(old){cacheBytes-=old.bytes;cache.delete(key);}
    if(bytes>BUDGET)return;
    cache.set(key,item);cacheBytes+=bytes;
    while(cacheBytes>BUDGET){
      let victim=null;
      // Keep the preview currently on screen; everything else is LRU order.
      for(const k of cache.keys()){if(session&&k===session.preview)continue;victim=k;break;}
      if(victim===null)break;
      const evicted=cache.get(victim);cacheBytes-=evicted.bytes;cache.delete(victim);
      const parts=victim.split('|');
      if(parts[1]==='tile')requested.delete(parts[2]+':'+parts[3]+':'+parts[4]);
    }
  }
  function cacheReport(){return {entries:cache.size,bytes:cacheBytes,budget:BUDGET};}

  const metrics=()=>{
    const rect=canvas.getBoundingClientRect(),dpr=Math.min(2,window.devicePixelRatio||1);
    const cssW=Math.max(1,rect.width),cssH=Math.max(1,rect.height);
    const fitScale=Math.min(cssW/session.info.width,cssH/session.info.height);
    const scale=session.fit?fitScale:session.scale;
    const drawW=session.info.width*scale,drawH=session.info.height*scale;
    return {cssW,cssH,dpr,fitScale,scale,drawW,drawH,originX:(cssW-drawW)/2+session.offsetX,originY:(cssH-drawH)/2+session.offsetY};
  };
  const apiRoot=suffix=>'/api/projects/'+session.key+suffix;
  const query=extra=>{const q=new URLSearchParams({view:session.info.view,token:session.info.token,result_id:session.info.result_id||'',frame_id:session.info.frame_id||''});for(const [k,v] of Object.entries(extra))q.set(k,v);return q;};

  function clampPan(){
    const d=metrics();
    if(session.fit){session.offsetX=session.offsetY=0;stage.classList.remove('zoomed');return;}
    const maxX=Math.max(0,(d.drawW-d.cssW)/2+d.cssW*.05),maxY=Math.max(0,(d.drawH-d.cssH)/2+d.cssH*.05);
    session.offsetX=Math.max(-maxX,Math.min(maxX,session.offsetX));
    session.offsetY=Math.max(-maxY,Math.min(maxY,session.offsetY));
  }
  // Detail tiles always come from level 0 (native pixels). Coarser levels are
  // only useful for the whole-image preview, which is generated separately.
  const detailLevel=()=>0;
  function levelSize(level){return [Math.max(1,Math.floor(session.info.width/Math.pow(2,level))),Math.max(1,Math.floor(session.info.height/Math.pow(2,level)))];}

  function draw(){
    raf=0;
    if(!session||!session.info.available)return;
    const d=metrics(),dpr=d.dpr;
    const wantW=Math.max(1,Math.round(d.cssW*dpr)),wantH=Math.max(1,Math.round(d.cssH*dpr));
    if(canvas.width!==wantW||canvas.height!==wantH){canvas.width=wantW;canvas.height=wantH;}
    ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);
    const x=d.originX*dpr,y=d.originY*dpr,w=d.drawW*dpr,h=d.drawH*dpr;
    const preview=session.preview&&lruGet(session.preview);
    ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';
    if(session.thumb&&session.thumb.complete&&session.thumb.naturalWidth)ctx.drawImage(session.thumb,x,y,w,h);
    if(preview)ctx.drawImage(preview,x,y,w,h);
    if(!session.fit){
      const level=detailLevel(),dest=session.scale*dpr;
      if(dest>=.999){
        // Native pixels stay crisp at 1:1; beyond 2x nearest keeps pixel edges.
        ctx.imageSmoothingEnabled=dest<2;
        ctx.save();ctx.beginPath();ctx.rect(x,y,w,h);ctx.clip();
        for(const [key,item] of cache){
          const parts=key.split('|');
          if(parts[0]!==session.info.token||parts[1]!=='tile')continue;
          if(Number(parts[2])!==level)continue;
          const tx=Number(parts[3]),ty=Number(parts[4]);
          const dx=x+tx*dest,dy=y+ty*dest;
          if(dx>canvas.width||dy>canvas.height||dx+item.canvas.width*dest<0||dy+item.canvas.height*dest<0)continue;
          ctx.drawImage(item.canvas,dx,dy,item.canvas.width*dest,item.canvas.height*dest);
        }
        ctx.restore();
      }
    }
    $('browseLevel').textContent=session.info.available&&!session.fit?'DETAIL ×'+Math.round(d.scale*100)+'%':'';
  }
  const schedule=()=>{if(!raf)raf=requestAnimationFrame(draw);};

  function fetchCanvas(url){
    if(inflight.has(url))return inflight.get(url);
    const promise=fetch(url).then(async response=>{
      if(!response.ok){let message='浏览请求失败';try{message=(await response.json()).error||message;}catch{}throw Error(message);}
      const bitmap=await createImageBitmap(await response.blob());
      const off=document.createElement('canvas');off.width=bitmap.width;off.height=bitmap.height;
      off.getContext('2d').drawImage(bitmap,0,0);bitmap.close&&bitmap.close();
      return off;
    }).finally(()=>inflight.delete(url));
    inflight.set(url,promise);return promise;
  }
  async function fetchInto(key,url,gen){
    if(lruGet(key))return;
    const off=await fetchCanvas(url);
    // Tokens are content-addressed (result id or asset hash), so a late
    // response is still valid for its own token and worth keeping.
    lruPut(key,{canvas:off});
    if(session&&session.generation===gen)schedule();
  }

  function visibleTiles(level){
    const d=metrics(),[lw,lh]=levelSize(level);
    const dest=session.scale*d.dpr;
    if(dest<.999||!lw||!lh)return;
    const x0=Math.max(0,Math.floor((-d.originX*d.dpr)/dest)-1),y0=Math.max(0,Math.floor((-d.originY*d.dpr)/dest)-1);
    const x1=Math.min(lw,Math.ceil((canvas.width-d.originX*d.dpr)/dest)+1),y1=Math.min(lh,Math.ceil((canvas.height-d.originY*d.dpr)/dest)+1);
    const gen=session.generation;
    for(let ty=Math.floor(y0/TILE)*TILE;ty<y1;ty+=TILE){
      for(let tx=Math.floor(x0/TILE)*TILE;tx<x1;tx+=TILE){
        const slot=level+':'+tx+':'+ty;
        if(requested.has(slot))continue;
        requested.add(slot);
        const key=session.info.token+'|tile|'+level+'|'+tx+'|'+ty;
        if(lruGet(key)){schedule();continue;}
        queue.push({key,url:apiRoot('/browse/tile')+'?'+query({level,x:tx,y:ty,size:TILE}).toString(),gen});
      }
    }
    pump();
  }
  function pump(){
    while(active<MAX_PARALLEL&&queue.length){
      const job=queue.shift();
      if(session&&job.gen!==session.generation)continue;
      active+=1;
      fetchInto(job.key,job.url,job.gen).catch(error=>{if(!/已更新/.test(error.message))console.warn(error.message);}).finally(()=>{active-=1;pump();});
    }
  }

  async function loadPreview(){
    const d=metrics();
    // Fit the preview request inside PREVIEW_CAP without changing aspect.
    const want=d.drawW*d.dpr, wantH=d.drawH*d.dpr;
    const shrink=Math.min(1,PREVIEW_CAP/Math.max(1,want),PREVIEW_CAP/Math.max(1,wantH));
    const width=Math.max(1,Math.min(session.info.width,Math.round(want*shrink)));
    const height=Math.max(1,Math.min(session.info.height,Math.round(wantH*shrink)));
    const key=session.info.token+'|preview|'+width+'x'+height;
    const gen=session.generation;
    session.preview=key;
    const hit=lruGet(key);
    if(hit){schedule();return;}
    const busy=fetchCanvas(apiRoot('/browse/preview')+'?'+query({width,height}).toString());
    setHint('正在载入适配窗口的预览…');
    try{
      const off=await busy;
      lruPut(key,{canvas:off});
      if(session&&session.generation===gen){
        setHint('预览已显示 · 滚轮放大加载细节');
        schedule();
      }
    }catch(error){if(session&&session.generation===gen)setHint(error.message);}
  }
  function setHint(text){const el=$('zoomHint');if(el)el.textContent=text;}

  function updateView(){
    if(!session||!session.info.available)return;
    if(!session.fit){
      const level=detailLevel();
      const dest=session.scale*metrics().dpr;
      if(dest>=1)visibleTiles(level);
    }
    schedule();
  }
  function zoom(factor,focus){
    if(!session)return;
    const d=metrics();
    const next=Math.max(d.fitScale*.5,Math.min(8,d.scale*factor));
    const before=session.scale;
    const ratio=next/before;
    session.scale=next;
    session.fit=next<=d.fitScale*1.001;
    if(!session.fit&&focus&&Math.abs(next-before)>1e-6){
      const rect=canvas.getBoundingClientRect();
      const cx=focus.x-rect.left,cy=focus.y-rect.top;
      session.offsetX=(d.cssW/2-cx)*(ratio-1)+session.offsetX*ratio;
      session.offsetY=(d.cssH/2-cy)*(ratio-1)+session.offsetY*ratio;
    }
    if(session.fit){session.offsetX=session.offsetY=0;stage.classList.remove('zoomed');}
    else{stage.classList.add('zoomed');clampPan();}
    updateView();
  }
  function fit(){if(!session)return;session.fit=true;session.scale=1;session.offsetX=session.offsetY=0;stage.classList.remove('zoomed');updateView();}
  function actual(){if(!session)return;session.fit=false;session.scale=1;session.offsetX=session.offsetY=0;stage.classList.add('zoomed');clampPan();updateView();}
  function zoomIn(){if(session)zoom(1.5,null);}
  function zoomOut(){if(session)zoom(1/1.5,null);}

  function show(key,info){
    queue=[];requested.clear();active=0;
    if(!info.available){
      session={key,info,generation:(session?session.generation+1:1),fit:true,scale:1,offsetX:0,offsetY:0,preview:null};
      canvas.hidden=true;$('emptyState').hidden=false;$('browseLevel').textContent='';
      $('imageDimensions').textContent=info.reason||'—';
      return;
    }
    const same=session&&session.key===key&&session.info.token===info.token;
    const previous=session;
    session={key,info,generation:(same?previous.generation:previous?previous.generation+1:1),
      fit:same?previous.fit:true,scale:same?previous.scale:1,offsetX:same?previous.offsetX:0,offsetY:same?previous.offsetY:0,
      preview:same?previous.preview:null,thumb:same?previous.thumb:null};
    if(!same){stage.classList.remove('zoomed','panning');pan=null;}
    else stage.classList.toggle('zoomed',!session.fit);
    canvas.hidden=false;$('emptyState').hidden=true;
    canvas.width=Math.max(1,Math.round(canvas.getBoundingClientRect().width*(Math.min(2,window.devicePixelRatio||1))));
    canvas.height=Math.max(1,Math.round(canvas.getBoundingClientRect().height*(Math.min(2,window.devicePixelRatio||1))));
    if(!same){
      // The stored thumbnail shows instantly while the window-sized preview is
      // generated; it is intentionally small so switching views stays cheap.
      const thumb=new Image();
      thumb.onload=()=>schedule();
      thumb.src=info.thumb_url;
      session.thumb=thumb;
    }
    loadPreview();schedule();
  }

  canvas.addEventListener('wheel',event=>{
    if(!session)return;event.preventDefault();
    zoom(event.deltaY<0?1.25:1/1.25,{x:event.clientX,y:event.clientY});
  },{passive:false});
  canvas.addEventListener('dblclick',event=>{
    if(!session)return;
    if(!session.fit||session.scale>1.2)fit();else zoom(2,{x:event.clientX,y:event.clientY});
  });
  let panMoved=false;
  canvas.addEventListener('pointerdown',event=>{
    if(!session||session.fit)return;
    pan={x:event.clientX,y:event.clientY};panMoved=false;stage.classList.add('panning');canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove',event=>{
    if(!pan)return;
    session.offsetX+=event.clientX-pan.x;session.offsetY+=event.clientY-pan.y;
    pan={x:event.clientX,y:event.clientY};panMoved=true;clampPan();updateView();
  });
  const endPan=()=>{
    if(!pan)return;pan=null;stage.classList.remove('panning');
    if(!panMoved)refreshTiles();else updateView();
  };
  canvas.addEventListener('pointerup',endPan);canvas.addEventListener('pointercancel',endPan);
  window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(!session)return;if(session.fit){loadPreview();updateView();}else{clampPan();updateView();}},180);});

  function refreshTiles(){
    if(!session||session.fit)return;
    requested.clear();
    visibleTiles(detailLevel());
    schedule();
  }

  return {show,zoomIn,zoomOut,fit,actual,cacheReport,refreshTiles,refreshPreview:loadPreview};
})();
