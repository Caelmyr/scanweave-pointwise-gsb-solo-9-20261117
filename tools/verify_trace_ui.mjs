// 通过 Chrome DevTools Protocol 驱动真实浏览器，端到端验证像素溯源 UI（仅用 Node 内置 WebSocket）。
const BASE = 'http://127.0.0.1:8912';
const PROJECT = 'f0639f7f0b234027afca946413c51d95';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getWsUrl() {
  const res = await fetch('http://127.0.0.1:9222/json');
  const tabs = await res.json();
  return tabs[0].webSocketDebuggerUrl;
}

class CDP {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.id = 0;
    this.pending = new Map();
    this.ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      }
    });
    return new Promise((resolve, reject) => {
      this.ws.addEventListener('open', () => resolve(this));
      this.ws.addEventListener('error', reject);
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
  async eval(expression) {
    const r = await this.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  }
  close() { this.ws.close(); }
}

async function main() {
  const cdp = await new CDP(await getWsUrl());
  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');
  await cdp.send('Page.navigate', { url: BASE + '/' });
  await sleep(2500);
  await cdp.eval(`localStorage.setItem('stacklightProject','${PROJECT}')`);
  await cdp.send('Page.navigate', { url: BASE + '/' });
  await sleep(4500);

  const state = await cdp.eval("JSON.stringify({hasProject:!!S.project, cur:S.project&&S.project.current_result, view:S.view})");
  console.log('initial state:', state);

  if (!(await cdp.eval('!!(S.project && S.project.current_result)'))) {
    console.log('无当前结果，触发合成…');
    await cdp.eval("(async()=>{await startJob('process');})()");
    for (let i = 0; i < 40; i++) {
      await sleep(1000);
      if (await cdp.eval('!!(S.project && S.project.current_result)')) break;
    }
  }
  console.log('result ready:', await cdp.eval('S.project.current_result'));

  await cdp.eval("document.getElementById('traceBtn').click()");
  await sleep(400);
  const active = await cdp.eval("document.getElementById('traceBtn').classList.contains('trace-on')");
  console.log('trace mode active:', active);
  if (active !== true) throw new Error('溯源按钮应进入激活态');

  const clicked = await cdp.eval(`(()=>{
    const img=document.getElementById('hero');
    const r=img.getBoundingClientRect();
    img.dispatchEvent(new MouseEvent('click',{clientX:r.left+r.width/2, clientY:r.top+r.height/2, bubbles:true}));
    return true;
  })()`);
  console.log('dispatched click:', clicked);
  await sleep(2500);

  const dialogOpen = await cdp.eval("document.getElementById('dialog').open");
  console.log('dialog open:', dialogOpen);
  if (!dialogOpen) throw new Error('点击后应弹出溯源对话框');

  const frameCards = await cdp.eval("document.querySelectorAll('.trace-frame').length");
  const statuses = await cdp.eval("Array.from(document.querySelectorAll('.trace-status')).map(e=>e.textContent)");
  const marker = await cdp.eval("!!document.querySelector('.trace-marker')");
  const links = await cdp.eval("Array.from(document.querySelectorAll('.trace-links a')).map(a=>a.textContent.trim())");
  const summary = await cdp.eval("document.querySelector('.trace-summary').innerText.slice(0,260)");
  console.log('frame cards:', frameCards);
  console.log('statuses:', JSON.stringify(statuses));
  console.log('marker shown:', marker);
  console.log('links:', JSON.stringify(links.slice(0, 4)));
  console.log('summary:\n' + summary);

  if (!(frameCards >= 1)) throw new Error('应至少渲染一帧贡献卡片');
  if (marker !== true) throw new Error('结果图上应显示取点标记');
  if (!links.some((x) => x.includes('对齐帧')) || !links.some((x) => x.includes('原帧'))) {
    throw new Error('应提供对齐帧与原帧入口');
  }

  // 停用态切换应清除标记
  await cdp.eval("document.getElementById('traceBtn').click()");
  await sleep(300);
  const markerCleared = await cdp.eval("!document.querySelector('.trace-marker')");
  console.log('marker cleared on toggle off:', markerCleared);
  if (!markerCleared) throw new Error('关闭溯源后应清除标记');

  cdp.close();
  console.log('\nUI 端到端验证通过 ✔');
}

main().catch((e) => { console.error('UI 测试失败:', e.message); process.exit(1); });
