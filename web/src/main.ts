// UltraRender Web Entry Point

interface WasmModule {
  default: () => Promise<void>;
  get_fps: () => number;
  get_draw_calls: () => number;
  get_sprite_count: () => number;
  get_anim_fps: () => number;
  get_anim_frame: () => number;
  get_anim_total_frames: () => number;
  get_subframes: () => number;
  get_tess_unique: () => number;
  request_sprite_count: (n: number) => void;
  add_sprites: (n: number) => void;
  set_zoom: (zoom: number) => void;
  set_pan: (x: number, y: number) => void;
  get_zoom: () => number;
}

async function main() {
  const sFps     = document.getElementById('s-fps')!;
  const sDraws   = document.getElementById('s-draws')!;
  const sSprites = document.getElementById('s-sprites')!;
  const sAnim    = document.getElementById('s-anim')!;
  const sFrame   = document.getElementById('s-frame')!;
  const sSub     = document.getElementById('s-sub')!;
  const sTess    = document.getElementById('s-tess')!;
  const sZoom    = document.getElementById('s-zoom')!;
  const sStatus  = document.getElementById('s-status')!;
  const cCount   = document.getElementById('c-count')!;
  const noWebGPU = document.getElementById('no-webgpu')!;

  // Sync canvas pixel dimensions before WASM loads
  const canvas = document.getElementById('ultra-canvas') as HTMLCanvasElement;
  const syncSize = () => {
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.round(window.innerWidth * dpr);
    canvas.height = Math.round(window.innerHeight * dpr);
  };
  syncSize();
  window.addEventListener('resize', syncSize);

  if (!(navigator as any).gpu) {
    noWebGPU.style.display = 'block';
    sStatus.textContent = 'WebGPU not supported';
    return;
  }

  sStatus.textContent = 'Initializing WebGPU...';

  let wasm: WasmModule | null = null;
  let target = 1;

  try {
    // @ts-ignore
    wasm = await import('../../pkg/ultra_render.js') as WasmModule;
    await wasm.default();
    sStatus.textContent = 'Running';
  } catch (e) {
    console.error('UltraRender init failed:', e);
    sStatus.textContent = `Error: ${e}`;
    return;
  }

  // -- Sprite controls --
  function setTarget(n: number) {
    target = Math.max(1, n);
    cCount.textContent = String(target);
    wasm?.request_sprite_count(target);
  }

  document.getElementById('b-sub1')!.addEventListener('click', () => setTarget(target - 1));
  document.getElementById('b-add1')!.addEventListener('click', () => setTarget(target + 1));
  document.getElementById('b-10')!.addEventListener('click', () => setTarget(target + 10));
  document.getElementById('b-100')!.addEventListener('click', () => setTarget(target + 100));
  document.getElementById('b-1k')!.addEventListener('click', () => setTarget(target + 1000));
  document.getElementById('b-reset')!.addEventListener('click', () => setTarget(1));

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.key === '=' || e.key === '+') setTarget(target + 1);
    if (e.key === '-') setTarget(target - 1);
    if (e.key === '1') setTarget(1);
    if (e.key === '0') setTarget(target + 100);
    if (e.key === 'r' || e.key === 'R') resetView();
  });

  // -- Zoom / Pan --
  let zoom = 1.0;
  let panX = 0.0;
  let panY = 0.0;
  let isPanning = false;
  let lastMouseX = 0;
  let lastMouseY = 0;

  function applyView() {
    wasm?.set_zoom(zoom);
    wasm?.set_pan(panX, panY);
  }

  function resetView() {
    zoom = 1.0;
    panX = 0.0;
    panY = 0.0;
    applyView();
  }

  function zoomBy(factor: number, centerX?: number, centerY?: number) {
    const newZoom = Math.max(0.1, Math.min(20.0, zoom * factor));
    if (centerX !== undefined && centerY !== undefined) {
      const w = canvas.width;
      const h = canvas.height;
      panX += (centerX - w / 2) * (1 / zoom - 1 / newZoom);
      panY += (centerY - h / 2) * (1 / zoom - 1 / newZoom);
    }
    zoom = newZoom;
    applyView();
  }

  // Mouse wheel → zoom to cursor
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const dpr = canvas.width / rect.width;
    const mx = (e.clientX - rect.left) * dpr;
    const my = (e.clientY - rect.top) * dpr;
    const speed = e.ctrlKey ? 0.01 : 0.003;
    const factor = Math.exp(-e.deltaY * speed);
    zoomBy(factor, mx, my);
  }, { passive: false });

  // Drag to pan
  canvas.addEventListener('mousedown', (e) => {
    isPanning = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
    canvas.style.cursor = 'grabbing';
  });

  window.addEventListener('mousemove', (e) => {
    if (!isPanning) return;
    const dpr = canvas.width / canvas.getBoundingClientRect().width;
    const dx = (e.clientX - lastMouseX) * dpr;
    const dy = (e.clientY - lastMouseY) * dpr;
    panX -= dx / zoom;
    panY -= dy / zoom;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
    applyView();
  });

  window.addEventListener('mouseup', () => {
    if (isPanning) {
      isPanning = false;
      canvas.style.cursor = 'grab';
    }
  });

  // Double-click to reset view
  canvas.addEventListener('dblclick', resetView);

  // View control buttons
  document.getElementById('b-zoom-in')!.addEventListener('click', () => {
    zoomBy(1.3, canvas.width / 2, canvas.height / 2);
  });
  document.getElementById('b-zoom-out')!.addEventListener('click', () => {
    zoomBy(1 / 1.3, canvas.width / 2, canvas.height / 2);
  });
  document.getElementById('b-view-reset')!.addEventListener('click', resetView);

  // -- Stats loop --
  function statsLoop() {
    if (!wasm) { requestAnimationFrame(statsLoop); return; }

    const fps = wasm.get_fps();
    const sprites = wasm.get_sprite_count();

    sFps.textContent     = fps > 0 ? fps.toFixed(0) + ' fps' : '--';
    sDraws.textContent   = String(wasm.get_draw_calls());
    sSprites.textContent = String(sprites);

    if (sprites > 0) {
      cCount.textContent = String(sprites);
      target = sprites;
    }

    const animFps   = wasm.get_anim_fps();
    const animFrame = wasm.get_anim_frame();
    const animTotal = wasm.get_anim_total_frames();
    sAnim.textContent  = animFps.toFixed(0) + ' Hz';
    sFrame.textContent = animFrame.toFixed(1) + ' / ' + animTotal.toFixed(0);
    sSub.textContent   = wasm.get_subframes().toFixed(1) + 'x';
    sTess.textContent  = String(wasm.get_tess_unique());
    sZoom.textContent  = zoom.toFixed(1) + 'x';

    requestAnimationFrame(statsLoop);
  }

  requestAnimationFrame(statsLoop);
}

main();
