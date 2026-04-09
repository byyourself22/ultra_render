// UltraRender Web Entry Point

interface WasmModule {
  default: () => Promise<void>;
  get_fps: () => number;
  get_draw_calls: () => number;
  get_sprite_count: () => number;
  get_anim_fps: () => number;
  get_anim_frame: () => number;
  get_anim_total_frames: () => number;
  get_stats_json: () => string;
  request_sprite_count: (n: number) => void;
  set_batch_synced: (synced: boolean) => void;
  get_batch_synced: () => boolean;
  add_sprites_batched: (n: number) => void;
  add_sprites_unbatched: (n: number) => void;
}

async function main() {
  const statFps       = document.getElementById('stat-fps')!;
  const statDraws     = document.getElementById('stat-draws')!;
  const statSprites   = document.getElementById('stat-sprites')!;
  const statAnimFps   = document.getElementById('stat-anim-fps')!;
  const statAnimFrame = document.getElementById('stat-anim-frame')!;
  const statMode      = document.getElementById('stat-mode')!;
  const statStatus    = document.getElementById('stat-status')!;
  const ctrlCount     = document.getElementById('ctrl-count')!;
  const btnLess       = document.getElementById('btn-less')! as HTMLButtonElement;
  const btnMore       = document.getElementById('btn-more')! as HTMLButtonElement;
  const btnBatch10    = document.getElementById('btn-batch-10')! as HTMLButtonElement;
  const btnUnbatch10  = document.getElementById('btn-unbatch-10')! as HTMLButtonElement;
  const btnBatch100   = document.getElementById('btn-batch-100')! as HTMLButtonElement;
  const btnUnbatch100 = document.getElementById('btn-unbatch-100')! as HTMLButtonElement;
  const noWebGPU      = document.getElementById('no-webgpu')!;

  // Ensure canvas pixel dimensions match the window before WASM loads
  const canvas = document.getElementById('ultra-canvas') as HTMLCanvasElement;
  const syncCanvasSize = () => {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  syncCanvasSize();
  window.addEventListener('resize', syncCanvasSize);

  if (!(navigator as any).gpu) {
    noWebGPU.style.display = 'block';
    statStatus.textContent = 'WebGPU not supported';
    return;
  }

  statStatus.textContent = 'Initializing WebGPU…';

  let wasm: WasmModule | null = null;
  let targetSprites = 1;

  try {
    // @ts-ignore
    wasm = await import('../../pkg/ultra_render.js') as WasmModule;
    await wasm.default();
    statStatus.textContent = 'Running';
  } catch (e) {
    console.error('Failed to initialize UltraRender:', e);
    statStatus.textContent = `Error: ${e}`;
    (document.getElementById('stats')! as HTMLElement).style.color = '#f55';
    return;
  }

  // ── Sprite count controls ──────────────────────────────────
  function setTarget(n: number) {
    targetSprites = Math.max(1, Math.min(256, n));
    ctrlCount.textContent = String(targetSprites);
    wasm?.request_sprite_count(targetSprites);
  }

  btnLess.addEventListener('click', () => setTarget(targetSprites - 1));
  btnMore.addEventListener('click', () => setTarget(targetSprites + 1));

  btnBatch10.addEventListener('click', () => {
    wasm?.add_sprites_batched(10);
  });
  btnUnbatch10.addEventListener('click', () => {
    wasm?.add_sprites_unbatched(10);
  });
  btnBatch100.addEventListener('click', () => {
    wasm?.add_sprites_batched(100);
  });
  btnUnbatch100.addEventListener('click', () => {
    wasm?.add_sprites_unbatched(100);
  });

  // ── Stats loop ─────────────────────────────────────────────
  let jsFrames = 0;
  let jsLastTime = performance.now();
  let jsFps = 0;

  function statsLoop(now: number) {
    jsFrames++;
    const elapsed = now - jsLastTime;
    if (elapsed >= 1000) {
      jsFps = Math.round(jsFrames * 1000 / elapsed);
      jsFrames = 0;
      jsLastTime = now;
    }

    if (wasm) {
      const wasmFps = wasm.get_fps();
      const displayFps = wasmFps > 0 ? wasmFps.toFixed(0) : String(jsFps);
      statFps.textContent       = displayFps + ' fps';
      statDraws.textContent     = String(wasm.get_draw_calls());

      const actualSprites = wasm.get_sprite_count();
      statSprites.textContent   = String(actualSprites);
      if (actualSprites > 0) {
        ctrlCount.textContent = String(actualSprites);
        targetSprites = actualSprites;
      }

      const animFps   = wasm.get_anim_fps();
      const animFrame = wasm.get_anim_frame();
      const animTotal = wasm.get_anim_total_frames();
      statAnimFps.textContent   = animFps.toFixed(0) + ' Hz';
      statAnimFrame.textContent = animFrame.toFixed(1) + ' / ' + animTotal.toFixed(0);

      const synced = wasm.get_batch_synced();
      statMode.textContent = synced ? 'batch (synced)' : 'unbatch';
      statMode.style.color = synced ? '#0ff' : '#0f0';
    } else {
      statFps.textContent = String(jsFps) + ' fps';
    }

    requestAnimationFrame(statsLoop);
  }

  requestAnimationFrame(statsLoop);
}

main();
