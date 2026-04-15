# UltraRender API Reference

Lottie animation renderer powered by WebGPU. Architecture follows Rive Runtime patterns:
zoom is applied as a scale transform on sprites (not camera projection), DPI is handled
via `scaleFactor` in layout computation, and synced sprites share a single GPU tessellation.

---

## WASM Exports

All functions below are exported via `wasm_bindgen` and callable from JavaScript.

### Initialization

The WASM module auto-starts on import via `#[wasm_bindgen(start)]`. It creates a `winit`
event loop, initializes WebGPU, loads the embedded Lottie animation, and begins rendering.

```js
import wasm from '../../pkg/ultra_render.js';
await wasm.default(); // init WebGPU + start render loop
```

---

### Sprites

| Function | Signature | Description |
|----------|-----------|-------------|
| `request_sprite_count` | `(n: u32)` | Set target sprite count (clamped 1..10000). Sprites are added/removed to match. |
| `add_sprites` | `(n: u32)` | Add N sprites to the current count. |

Synced sprites (same animation, same frame) share one GPU tessellation and are drawn
with instanced rendering. Only per-sprite transforms differ.

---

### Playback

| Function | Signature | Description |
|----------|-----------|-------------|
| `play` | `()` | Resume animation playback. |
| `pause` | `()` | Pause animation playback. |
| `is_paused` | `() -> bool` | Check if currently paused. |
| `set_speed` | `(speed: f32)` | Set playback speed multiplier. 1.0 = normal, 0.5 = half, 2.0 = double. Clamped 0.0..10.0. |
| `get_speed` | `() -> f32` | Get current playback speed. |

Speed is applied as a multiplier on the delta time: `canvas.update(dt * speed)`.

---

### View (Zoom / Pan / Fit)

Zoom is applied as a **scale transform on sprites** (Rive-style), not as a camera
projection change. This preserves stroke widths and keeps the rendering pipeline simple.

| Function | Signature | Description |
|----------|-----------|-------------|
| `set_zoom` | `(zoom: f32)` | Set zoom level. 1.0 = default, >1 = closer. Clamped 0.1..20.0. |
| `get_zoom` | `() -> f32` | Get current zoom level. |
| `set_pan` | `(x: f32, y: f32)` | Set pan offset in world pixels. |
| `set_fit` | `(fit: u32)` | Set fit mode (see Fit Modes below). |
| `set_scale_factor` | `(dpr: f32)` | Set DPI scale factor (typically `window.devicePixelRatio`). Clamped >= 0.5. |

#### Fit Modes

| Value | Mode | Description |
|-------|------|-------------|
| 0 | Cover | Scale uniformly to cover the frame (may crop). |
| 1 | Contain | Scale uniformly to fit inside the frame (may letterbox). |
| 2 | Fill | Stretch to fill the frame (may distort). |
| 3 | ScaleDown | Like Contain, but never upscales. |
| 4 | None | No scaling, content rendered at natural size. |

Fit and alignment are computed via `compute_alignment(fit, alignment, frame, content, scale_factor)`,
matching Rive Runtime's layout system. DPI is factored into the alignment transform.

---

### Stats

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_fps` | `() -> f32` | Current FPS (rolling 120-frame window). |
| `get_draw_calls` | `() -> u32` | GPU draw calls last frame. |
| `get_sprite_count` | `() -> u32` | Active sprite count. |
| `get_anim_fps` | `() -> f32` | Animation source frame rate (Hz). |
| `get_anim_frame` | `() -> f32` | Current animation frame (fractional). |
| `get_anim_total_frames` | `() -> f32` | Total animation frames. |
| `get_subframes` | `() -> f32` | Subframe interpolation multiplier. |
| `get_tess_unique` | `() -> u32` | Unique tessellations this frame. |
| `get_stats_json` | `() -> String` | All stats as a JSON string. |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Toggle play/pause |
| `+` / `=` | Add 1 sprite |
| `-` | Remove 1 sprite |
| `0` | Add 100 sprites |
| `1` | Reset to 1 sprite |
| `[` | Decrease speed by 0.25x |
| `]` | Increase speed by 0.25x |
| `R` | Reset zoom/pan to default |

---

## Mouse Controls

| Action | Effect |
|--------|--------|
| Scroll wheel | Zoom to cursor position |
| Ctrl + scroll | Fine zoom (slower) |
| Click + drag | Pan |
| Double-click | Reset view |

---

## Rendering Pipeline

1. **CPU**: Lottie animation advance + artboard update (shape evaluation, keyframe interpolation)
2. **CPU**: Collect draw commands per sprite in local artboard space
3. **CPU**: Encode paths — cubic contours, fill topology (midpoint fan), stroke segments
4. **GPU Compute**: `cs_fill_tessellate` — subdivide cubic curves into vertices (Wang's formula, precision=4.0)
5. **GPU Compute**: `cs_tessellate` — expand stroke segments into triangle strips
6. **GPU Render (stencil pass)**: Accumulate winding numbers via stencil buffer (NonZero or EvenOdd)
7. **GPU Render (cover pass)**: Write color where stencil is non-zero, reset stencil to 0
8. **GPU Render (simple fills)**: Direct indexed draw for convex fills

All fills go through the stencil pipeline for 1:1 fidelity with the Lottie specification.
Synced sprites reuse the single tessellation via GPU instancing (`instance_count > 1`).

---

## DPI Handling

DPI is handled at two levels:

1. **Canvas resolution**: `canvas.width = window.innerWidth * devicePixelRatio`
2. **Layout transform**: `set_scale_factor(dpr)` passes DPI into `compute_alignment`,
   which scales the sprite world transform so vector content renders at native resolution.

This matches Rive Runtime's approach where `scaleFactor` is a parameter to the alignment
computation, not a separate render pass or post-process.
