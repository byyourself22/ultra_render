# UltraRender - High-Performance Lottie WebGPU Renderer in Rust

## Objetivo

Renderizador de animacoes Lottie/JSON em Rust usando WebGPU, com arquitetura inspirada no **Rive Runtime** (GPU pipeline, draw batching, tessellation) e no **ThorVG** (parsing Lottie, scene tree, efeitos visuais). Canvas unica para multiplos sprites, 100+ FPS, state machine.

---

## Arquitetura (baseada em Rive + ThorVG)

### 1. Lottie Parser (inspirado em ThorVG `tvgLottieParser/tvgLottieModel`)

**O que usaremos do ThorVG:**
- Modelo de dados Lottie: `LottieObject` hierarquia (Composition > Layer > Group > Shape)
- Tipos de objetos: `SolidFill`, `SolidStroke`, `GradientFill`, `GradientStroke`, `Rect`, `Ellipse`, `Path`, `Polystar`, `Trimpath`, `Repeater`, `RoundedCorner`, `OffsetPath`, `PuckerBloat`
- Efeitos: `DropShadow`, `GaussianBlur`, `Tint`, `Tritone`, `Fill`, `Stroke`
- Mascaras: `LottieMask` com metodos (Add, Subtract, Intersect, Difference)
- Propriedades animaveis: `LottieFloat`, `LottieColor`, `LottieVector`, `LottieOpacity`, `LottiePathSet`
- Interpoladores (easing): cubic bezier, hold, linear (tvgLottieInterpolator)
- Texto: `LottieTextRange`, `LottieGlyph`, `TextDocument`

**Nosso modulo `lottie/`:**
```
lottie/
  mod.rs          - re-exports
  parser.rs       - JSON parsing (serde_json)
  model.rs        - LottieComposition, Layer, Shape, etc.
  property.rs     - Animated properties with keyframes
  interpolator.rs - Easing/bezier interpolation
  effects.rs      - DropShadow, GaussianBlur, Tint, etc.
  modifiers.rs    - TrimPath, RoundedCorner, OffsetPath, PuckerBloat, Repeater
```

### 2. Scene Graph / Animation Engine

**O que usaremos do Rive:**
- Artboard como container principal de uma animacao
- StateMachine para controle de estados, transicoes, inputs
- LinearAnimation com keyframes
- Play/Stop/Pause controle por sprite

**O que usaremos do ThorVG:**
- Arvore de layers com pre-composicoes
- Blend modes por layer (Normal, Multiply, Screen, Add, etc.)
- Matte/Track matte (Alpha, Luma, etc.)
- Transform hierarchy (position, anchor, scale, rotation, opacity)

**Nosso modulo `scene/`:**
```
scene/
  mod.rs          - re-exports
  composition.rs  - Root composition container
  layer.rs        - Layer types (Shape, Precomp, Solid, Image, Null, Text)
  transform.rs    - Animated transforms (Mat2D, position, scale, rotation, anchor)
  animation.rs    - Animation playback, time, frame management
  state_machine.rs - State machine com inputs, transicoes, estados
  sprite.rs       - Sprite instance (posicao na canvas, animacao, state)
```

### 3. Geometry Pipeline (inspirado em Rive `renderer/`)

**O que usaremos do Rive:**
- Tessellation de cubicas via shaders (Wang's formula para contagem de segmentos)
- Merge de segmentos parametricos + polares para curvas suaves
- De Casteljau para avaliacao de pontos em cubicas
- Contour/Path buffer architecture (pathBuffer, contourBuffer)
- Stroke expansion com caps (Round, Butt, Square) e joins (Round, Miter, Bevel)
- Fill rules: EvenOdd e NonZero (Winding)
- Interior triangulation para preenchimento
- "Retrofitted triangles" para otimizacao de geometria simples
- Feather (anti-aliasing suave em bordas)

**O que usaremos do ThorVG:**
- Conversao de shapes Lottie em paths (Rect, Ellipse, Polystar -> bezier paths)
- PathSet com pts/cmds (MoveTo, LineTo, CubicTo, Close)

**Nosso modulo `geometry/`:**
```
geometry/
  mod.rs          - re-exports
  path.rs         - RawPath com commands (move, line, cubic, close)
  tessellation.rs - GPU tessellation de cubicas (Wang's formula)
  stroke.rs       - Stroke expansion, caps, joins
  fill.rs         - Fill triangulation (even-odd, winding)
  shapes.rs       - Lottie shapes -> bezier paths
  math.rs         - Vec2D, Mat2D, AABB, bezier utilities
```

### 4. GPU Renderer (inspirado em Rive PLS/WebGPU)

**O que usaremos do Rive:**
- Draw batching: agrupar draws similares para reduzir draw calls
- Draw types: `PathDraw`, `ImageRectDraw`, `ImageMeshDraw`
- Render pipeline: beginFrame -> draws -> flush
- Tessellation texture: resultados de tessellacao armazenados em textura
- Gradient library: ramp de cores em textura
- Clip system: clipID + clip rects com inverse matrix
- Blend modes via shader
- Coverage/stencil para fills complexos
- Uniforms: FlushUniforms, PathData, ContourData

**Shaders WGSL que precisaremos:**
1. `tessellate.wgsl` - Tessellation de cubicas Bezier na GPU
2. `draw_path.wgsl` - Vertex/Fragment shader para paths (fill + stroke)
3. `gradient.wgsl` - Gradient ramp generation
4. `blend.wgsl` - Advanced blend modes
5. `blur.wgsl` - Gaussian blur (efeito)
6. `shadow.wgsl` - Drop shadow (efeito)
7. `blit.wgsl` - Blit/composite final

**Nosso modulo `renderer/`:**
```
renderer/
  mod.rs           - re-exports, Renderer trait
  context.rs       - RenderContext (GPU resources, frame management)
  canvas.rs        - Canvas unica para multiplos sprites
  draw.rs          - Draw commands, batching
  pipeline.rs      - WebGPU render pipelines
  buffers.rs       - GPU buffers (vertex, index, uniform, storage)
  textures.rs      - Texture management (tessellation, gradients, atlas)
  shaders/
    tessellate.wgsl
    draw_path.wgsl
    gradient.wgsl
    blend.wgsl
    effects.wgsl
    blit.wgsl
```

### 5. Canvas / Sprite Engine

**Inspirado em Rive Artboard + ThorVG Canvas:**
- Canvas unica renderiza N sprites simultaneamente
- Cada sprite tem: posicao, escala, rotacao, animacao, estado
- Frustum culling por sprite (nao renderiza sprites fora da view)
- Z-ordering para sobreposicao correta
- Batch rendering: sprites com mesma animacao compartilham recursos

```
engine/
  mod.rs           - re-exports
  canvas.rs        - UltraCanvas com multi-sprite support
  sprite.rs        - Sprite instances com transform + animation state
  batch.rs         - Sprite batching por animacao/material
  culling.rs       - View frustum culling
```

---

## Dependencias Rust

```toml
[dependencies]
wgpu = "24"            # WebGPU abstraction
winit = "0.30"          # Windowing
serde = { version = "1", features = ["derive"] }
serde_json = "1"        # Lottie JSON parsing
bytemuck = { version = "1", features = ["derive"] }
glam = "0.29"           # Math (Vec2, Mat2, Mat4)
pollster = "0.4"        # Async runtime for wgpu
log = "0.4"
env_logger = "0.11"
zip = "2"               # .lottie files (ZIP format)
lyon = "1"              # CPU tessellation fallback
image = "0.25"          # Image loading for assets
flate2 = "1"            # Decompression
```

---

## Pipeline de Rendering (por frame)

```
1. UPDATE PHASE
   - Advance animation time
   - Evaluate keyframes + interpolation
   - Update transforms (hierarchy)
   - Apply modifiers (trim, round corners, etc.)
   - Apply effects (blur, shadow)
   - Resolve shapes -> paths

2. GEOMETRY PHASE
   - Convert paths to GPU-ready geometry
   - Tessellate curves (Wang's formula)
   - Generate stroke geometry
   - Triangulate fills
   - Upload to GPU buffers

3. DRAW PHASE (batched)
   - Sort draws by z-order, blend mode, texture
   - Batch compatible draws
   - Set uniforms (transforms, colors, gradients)
   - Execute draw calls:
     a. Gradient ramp generation
     b. Path fills (stencil + cover or direct)
     c. Path strokes
     d. Image draws
     e. Effects (blur, shadow as post-process)
     f. Blend/composite

4. PRESENT
   - Final composite to swapchain
```

---

## Features Completas

### Shapes (do ThorVG LottieObject types)
- [x] Rectangle (com rounded corners)
- [x] Ellipse
- [x] Path (bezier paths arbitrarios)
- [x] Polystar (star/polygon)
- [x] Group (container)

### Paint
- [x] Solid Fill (color + opacity)
- [x] Solid Stroke (color + width + opacity)
- [x] Gradient Fill (linear + radial)
- [x] Gradient Stroke
- [x] Blend modes (Normal, Multiply, Screen, Overlay, Add, etc.)

### Modifiers
- [x] Trim Path (start, end, offset)
- [x] Rounded Corners
- [x] Offset Path
- [x] Pucker & Bloat
- [x] Repeater

### Efeitos
- [x] Gaussian Blur
- [x] Drop Shadow (color, angle, distance, blur)
- [x] Tint
- [x] Tritone
- [x] Fill effect
- [x] Stroke effect

### Animacao
- [x] Keyframes com easing (cubic bezier, linear, hold)
- [x] Multi-dimensional keyframes (position, scale)
- [x] Time remapping
- [x] Pre-compositions
- [x] Masks (add, subtract, intersect, difference)
- [x] Mattes (alpha, luma)
- [x] Expressions (basic)

### Engine
- [x] Multi-sprite canvas
- [x] Play / Stop / Pause / Seek
- [x] State Machine (inputs, transitions)
- [x] 100+ FPS target
- [x] Draw batching (Rive-style)
- [x] GPU tessellation (Rive-style)

---

## Estrutura de Arquivos Final

```
ultra_render/
  Cargo.toml
  src/
    main.rs              - Entry point, window, event loop
    lib.rs               - Public API
    lottie/              - Lottie format parsing
    scene/               - Scene graph, animation, state machine
    geometry/            - Paths, tessellation, stroke, fill
    renderer/            - WebGPU GPU renderer
      shaders/           - WGSL shaders
    engine/              - Canvas, sprites, batching
  animations/            - Test animation files
    json/
    lottie/
```

---

## O que NAO usaremos

- **Rive .riv format** - Apenas Lottie/JSON
- **Rive C++ runtime** - Reimplementamos em Rust
- **ThorVG C++ renderer** - Usamos apenas como referencia para parsing
- **ThorVG SVG loader** - Fora de escopo
- **Rive text layout engine** - Simplificado para Lottie text
- **Rive audio** - Fora de escopo

---

## Como Rodar

```bash
cargo build
cargo run -- animations/json/gfunny.json
cargo test
```
