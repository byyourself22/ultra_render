# UltraRender

High-performance **Lottie renderer** written in Rust and powered by **WebGPU**.

UltraRender is designed for rendering many animated Lottie/JSON objects efficiently on a single GPU canvas, with a lightweight architecture focused on games, interactive interfaces and real-time applications.

## Goals

* Rust + WebGPU
* Lottie / JSON animation support
* Single canvas for multiple sprites
* GPU draw batching
* GPU-friendly tessellation
* Scene tree
* Visual effects
* State machines
* Native + WebAssembly targets
* High sprite count
* 100+ FPS target

## Architecture

UltraRender takes inspiration from two mature rendering architectures:

**skia Runtime**

* GPU rendering pipeline
* Draw batching
* Tessellation
* Efficient animation runtime

**ThorVG**

* Lottie parsing
* Scene tree
* Shapes and paths
* Masks and clipping
* Visual effects

```text
Lottie JSON
    │
    ▼
  Parser
    │
    ▼
Scene Tree
    │
    ▼
Animation Runtime
    │
    ▼
Tessellation
    │
    ▼
Draw Batcher
    │
    ▼
WebGPU Renderer
    │
    ▼
   Canvas
```

## Example

```rust
let mut renderer = UltraRenderer::new(&device, &queue);

let animation = renderer.load("character.json")?;

let sprite = renderer.spawn(animation);

sprite.set_position(320.0, 180.0);
sprite.play("idle");

renderer.render();
```

## State Machine

```rust
sprite.set_bool("running", true);
sprite.set_number("speed", 2.0);
sprite.fire("attack");
```

State machines are evaluated by the runtime while rendering remains fully GPU-oriented.

## Performance

UltraRender is designed around a shared rendering context instead of creating an isolated renderer for every animation.

```text
1 WebGPU Device
1 Canvas
1 Render Pipeline

        │
        ├── Sprite
        ├── Sprite
        ├── Sprite
        ├── Sprite
        └── ...
```

Animations are grouped and submitted through batched GPU draw calls whenever possible.

The target is smooth rendering of large numbers of animated objects at **100+ FPS**, depending on animation complexity and hardware.

## Targets

```text
Native
├── Windows
├── Linux
└── macOS

Web
└── WebAssembly + WebGPU
```

## Status

🚧 **Experimental**

UltraRender is currently under active development.

Planned areas include:

* Lottie parser
* Shapes and paths
* Gradients
* Masks
* Clipping
* Trim paths
* Blend modes
* Text
* Images
* Effects
* Draw batching
* State machines
* WASM bindings

## License

TBD
