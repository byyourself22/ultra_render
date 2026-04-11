// UltraRender - Path Drawing Shader (Rive GPU + ThorVG gradient eval)
//
// Per-sprite GPU instancing:
//   Each GpuVertex carries a sprite_index.  The vertex shader reads the
//   corresponding SpriteTransform from a storage buffer and transforms the
//   LOCAL-space position to world space.  This lets N synced sprites share
//   one tessellation with only a per-sprite transform upload each frame.
//
// Gradient coordinates are stored in LOCAL space in the paint buffer;
// the fragment shader transforms them to world space the same way.

// ─── Constants ───────────────────────────────────────────────

const MAX_STOPS: u32 = 16u;
const PAINT_SOLID: u32 = 0u;
const PAINT_LINEAR_GRAD: u32 = 1u;
const PAINT_RADIAL_GRAD: u32 = 2u;

// ─── Uniforms ────────────────────────────────────────────────

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    time: f32,
    is_srgb: u32,
};

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;

// ─── Paint storage buffer ────────────────────────────────────

struct GpuPaint {
    paint_type: u32,
    stop_count: u32,
    opacity: f32,
    _pad: u32,
    grad_start: vec2<f32>,   // LOCAL space
    grad_end:   vec2<f32>,   // LOCAL space
    stops: array<f32, 16>,
    stop_colors: array<vec4<f32>, 16>,
};

@group(1) @binding(0) var<storage, read> paints: array<GpuPaint>;

// ─── Sprite transforms storage buffer ───────────────────────
// One entry per sprite; holds the 2D affine transform that maps
// local artboard space → world (screen) space.
// Layout: [a, b, c, d, tx, ty, _p0, _p1]
// Transform: x' = a*lx + c*ly + tx,  y' = b*lx + d*ly + ty

struct SpriteTransform {
    a:  f32, b:  f32, c:  f32, d:  f32,
    tx: f32, ty: f32, p0: f32, p1: f32,
};

@group(2) @binding(0) var<storage, read> sprite_transforms: array<SpriteTransform>;

// ─── Vertex Input / Output ───────────────────────────────────

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position:     vec2<f32>,
    @location(1) uv:           vec2<f32>,
    @location(2) color:        vec4<f32>,
    @location(3) world_pos:    vec2<f32>,   // ignored; computed in shader
    @location(4) paint_index:  u32,
    @location(5) sprite_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color:        vec4<f32>,
    @location(1) uv:           vec2<f32>,
    @location(2) world_pos:    vec2<f32>,
    @location(3) @interpolate(flat) paint_index:  u32,
    @location(4) @interpolate(flat) sprite_index: u32,
};

// ─── Helpers ─────────────────────────────────────────────────

fn apply_transform(t: SpriteTransform, local: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        t.a * local.x + t.c * local.y + t.tx,
        t.b * local.x + t.d * local.y + t.ty,
    );
}

// ─── Vertex Shader ───────────────────────────────────────────

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // GPU instancing: sprite_index from vertex + instance_index offset.
    // For synced sprites, vertex.sprite_index=0 and instance_index selects the sprite.
    // For unsynced sprites, vertex.sprite_index is already correct and instance_index=0.
    let sid   = in.sprite_index + in.instance_index;
    let t     = sprite_transforms[sid];
    let world = apply_transform(t, in.position);
    out.clip_position = uniforms.view_proj * vec4<f32>(world, 0.0, 1.0);
    out.color        = in.color;
    out.uv           = in.uv;
    out.world_pos    = world;
    out.paint_index  = in.paint_index;
    out.sprite_index = sid;
    return out;
}

// ─── Gradient evaluation ─────────────────────────────────────

fn sample_gradient(paint: GpuPaint, t_raw: f32) -> vec4<f32> {
    let t     = clamp(t_raw, 0.0, 1.0);
    let count = paint.stop_count;

    if (count == 0u) { return vec4<f32>(1.0); }
    if (count == 1u) { return paint.stop_colors[0]; }

    if (t <= paint.stops[0])            { return paint.stop_colors[0]; }
    if (t >= paint.stops[count - 1u])   { return paint.stop_colors[count - 1u]; }

    for (var i: u32 = 1u; i < count; i = i + 1u) {
        if (t <= paint.stops[i]) {
            let s0    = paint.stops[i - 1u];
            let s1    = paint.stops[i];
            let range = s1 - s0;
            var frac  = 0.0;
            if (range > 1e-6) { frac = (t - s0) / range; }
            return mix(paint.stop_colors[i - 1u], paint.stop_colors[i], frac);
        }
    }

    return paint.stop_colors[count - 1u];
}

// ─── sRGB ↔ Linear ───────────────────────────────────────────

fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) { return c / 12.92; }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear3(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b));
}

// ─── Fragment Shader ─────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let paint = paints[in.paint_index];
    let t     = sprite_transforms[in.sprite_index];
    var color: vec4<f32>;

    if (paint.paint_type == PAINT_LINEAR_GRAD) {
        // Transform LOCAL-space gradient endpoints to world space
        let ws = apply_transform(t, paint.grad_start);
        let we = apply_transform(t, paint.grad_end);
        let d      = we - ws;
        let len_sq = dot(d, d);
        var tt = 0.0;
        if (len_sq > 1e-8) { tt = dot(in.world_pos - ws, d) / len_sq; }
        color    = sample_gradient(paint, tt);
        color.a *= paint.opacity;

    } else if (paint.paint_type == PAINT_RADIAL_GRAD) {
        let ws     = apply_transform(t, paint.grad_start);
        let we     = apply_transform(t, paint.grad_end);
        let radius = length(we - ws);
        var tt = 0.0;
        if (radius > 1e-6) { tt = length(in.world_pos - ws) / radius; }
        color    = sample_gradient(paint, tt);
        color.a *= paint.opacity;

    } else {
        color = in.color;
    }

    // sRGB surface: output linear so GPU converts linear→sRGB on write.
    // Non-sRGB surface: output sRGB directly (ThorVG-compatible).
    if (uniforms.is_srgb != 0u) {
        color = vec4<f32>(srgb_to_linear3(color.rgb), color.a);
    }

    let stroke_edge_dist = min(in.uv.x, 1.0 - in.uv.x);
    let stroke_aa_width = max(fwidth(in.uv.x), 1e-4) * 1.5;
    let stroke_coverage = smoothstep(0.0, stroke_aa_width, stroke_edge_dist);
    if (in.uv.y > 0.5) {
        color.a *= stroke_coverage;
    }

    // Premultiply alpha for correct blending
    color = vec4<f32>(color.rgb * color.a, color.a);

    return color;
}

@fragment
fn fs_stencil(_in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
