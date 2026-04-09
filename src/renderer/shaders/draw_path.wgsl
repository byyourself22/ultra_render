// UltraRender - Path Drawing Shader (Rive GPU + ThorVG gradient eval)
//
// Renders filled and stroked paths with per-fragment gradient evaluation.
// Reads tessellated vertex data and paint descriptors from storage buffer.

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
    _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;

// ─── Paint storage buffer (ThorVG-style gradient data) ───────

struct GpuPaint {
    paint_type: u32,
    stop_count: u32,
    opacity: f32,
    _pad: u32,
    grad_start: vec2<f32>,
    grad_end: vec2<f32>,
    stops: array<f32, 16>,
    stop_colors: array<vec4<f32>, 16>,
};

@group(1) @binding(0) var<storage, read> paints: array<GpuPaint>;

// ─── Vertex Input/Output ─────────────────────────────────────

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) world_pos: vec2<f32>,
    @location(4) paint_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec2<f32>,
    @location(3) @interpolate(flat) paint_index: u32,
};

// ─── Vertex Shader ───────────────────────────────────────────

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    out.uv = in.uv;
    out.world_pos = in.world_pos;
    out.paint_index = in.paint_index;
    return out;
}

// ─── Gradient evaluation (ThorVG-style, in fragment shader) ──

fn sample_gradient(paint: GpuPaint, t_raw: f32) -> vec4<f32> {
    let t = clamp(t_raw, 0.0, 1.0);
    let count = paint.stop_count;

    if (count == 0u) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    if (count == 1u) {
        return paint.stop_colors[0];
    }

    // Find the two stops surrounding t
    if (t <= paint.stops[0]) {
        return paint.stop_colors[0];
    }
    if (t >= paint.stops[count - 1u]) {
        return paint.stop_colors[count - 1u];
    }

    for (var i: u32 = 1u; i < count; i = i + 1u) {
        if (t <= paint.stops[i]) {
            let s0 = paint.stops[i - 1u];
            let s1 = paint.stops[i];
            let range = s1 - s0;
            var frac = 0.0;
            if (range > 1e-6) {
                frac = (t - s0) / range;
            }
            return mix(paint.stop_colors[i - 1u], paint.stop_colors[i], frac);
        }
    }

    return paint.stop_colors[count - 1u];
}

// ─── sRGB ↔ Linear conversion ────────────────────────────────
// Lottie colors are specified in sRGB space. When rendering to an sRGB
// surface, the GPU auto-converts shader output (linear) → sRGB. So we
// must ensure our colors are in linear space before output.

fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear3(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b));
}

// ─── Fragment Shader ─────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let paint = paints[in.paint_index];
    var color: vec4<f32>;

    if (paint.paint_type == PAINT_LINEAR_GRAD) {
        // Linear gradient: project world_pos onto gradient line
        let d = paint.grad_end - paint.grad_start;
        let len_sq = dot(d, d);
        var t = 0.0;
        if (len_sq > 1e-8) {
            t = dot(in.world_pos - paint.grad_start, d) / len_sq;
        }
        color = sample_gradient(paint, t);
        color.a *= paint.opacity;
    } else if (paint.paint_type == PAINT_RADIAL_GRAD) {
        // Radial gradient: distance from start / distance(start, end)
        let radius = length(paint.grad_end - paint.grad_start);
        var t = 0.0;
        if (radius > 1e-6) {
            t = length(in.world_pos - paint.grad_start) / radius;
        }
        color = sample_gradient(paint, t);
        color.a *= paint.opacity;
    } else {
        // Solid color — read from vertex
        color = in.color;
    }

    // Convert sRGB input colors to linear space
    // (the sRGB surface will convert linear → sRGB on write)
    color = vec4<f32>(srgb_to_linear3(color.rgb), color.a);

    // Premultiply alpha for correct blending
    color = vec4<f32>(color.rgb * color.a, color.a);

    return color;
}
