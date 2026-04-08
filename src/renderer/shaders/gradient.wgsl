// UltraRender - Gradient shader
// Renders linear and radial gradients

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
};

struct GradientStop {
    offset: f32,
    r: f32,
    g: f32,
    b: f32,
};

struct GradientData {
    start: vec2<f32>,
    end: vec2<f32>,
    gradient_type: u32, // 1 = linear, 2 = radial
    stop_count: u32,
    _pad0: u32,
    _pad1: u32,
    stops: array<GradientStop, 16>,
};

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;
@group(1) @binding(0) var<storage, read> gradient: GradientData;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    out.world_pos = in.position;
    out.uv = in.uv;
    return out;
}

fn sample_gradient_color(t: f32) -> vec4<f32> {
    let clamped_t = clamp(t, 0.0, 1.0);
    let count = gradient.stop_count;

    if count == 0u {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    if count == 1u {
        let s = gradient.stops[0];
        return vec4<f32>(s.r, s.g, s.b, 1.0);
    }

    // Find surrounding stops
    var prev_idx: u32 = 0u;
    var next_idx: u32 = 0u;

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        if gradient.stops[i].offset <= clamped_t {
            prev_idx = i;
        }
        if gradient.stops[i].offset >= clamped_t {
            next_idx = i;
            break;
        }
        next_idx = i;
    }

    let prev_stop = gradient.stops[prev_idx];
    let next_stop = gradient.stops[next_idx];

    if prev_idx == next_idx {
        return vec4<f32>(prev_stop.r, prev_stop.g, prev_stop.b, 1.0);
    }

    let range = next_stop.offset - prev_stop.offset;
    var segment_t: f32 = 0.0;
    if range > 0.0001 {
        segment_t = (clamped_t - prev_stop.offset) / range;
    }

    let r = mix(prev_stop.r, next_stop.r, segment_t);
    let g = mix(prev_stop.g, next_stop.g, segment_t);
    let b = mix(prev_stop.b, next_stop.b, segment_t);

    return vec4<f32>(r, g, b, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var t: f32;

    if gradient.gradient_type == 1u {
        // Linear gradient
        let dir = gradient.end - gradient.start;
        let len_sq = dot(dir, dir);
        if len_sq < 0.0001 {
            t = 0.0;
        } else {
            t = dot(in.world_pos - gradient.start, dir) / len_sq;
        }
    } else {
        // Radial gradient
        let center = gradient.start;
        let radius = length(gradient.end - gradient.start);
        if radius < 0.0001 {
            t = 0.0;
        } else {
            t = length(in.world_pos - center) / radius;
        }
    }

    return sample_gradient_color(t);
}
