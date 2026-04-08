// UltraRender - Path Drawing Shader (Rive-style)
//
// Renders solid-color filled and stroked paths with coverage-based AA.
// Reads tessellated vertex data and applies paint colors.

// ─── Uniforms ────────────────────────────────────────────────

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;

// ─── Vertex Input/Output ─────────────────────────────────────

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) coverage: f32,
};

// ─── Vertex Shader ───────────────────────────────────────────

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    out.uv = in.uv;
    out.coverage = 1.0; // Full coverage for now (feather will modulate this)
    return out;
}

// ─── Fragment Shader ─────────────────────────────────────────

// Feather AA: smooth edge coverage using distance-based falloff
fn compute_feather_coverage(uv: vec2<f32>, feather_radius: f32) -> f32 {
    // For future: sample from 1D Gaussian feather texture
    // For now: use simple smooth-step based on UV distance from edge
    if feather_radius <= 0.0 {
        return 1.0;
    }

    // Distance from edge (uv.x = 0 at center, 1 at edge for strokes)
    let edge_dist = 1.0 - abs(uv.x * 2.0 - 1.0);
    let feather_t = edge_dist / feather_radius;
    return smoothstep(0.0, 1.0, feather_t);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = in.color;

    // Apply coverage (for future feather AA)
    color.a *= in.coverage;

    // Premultiply alpha for correct blending
    color = vec4<f32>(color.rgb * color.a, color.a);

    return color;
}
