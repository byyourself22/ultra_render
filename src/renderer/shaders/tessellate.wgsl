// UltraRender GPU Tessellation Compute Shader (Rive-style)
//
// Evaluates cubic bezier segments on the GPU using De Casteljau's algorithm.
// Each workgroup thread processes one cubic segment and writes evaluated
// vertex positions into the output buffer.
//
// For fills: outputs plain positions (midpoint fan indexing done on CPU).
// For strokes: evaluates tangent → normal, offsets by ±half_width.
//
// Input:  GpuCubicSegment[] (storage, read)
// Output: GpuVertex[]       (storage, read_write)

// ─── Data structures (must match Rust repr(C)) ──────────────

struct GpuCubicSegment {
    p0: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
    p3: vec2<f32>,
    color: vec4<f32>,
    normal_offset: f32,
    segment_count: u32,
    first_vertex_idx: u32,
    flags: u32,
    paint_index: u32,
    sprite_index: u32,
    _pad0: u32,
    _pad1: u32,
};

struct GpuVertex {
    position: vec2<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
    world_pos: vec2<f32>,
    paint_index: u32,
    sprite_index: u32,
};

struct TessUniforms {
    total_vertices: u32,
    total_segments: u32,
    _pad0: u32,
    _pad1: u32,
};

// ─── Bindings ───────────────────────────────────────────────

@group(0) @binding(0) var<uniform> uniforms: TessUniforms;
@group(0) @binding(1) var<storage, read> segments: array<GpuCubicSegment>;
@group(0) @binding(2) var<storage, read_write> output_vertices: array<GpuVertex>;

// ─── De Casteljau cubic evaluation ──────────────────────────

fn eval_cubic(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let ab = mix(p0, p1, t);
    let bc = mix(p1, p2, t);
    let cd = mix(p2, p3, t);
    let abc = mix(ab, bc, t);
    let bcd = mix(bc, cd, t);
    return mix(abc, bcd, t);
}

fn eval_cubic_tangent(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let ab = mix(p0, p1, t);
    let bc = mix(p1, p2, t);
    let cd = mix(p2, p3, t);
    let abc = mix(ab, bc, t);
    let bcd = mix(bc, cd, t);
    return bcd - abc;
}

// ─── Compute entry point ────────────────────────────────────

@compute @workgroup_size(64)
fn cs_tessellate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seg_idx = gid.x;
    if (seg_idx >= uniforms.total_segments) {
        return;
    }

    let seg = segments[seg_idx];
    let n = seg.segment_count;
    let is_stroke = (seg.flags & 1u) != 0u;
    let write_last = (seg.flags & 2u) != 0u;

    // Determine how many vertices to write
    let count = select(n, n + 1u, write_last);

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let t = f32(i) / f32(n);
        var pos = eval_cubic(seg.p0, seg.p1, seg.p2, seg.p3, t);
        let world_pos = pos; // pre-offset position for gradient eval

        var uv = vec2<f32>(0.0, 0.0);

        // Stroke: offset by normal * normal_offset
        if (is_stroke && seg.normal_offset != 0.0) {
            let tangent = eval_cubic_tangent(seg.p0, seg.p1, seg.p2, seg.p3, t);
            let len = length(tangent);
            if (len > 1e-6) {
                let normal = vec2<f32>(-tangent.y, tangent.x) / len;
                pos = pos + normal * seg.normal_offset;
            }
            // UV encodes stroke side for analytic edge AA in the draw shader.
            uv = vec2<f32>(select(0.0, 1.0, seg.normal_offset < 0.0), 1.0);
        }

        let vi = seg.first_vertex_idx + i;
        if (vi < uniforms.total_vertices) {
            output_vertices[vi] = GpuVertex(
                pos,
                uv,
                seg.color,
                world_pos,
                seg.paint_index,
                seg.sprite_index,
            );
        }
    }
}
