// UltraRender GPU Tessellation Shader (Rive-style)
//
// Two-phase tessellation:
// 1. Vertex shader: outputs 1px-tall rectangles (spans) with cubic bezier data
// 2. Fragment shader: evaluates parametric + polar merge via binary search,
//    then De Casteljau for final curve point
//
// Input: TessVertexSpan buffer (storage)
// Output: Tessellation texture (RGBA32F) with [position.xy, angle, contourID]

struct FlushUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: FlushUniforms;

// TessVertexSpan data unpacked per-instance
struct TessSpanData {
    p0: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
    p3: vec2<f32>,
    join_tangent: vec2<f32>,
    y: f32,
    reflection_y: f32,
    x0: i32,
    x1: i32,
    parametric_segments: u32,
    polar_segments: u32,
    join_segments: u32,
    contour_id_with_flags: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) p0p1: vec4<f32>,
    @location(1) p2p3: vec4<f32>,
    @location(2) span_coord: vec2<f32>,  // local coordinate within span
    @location(3) @interpolate(flat) segment_counts: vec3<u32>,  // parametric, polar, join
    @location(4) @interpolate(flat) contour_id: u32,
};

// ─── Vertex Shader ───────────────────────────────────────────

@vertex
fn vs_main(
    @location(0) a_p0p1: vec4<f32>,
    @location(1) a_p2p3: vec4<f32>,
    @location(2) a_join_tan_ys: vec4<f32>,
    @location(3) a_args: vec4<u32>,
) -> VertexOutput {
    var out: VertexOutput;

    // Unpack control points
    out.p0p1 = a_p0p1;
    out.p2p3 = a_p2p3;

    // Unpack segment counts from packed u32
    let seg_packed = a_args.z;
    let parametric = seg_packed & 0x3FFu;
    let polar = (seg_packed >> 10u) & 0x3FFu;
    let join = (seg_packed >> 20u) & 0x3FFu;
    out.segment_counts = vec3<u32>(parametric, polar, join);
    out.contour_id = a_args.w;

    // Unpack x0, x1 from packed i32
    let x0x1 = bitcast<i32>(a_args.x);
    let x0 = f32(x0x1 & 0xFFFF);
    let x1 = f32((x0x1 >> 16) & 0xFFFF);
    let y = a_join_tan_ys.z;

    // Emit 1px-tall rectangle in tessellation texture space
    // Each vertex of the quad maps to a different corner
    out.span_coord = vec2<f32>(0.0, 0.0); // Will be set per-vertex
    out.position = vec4<f32>(x0 / uniforms.resolution.x * 2.0 - 1.0, y, 0.0, 1.0);

    return out;
}

// ─── Fragment Shader: Parametric + Polar Merge ───────────────

// De Casteljau cubic bezier evaluation
fn eval_cubic(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let ab = mix(p0, p1, t);
    let bc = mix(p1, p2, t);
    let cd = mix(p2, p3, t);
    let abc = mix(ab, bc, t);
    let bcd = mix(bc, cd, t);
    return mix(abc, bcd, t);
}

// Cubic tangent at parameter t
fn eval_cubic_tangent(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let ab = mix(p0, p1, t);
    let bc = mix(p1, p2, t);
    let cd = mix(p2, p3, t);
    let abc = mix(ab, bc, t);
    let bcd = mix(bc, cd, t);
    return bcd - abc;
}

// Wang's formula: segment count for cubic bezier (GPU verification)
fn wang_cubic_segments(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, precision: f32) -> u32 {
    let d0 = p0 - 2.0 * p1 + p2;
    let d1 = p1 - 2.0 * p2 + p3;
    let m = max(dot(d0, d0), dot(d1, d1));
    let n = ceil(sqrt(sqrt(0.75 * 4.0 * m)) * precision);
    return u32(clamp(n, 1.0, 1024.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p0 = in.p0p1.xy;
    let p1 = in.p0p1.zw;
    let p2 = in.p2p3.xy;
    let p3 = in.p2p3.zw;

    let parametric_count = in.segment_counts.x;
    let polar_count = in.segment_counts.y;
    let total_segments = parametric_count + polar_count;

    if total_segments == 0u {
        discard;
    }

    // Determine which merged vertex this fragment represents
    let frag_x = in.span_coord.x;
    let merged_vertex_id = u32(frag_x * f32(total_segments));

    // Binary search: find the highest parametric vertex where
    // parametricVertexID + floor(numPolarSegmentsAtT) <= mergedVertexID
    var lo: u32 = 0u;
    var hi: u32 = parametric_count;

    for (var iter = 0u; iter < 10u; iter = iter + 1u) {
        if lo >= hi {
            break;
        }
        let mid = (lo + hi) >> 1u;
        let t_at_mid = f32(mid) / f32(parametric_count);
        let polar_at_mid = u32(t_at_mid * f32(polar_count));

        if mid + polar_at_mid <= merged_vertex_id {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    // Compute final T parameter
    let parametric_vertex_id = min(lo, parametric_count);
    let t = f32(parametric_vertex_id) / f32(max(parametric_count, 1u));

    // Evaluate curve position via De Casteljau
    let pos = eval_cubic(p0, p1, p2, p3, t);

    // Compute tangent angle for stroke joins
    let tangent = eval_cubic_tangent(p0, p1, p2, p3, t);
    let angle = atan2(tangent.y, tangent.x);

    // Output: [position.x, position.y, angle, contourID]
    return vec4<f32>(pos.x, pos.y, angle, bitcast<f32>(in.contour_id));
}
