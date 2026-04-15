// UltraRender GPU Fill Tessellation + Topology
//
// The GPU evaluates contour cubics via De Casteljau, writes sampled edge
// vertices into the shared vertex buffer, and materializes midpoint-fan
// topology for all contours (simple and complex/stencil-based).
// No CPU fallback — all geometry is GPU-generated.

struct GpuFillContour {
    midpoint: vec2<f32>,
    first_cubic: u32,
    cubic_count: u32,
    first_vertex: u32,
    total_edge_verts: u32,
    first_index: u32,
    simple_fill: u32,
    color: vec4<f32>,
    paint_index: u32,
    sprite_index: u32,
    _pad: vec2<u32>,
};

struct GpuFillCubic {
    p0: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
    p3: vec2<f32>,
    segment_count: u32,
    first_edge_vertex: u32,
    _pad: vec2<u32>,
};

struct GpuVertex {
    position: vec2<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
    world_pos: vec2<f32>,
    paint_index: u32,
    sprite_index: u32,
};

struct FillTessUniforms {
    total_cubics: u32,
    total_contours: u32,
    _pad: vec2<u32>,
};

const FILL_TOPOLOGY_SIMPLE: u32 = 1u;
const FILL_TOPOLOGY_COMPLEX_FAN: u32 = 2u;

@group(0) @binding(0) var<uniform> uniforms: FillTessUniforms;
@group(0) @binding(1) var<storage, read> contours: array<GpuFillContour>;
@group(0) @binding(2) var<storage, read> cubics: array<GpuFillCubic>;
@group(0) @binding(3) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;

fn eval_cubic(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let ab = mix(p0, p1, t);
    let bc = mix(p1, p2, t);
    let cd = mix(p2, p3, t);
    let abc = mix(ab, bc, t);
    let bcd = mix(bc, cd, t);
    return mix(abc, bcd, t);
}

// Tessellate cubic edges and emit midpoint-fan indices.
// Each invocation processes one cubic segment.
@compute @workgroup_size(64)
fn cs_fill_tessellate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cubic_idx = gid.x;
    if (cubic_idx >= uniforms.total_cubics) {
        return;
    }

    let cubic = cubics[cubic_idx];
    let n = cubic.segment_count;
    if (n == 0u) {
        return;
    }

    var contour_idx = 0u;
    for (var c = 0u; c < uniforms.total_contours; c = c + 1u) {
        let ct = contours[c];
        if (cubic_idx >= ct.first_cubic && cubic_idx < ct.first_cubic + ct.cubic_count) {
            contour_idx = c;
            break;
        }
    }

    let contour = contours[contour_idx];
    let base_vertex = contour.first_vertex + cubic.first_edge_vertex;
    let is_last_cubic = cubic_idx == contour.first_cubic + contour.cubic_count - 1u;
    let write_count = select(n, n + 1u, is_last_cubic);

    for (var i = 0u; i < write_count; i = i + 1u) {
        let t = f32(i) / f32(n);
        let pos = eval_cubic(cubic.p0, cubic.p1, cubic.p2, cubic.p3, t);
        let vi = base_vertex + i;
        // Edge vertex: uv.x = 0.0 (on boundary edge, coverage = 0 for AA),
        //              uv.y = 0.0 (not a stroke).
        vertices[vi] = GpuVertex(
            pos,
            vec2<f32>(0.0, 0.0),
            contour.color,
            vec2<f32>(0.0, 0.0),
            contour.paint_index,
            contour.sprite_index,
        );
    }

    // First cubic of the contour writes the midpoint vertex and fan indices.
    if (cubic_idx == contour.first_cubic) {
        let mid_vi = contour.first_vertex + contour.total_edge_verts;
        // Midpoint: uv.x = 1.0 (fill coverage = fully interior),
        //           uv.y = 0.0 (not a stroke).
        vertices[mid_vi] = GpuVertex(
            contour.midpoint,
            vec2<f32>(1.0, 0.0),
            contour.color,
            vec2<f32>(0.0, 0.0),
            contour.paint_index,
            contour.sprite_index,
        );

        // Both simple and complex fills use midpoint fan topology.
        // Complex (stencil-based) fills rely on winding accumulation,
        // so a fan from any interior point is correct even for concave shapes.
        if (contour.simple_fill == FILL_TOPOLOGY_SIMPLE || contour.simple_fill == FILL_TOPOLOGY_COMPLEX_FAN) {
            for (var i = 0u; i < contour.total_edge_verts; i = i + 1u) {
                let next = select(i + 1u, 0u, i + 1u == contour.total_edge_verts);
                let tri = contour.first_index + i * 3u;
                indices[tri] = mid_vi;
                indices[tri + 1u] = contour.first_vertex + i;
                indices[tri + 2u] = contour.first_vertex + next;
            }
        }
    }
}

