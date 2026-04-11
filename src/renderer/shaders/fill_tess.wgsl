// UltraRender GPU Fill Tessellation + Topology
//
// The GPU evaluates contour cubics via De Casteljau, writes sampled edge
// vertices into the shared vertex buffer, and materializes topology for the
// contours that fit the current GPU fast path.

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
const FILL_TOPOLOGY_COMPLEX_GPU: u32 = 3u;
const MAX_GPU_COMPLEX_FILL_VERTS: u32 = 512u;
const POSITION_EPSILON: f32 = 1e-3;
const AREA_EPSILON: f32 = 1e-4;
const EAR_EPSILON: f32 = 1e-5;

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

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

fn triangle_cross(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return cross2(b - a, c - a);
}

fn point_in_triangle(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {
    let ab = triangle_cross(a, b, p);
    let bc = triangle_cross(b, c, p);
    let ca = triangle_cross(c, a, p);
    let has_neg = ab < -EAR_EPSILON || bc < -EAR_EPSILON || ca < -EAR_EPSILON;
    let has_pos = ab > EAR_EPSILON || bc > EAR_EPSILON || ca > EAR_EPSILON;
    return !(has_neg && has_pos);
}

fn almost_equal(a: vec2<f32>, b: vec2<f32>) -> bool {
    let delta = a - b;
    return dot(delta, delta) <= POSITION_EPSILON * POSITION_EPSILON;
}

fn contour_vertex_pos(contour: GpuFillContour, local_idx: u32) -> vec2<f32> {
    return vertices[contour.first_vertex + local_idx].position;
}

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
        vertices[vi] = GpuVertex(
            pos,
            vec2<f32>(0.0, 0.0),
            contour.color,
            vec2<f32>(0.0, 0.0),
            contour.paint_index,
            contour.sprite_index,
        );
    }

    if (cubic_idx == contour.first_cubic) {
        let mid_vi = contour.first_vertex + contour.total_edge_verts;
        vertices[mid_vi] = GpuVertex(
            contour.midpoint,
            vec2<f32>(0.5, 0.5),
            contour.color,
            vec2<f32>(0.0, 0.0),
            contour.paint_index,
            contour.sprite_index,
        );

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

@compute @workgroup_size(1)
fn cs_fill_topology(@builtin(global_invocation_id) gid: vec3<u32>) {
    let contour_idx = gid.x;
    if (contour_idx >= uniforms.total_contours) {
        return;
    }

    let contour = contours[contour_idx];
    if (contour.simple_fill != FILL_TOPOLOGY_COMPLEX_GPU) {
        return;
    }

    let reserved_triangles = contour.total_edge_verts;
    for (var tri_idx = 0u; tri_idx < reserved_triangles; tri_idx = tri_idx + 1u) {
        let tri = contour.first_index + tri_idx * 3u;
        indices[tri] = contour.first_vertex;
        indices[tri + 1u] = contour.first_vertex;
        indices[tri + 2u] = contour.first_vertex;
    }

    if (contour.total_edge_verts < 3u || contour.total_edge_verts > MAX_GPU_COMPLEX_FILL_VERTS) {
        return;
    }

    var point_count = contour.total_edge_verts;
    if (point_count > 1u) {
        let first_pos = contour_vertex_pos(contour, 0u);
        let last_pos = contour_vertex_pos(contour, point_count - 1u);
        if (almost_equal(first_pos, last_pos)) {
            point_count = point_count - 1u;
        }
    }
    if (point_count < 3u) {
        return;
    }

    var filtered: array<u32, 512>;
    var filtered_count = 0u;
    for (var i = 0u; i < point_count; i = i + 1u) {
        var prev_i = point_count - 1u;
        if (i > 0u) {
            prev_i = i - 1u;
        }
        let curr_pos = contour_vertex_pos(contour, i);
        let prev_pos = contour_vertex_pos(contour, prev_i);
        if (!almost_equal(curr_pos, prev_pos)) {
            filtered[filtered_count] = i;
            filtered_count = filtered_count + 1u;
        }
    }
    if (filtered_count < 3u) {
        return;
    }

    var changed = true;
    loop {
        if (!changed || filtered_count <= 3u) {
            break;
        }
        changed = false;
        var i = 0u;
        loop {
            if (i >= filtered_count) {
                break;
            }
            var prev_i = filtered_count - 1u;
            if (i > 0u) {
                prev_i = i - 1u;
            }
            let next_i = select(i + 1u, 0u, i + 1u == filtered_count);
            let a = contour_vertex_pos(contour, filtered[prev_i]);
            let b = contour_vertex_pos(contour, filtered[i]);
            let c = contour_vertex_pos(contour, filtered[next_i]);
            if (abs(triangle_cross(a, b, c)) <= AREA_EPSILON) {
                for (var shift = i; shift + 1u < filtered_count; shift = shift + 1u) {
                    filtered[shift] = filtered[shift + 1u];
                }
                filtered_count = filtered_count - 1u;
                changed = true;
                break;
            }
            i = i + 1u;
        }
    }
    if (filtered_count < 3u) {
        return;
    }

    var area = 0.0;
    for (var i = 0u; i < filtered_count; i = i + 1u) {
        let next_i = select(i + 1u, 0u, i + 1u == filtered_count);
        let p0 = contour_vertex_pos(contour, filtered[i]);
        let p1 = contour_vertex_pos(contour, filtered[next_i]);
        area = area + cross2(p0, p1);
    }
    area = area * 0.5;
    if (abs(area) < AREA_EPSILON) {
        return;
    }
    let is_ccw = area > 0.0;

    var remaining: array<u32, 512>;
    for (var i = 0u; i < filtered_count; i = i + 1u) {
        remaining[i] = filtered[i];
    }
    var remaining_count = filtered_count;
    var triangles_written = 0u;
    var guard = remaining_count * remaining_count;

    loop {
        if (remaining_count <= 3u || guard == 0u) {
            break;
        }
        guard = guard - 1u;
        var ear_found = false;
        var i = 0u;
        loop {
            if (i >= remaining_count) {
                break;
            }

            var prev_i = remaining_count - 1u;
            if (i > 0u) {
                prev_i = i - 1u;
            }
            let next_i = select(i + 1u, 0u, i + 1u == remaining_count);
            let prev = remaining[prev_i];
            let curr = remaining[i];
            let next = remaining[next_i];
            let a = contour_vertex_pos(contour, prev);
            let b = contour_vertex_pos(contour, curr);
            let c = contour_vertex_pos(contour, next);
            let cross = triangle_cross(a, b, c);

            var is_convex = false;
            if (is_ccw) {
                is_convex = cross > EAR_EPSILON;
            } else {
                is_convex = cross < -EAR_EPSILON;
            }

            if (is_convex) {
                var contains_point = false;
                for (var j = 0u; j < remaining_count; j = j + 1u) {
                    let test = remaining[j];
                    if (test == prev || test == curr || test == next) {
                        continue;
                    }
                    if (point_in_triangle(contour_vertex_pos(contour, test), a, b, c)) {
                        contains_point = true;
                        break;
                    }
                }

                if (!contains_point) {
                    let tri = contour.first_index + triangles_written * 3u;
                    indices[tri] = contour.first_vertex + prev;
                    indices[tri + 1u] = contour.first_vertex + curr;
                    indices[tri + 2u] = contour.first_vertex + next;
                    triangles_written = triangles_written + 1u;

                    for (var shift = i; shift + 1u < remaining_count; shift = shift + 1u) {
                        remaining[shift] = remaining[shift + 1u];
                    }
                    remaining_count = remaining_count - 1u;
                    ear_found = true;
                    break;
                }
            }

            i = i + 1u;
        }

        if (!ear_found) {
            return;
        }
    }

    if (remaining_count == 3u) {
        let tri = contour.first_index + triangles_written * 3u;
        indices[tri] = contour.first_vertex + remaining[0];
        indices[tri + 1u] = contour.first_vertex + remaining[1];
        indices[tri + 2u] = contour.first_vertex + remaining[2];
    }
}