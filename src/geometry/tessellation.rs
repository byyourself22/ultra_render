//! GPU tessellation data structures (Rive-style).
//!
//! Path curves are sent to the GPU as cubic segments. A compute shader evaluates
//! De Casteljau at each subdivision point (Wang's formula), producing a flat
//! vertex buffer that the draw pass consumes.
//!
//! CPU work: walk contours, pack cubics, compute segment counts, build index buffer.
//! GPU work: evaluate curves (De Casteljau), offset strokes, write vertices.

use super::math::{Vec2D, Mat2D, Color, wang_cubic_segment_count};
use super::path::{RawPath, PathVerb};
use bytemuck::{Pod, Zeroable};

// ─── GPU vertex (output of compute, input to draw) ──────────

/// GPU vertex consumed by the draw shader.
///
/// Each vertex carries a paint_index into the paint storage buffer, so the
/// fragment shader can look up gradient parameters (ThorVG-style fragment
/// gradient evaluation). world_pos is the pre-transform position needed
/// for gradient coordinate computation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 2],   // screen-space position (after world transform)
    pub uv: [f32; 2],         // texture coords / stroke AA side
    pub color: [f32; 4],      // solid color (used when paint_type=0)
    pub world_pos: [f32; 2],  // pre-transform position for gradient eval
    pub paint_index: u32,     // index into paint storage buffer
    pub _pad: u32,            // alignment
}

// ─── Compute shader inputs ──────────────────────────────────

/// Maximum gradient color stops per paint.
pub const MAX_GRADIENT_STOPS: usize = 16;

/// GPU paint descriptor read by the fragment shader.
///
/// Encodes solid color or gradient (linear/radial) with up to MAX_GRADIENT_STOPS
/// color stops. The fragment shader evaluates gradients per-fragment using the
/// vertex's world_pos (ThorVG-style).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPaint {
    pub paint_type: u32,        // 0=solid, 1=linear_gradient, 2=radial_gradient
    pub stop_count: u32,        // number of valid stops
    pub opacity: f32,           // overall paint opacity
    pub _pad: u32,
    pub grad_start: [f32; 2],   // gradient start point (local space)
    pub grad_end: [f32; 2],     // gradient end point (local space)
    pub stops: [f32; MAX_GRADIENT_STOPS],          // stop positions [0..1]
    pub stop_colors: [[f32; 4]; MAX_GRADIENT_STOPS], // RGBA per stop
}

impl Default for GpuPaint {
    fn default() -> Self {
        Self {
            paint_type: 0,
            stop_count: 0,
            opacity: 1.0,
            _pad: 0,
            grad_start: [0.0; 2],
            grad_end: [0.0; 2],
            stops: [0.0; MAX_GRADIENT_STOPS],
            stop_colors: [[0.0; 4]; MAX_GRADIENT_STOPS],
        }
    }
}

/// A cubic bezier segment packed for GPU evaluation.
/// The compute shader reads this and writes GpuVertex entries.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuCubicSegment {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub p2: [f32; 2],
    pub p3: [f32; 2],
    pub color: [f32; 4],           // Paint color (RGBA, for solid paints)
    pub normal_offset: f32,        // 0 = fill vertex, ±half_width = stroke left/right
    pub segment_count: u32,        // Wang's formula subdivisions
    pub first_vertex_idx: u32,     // Where this segment's output starts in vertex buffer
    pub flags: u32,                // bit0: is_stroke, bit1: write_last
    pub paint_index: u32,          // Index into paint storage buffer
    pub _pad: [u32; 3],            // Pad to 16-byte alignment
}

/// Per-dispatch metadata uploaded as a uniform for the compute shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TessellateUniforms {
    pub total_vertices: u32,
    pub total_segments: u32,
    pub _pad: [u32; 2],
}

// ─── CPU path encoder ───────────────────────────────────────

/// Result of encoding paths for the GPU tessellation pipeline.
pub struct EncodedPathData {
    /// Cubic segments (storage buffer input to compute shader)
    pub segments: Vec<GpuCubicSegment>,
    /// Index buffer for the draw pass (triangle list)
    pub indices: Vec<u32>,
    /// Total output vertex count (compute shader writes this many GpuVertex)
    pub total_vertices: u32,
    /// Paint descriptors (storage buffer read by fragment shader)
    pub paints: Vec<GpuPaint>,
}

/// Encode a set of ShapeDrawCommands into GPU-ready tessellation data.
///
/// For fills: midpoint-fan triangulation (each contour fans from its centroid).
/// For strokes: offset quad-strip (left/right vertices along the curve).
///
/// The compute shader evaluates De Casteljau to produce vertex positions.
pub fn encode_paths(
    commands: &[super::super::scene::layer::ShapeDrawCommand],
    world_transform: &Mat2D,
) -> EncodedPathData {
    let mut segments: Vec<GpuCubicSegment> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut paints: Vec<GpuPaint> = Vec::new();
    let mut vertex_cursor: u32 = 0;

    for cmd in commands {
        let (color, is_stroke, half_width) = extract_paint_info(&cmd.paint);

        // Build GPU paint entry for this draw command
        let paint_index = paints.len() as u32;
        paints.push(build_gpu_paint(&cmd.paint, world_transform));

        // Transform path by world transform
        let path = cmd.path.transform(world_transform);

        // Walk contours
        for contour in iter_contours(&path) {
            if contour.cubics.is_empty() {
                continue;
            }

            if is_stroke {
                vertex_cursor = encode_stroke_contour(
                    &contour,
                    color,
                    half_width,
                    paint_index,
                    &mut segments,
                    &mut indices,
                    vertex_cursor,
                );
            } else {
                vertex_cursor = encode_fill_contour(
                    &contour,
                    color,
                    paint_index,
                    &mut segments,
                    &mut indices,
                    vertex_cursor,
                );
            }
        }
    }

    // Ensure at least one paint entry (shader needs valid buffer)
    if paints.is_empty() {
        paints.push(GpuPaint::default());
    }

    EncodedPathData {
        segments,
        indices,
        total_vertices: vertex_cursor,
        paints,
    }
}

// ─── Contour extraction ─────────────────────────────────────

/// A contour (subpath) decomposed into cubic bezier segments.
struct Contour {
    cubics: Vec<[Vec2D; 4]>,   // [p0, p1, p2, p3] per cubic
    closed: bool,
}

/// Walk a RawPath and yield contours as lists of cubic segments.
/// Lines are promoted to degenerate cubics (Rive-style).
fn iter_contours(path: &RawPath) -> Vec<Contour> {
    let mut contours = Vec::new();
    let mut cubics: Vec<[Vec2D; 4]> = Vec::new();
    let mut current_pos = Vec2D::ZERO;
    let mut move_pos = Vec2D::ZERO;
    let mut point_idx = 0;

    for verb in &path.verbs {
        match verb {
            PathVerb::Move => {
                if !cubics.is_empty() {
                    contours.push(Contour { cubics: std::mem::take(&mut cubics), closed: false });
                }
                current_pos = path.points[point_idx];
                move_pos = current_pos;
                point_idx += 1;
            }
            PathVerb::Line => {
                let p3 = path.points[point_idx];
                point_idx += 1;
                // Promote line to degenerate cubic: p0, lerp(p0,p3,1/3), lerp(p0,p3,2/3), p3
                let p1 = current_pos.lerp(p3, 1.0 / 3.0);
                let p2 = current_pos.lerp(p3, 2.0 / 3.0);
                cubics.push([current_pos, p1, p2, p3]);
                current_pos = p3;
            }
            PathVerb::Quad => {
                let q1 = path.points[point_idx];
                let q2 = path.points[point_idx + 1];
                point_idx += 2;
                // Quad → cubic elevation
                let p1 = Vec2D::new(
                    current_pos.x + 2.0 / 3.0 * (q1.x - current_pos.x),
                    current_pos.y + 2.0 / 3.0 * (q1.y - current_pos.y),
                );
                let p2 = Vec2D::new(
                    q2.x + 2.0 / 3.0 * (q1.x - q2.x),
                    q2.y + 2.0 / 3.0 * (q1.y - q2.y),
                );
                cubics.push([current_pos, p1, p2, q2]);
                current_pos = q2;
            }
            PathVerb::Cubic => {
                let c1 = path.points[point_idx];
                let c2 = path.points[point_idx + 1];
                let p3 = path.points[point_idx + 2];
                point_idx += 3;
                cubics.push([current_pos, c1, c2, p3]);
                current_pos = p3;
            }
            PathVerb::Close => {
                // Close back to move point if not already there
                if current_pos.distance(move_pos) > 0.01 {
                    let p1 = current_pos.lerp(move_pos, 1.0 / 3.0);
                    let p2 = current_pos.lerp(move_pos, 2.0 / 3.0);
                    cubics.push([current_pos, p1, p2, move_pos]);
                    current_pos = move_pos;
                }
                if !cubics.is_empty() {
                    contours.push(Contour { cubics: std::mem::take(&mut cubics), closed: true });
                }
            }
        }
    }

    if !cubics.is_empty() {
        contours.push(Contour { cubics, closed: false });
    }

    contours
}

// ─── Fill encoding (midpoint fan) ───────────────────────────

/// Encode a filled contour as midpoint-fan triangles.
///
/// Midpoint fan (Rive-style): compute contour centroid, then for each edge
/// vertex pair (v[i], v[i+1]) emit triangle (centroid, v[i], v[i+1]).
///
/// Returns the new vertex cursor.
fn encode_fill_contour(
    contour: &Contour,
    color: Color,
    paint_index: u32,
    segments: &mut Vec<GpuCubicSegment>,
    indices: &mut Vec<u32>,
    mut vertex_cursor: u32,
) -> u32 {
    // --- Compute contour midpoint (centroid of cubic endpoints) ---
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut n_pts = 0;
    for cubic in &contour.cubics {
        cx += cubic[0].x;
        cy += cubic[0].y;
        n_pts += 1;
    }
    if n_pts == 0 {
        return vertex_cursor;
    }
    cx /= n_pts as f32;
    cy /= n_pts as f32;

    // The midpoint is a special vertex: we'll encode it as a degenerate cubic
    // (all 4 points = midpoint) with segment_count = 1.
    let midpoint_vertex = vertex_cursor;
    let midpoint = Vec2D::new(cx, cy);
    segments.push(GpuCubicSegment {
        p0: [midpoint.x, midpoint.y],
        p1: [midpoint.x, midpoint.y],
        p2: [midpoint.x, midpoint.y],
        p3: [midpoint.x, midpoint.y],
        color: [color.r, color.g, color.b, color.a],
        normal_offset: 0.0,
        segment_count: 1,
        first_vertex_idx: vertex_cursor,
        flags: 0,
        paint_index,
        _pad: [0; 3],
    });
    vertex_cursor += 1; // 1 vertex for the midpoint

    // --- Encode edge vertices from cubics ---
    let ring_start = vertex_cursor;
    let mut total_ring_vertices: u32 = 0;

    for (i, cubic) in contour.cubics.iter().enumerate() {
        let seg_count = wang_cubic_segment_count(
            cubic[0], cubic[1], cubic[2], cubic[3], 1.0,
        ).max(1);

        // How many NEW vertices this segment contributes:
        // First segment: seg_count vertices (t=0..seg_count-1)/seg_count
        // (the t=1 vertex is shared with next segment's t=0)
        // Last segment: seg_count vertices (includes final point if closed)
        let num_new = if i == contour.cubics.len() - 1 {
            seg_count + 1 // include the final endpoint
        } else {
            seg_count // exclude last point (shared with next)
        };

        segments.push(GpuCubicSegment {
            p0: [cubic[0].x, cubic[0].y],
            p1: [cubic[1].x, cubic[1].y],
            p2: [cubic[2].x, cubic[2].y],
            p3: [cubic[3].x, cubic[3].y],
            color: [color.r, color.g, color.b, color.a],
            normal_offset: 0.0,
            segment_count: seg_count,
            first_vertex_idx: vertex_cursor,
            flags: if i == contour.cubics.len() - 1 { 2 } else { 0 }, // bit1 = write_last
            paint_index,
            _pad: [0; 3],
        });

        vertex_cursor += num_new;
        total_ring_vertices += num_new;
    }

    // --- Generate midpoint-fan indices ---
    if total_ring_vertices >= 2 {
        for i in 0..total_ring_vertices - 1 {
            indices.push(midpoint_vertex);
            indices.push(ring_start + i);
            indices.push(ring_start + i + 1);
        }
        // Close the fan (last ring vertex → first ring vertex)
        if contour.closed {
            indices.push(midpoint_vertex);
            indices.push(ring_start + total_ring_vertices - 1);
            indices.push(ring_start);
        }
    }

    vertex_cursor
}

// ─── Stroke encoding (offset quad strip) ────────────────────

/// Encode a stroked contour as offset quad strips.
///
/// For each curve point, the compute shader evaluates position and tangent,
/// then offsets left/right by half_width along the normal. The CPU generates
/// quad-strip indices connecting consecutive left/right pairs.
///
/// Returns the new vertex cursor.
fn encode_stroke_contour(
    contour: &Contour,
    color: Color,
    half_width: f32,
    paint_index: u32,
    segments: &mut Vec<GpuCubicSegment>,
    indices: &mut Vec<u32>,
    mut vertex_cursor: u32,
) -> u32 {
    let strip_start = vertex_cursor;
    let mut total_pairs: u32 = 0;

    for (i, cubic) in contour.cubics.iter().enumerate() {
        let seg_count = wang_cubic_segment_count(
            cubic[0], cubic[1], cubic[2], cubic[3], 1.0,
        ).max(1);

        let is_last = i == contour.cubics.len() - 1;
        let num_new = if is_last { seg_count + 1 } else { seg_count };

        // Left-side vertices (normal_offset = +half_width)
        segments.push(GpuCubicSegment {
            p0: [cubic[0].x, cubic[0].y],
            p1: [cubic[1].x, cubic[1].y],
            p2: [cubic[2].x, cubic[2].y],
            p3: [cubic[3].x, cubic[3].y],
            color: [color.r, color.g, color.b, color.a],
            normal_offset: half_width,
            segment_count: seg_count,
            first_vertex_idx: vertex_cursor,
            flags: 1 | if is_last { 2 } else { 0 }, // bit0=stroke, bit1=write_last
            paint_index,
            _pad: [0; 3],
        });
        vertex_cursor += num_new;

        // Right-side vertices (normal_offset = -half_width)
        segments.push(GpuCubicSegment {
            p0: [cubic[0].x, cubic[0].y],
            p1: [cubic[1].x, cubic[1].y],
            p2: [cubic[2].x, cubic[2].y],
            p3: [cubic[3].x, cubic[3].y],
            color: [color.r, color.g, color.b, color.a],
            normal_offset: -half_width,
            segment_count: seg_count,
            first_vertex_idx: vertex_cursor,
            flags: 1 | if is_last { 2 } else { 0 },
            paint_index,
            _pad: [0; 3],
        });
        vertex_cursor += num_new;

        total_pairs += num_new;
    }

    // --- Generate quad-strip indices ---
    // Vertices are interleaved: left[0], right[0], left[1], right[1], ...
    // But we stored left block then right block per segment.
    // Re-index: for pair i, left = strip_start + i, right = strip_start + total_pairs + i
    if total_pairs >= 2 {
        for i in 0..total_pairs - 1 {
            let l0 = strip_start + i;
            let r0 = strip_start + total_pairs + i;
            let l1 = strip_start + i + 1;
            let r1 = strip_start + total_pairs + i + 1;

            indices.push(l0);
            indices.push(r0);
            indices.push(l1);

            indices.push(r0);
            indices.push(r1);
            indices.push(l1);
        }
    }

    vertex_cursor
}

// ─── Paint helpers ──────────────────────────────────────────

use crate::scene::layer::ShapePaint;
use crate::lottie::model::{GradientType, GradientColors};

/// Build a GpuPaint from a ShapePaint for fragment shader gradient evaluation.
fn build_gpu_paint(paint: &ShapePaint, world_transform: &Mat2D) -> GpuPaint {
    match paint {
        ShapePaint::SolidFill { color, opacity, .. } => {
            let mut p = GpuPaint::default();
            p.paint_type = 0; // solid
            p.opacity = *opacity;
            p.stop_colors[0] = [color.r, color.g, color.b, color.a];
            p
        }
        ShapePaint::SolidStroke { color, opacity, .. } => {
            let mut p = GpuPaint::default();
            p.paint_type = 0;
            p.opacity = *opacity;
            p.stop_colors[0] = [color.r, color.g, color.b, color.a];
            p
        }
        ShapePaint::GradientFill { gradient_type, start, end, colors, opacity, .. } => {
            build_gradient_paint(*gradient_type, *start, *end, colors, *opacity, world_transform)
        }
        ShapePaint::GradientStroke { gradient_type, start, end, colors, opacity, .. } => {
            build_gradient_paint(*gradient_type, *start, *end, colors, *opacity, world_transform)
        }
    }
}

fn build_gradient_paint(
    gradient_type: GradientType,
    start: Vec2D,
    end: Vec2D,
    colors: &GradientColors,
    opacity: f32,
    world_transform: &Mat2D,
) -> GpuPaint {
    let mut p = GpuPaint::default();
    p.paint_type = match gradient_type {
        GradientType::Linear => 1,
        GradientType::Radial => 2,
    };
    p.opacity = opacity;

    // Transform gradient points to world space
    let ws = world_transform.transform_point(start);
    let we = world_transform.transform_point(end);
    p.grad_start = [ws.x, ws.y];
    p.grad_end = [we.x, we.y];

    // Parse Lottie gradient color stops: [offset, r, g, b, offset, r, g, b, ...]
    let count = colors.color_count.min(MAX_GRADIENT_STOPS);
    p.stop_count = count as u32;

    for i in 0..count {
        let base = i * 4;
        if base + 3 < colors.colors.len() {
            p.stops[i] = colors.colors[base];         // offset [0..1]
            p.stop_colors[i] = [
                colors.colors[base + 1],               // r
                colors.colors[base + 2],               // g
                colors.colors[base + 3],               // b
                1.0,                                    // a (alpha stops are separate in Lottie)
            ];
        }
    }

    p
}

/// Extract color, stroke flag, and half-width from a ShapePaint.
fn extract_paint_info(paint: &ShapePaint) -> (Color, bool, f32) {
    match paint {
        ShapePaint::SolidFill { color, opacity, .. } => {
            (color.with_opacity(*opacity), false, 0.0)
        }
        ShapePaint::SolidStroke { color, opacity, width, .. } => {
            (color.with_opacity(*opacity), true, *width * 0.5)
        }
        ShapePaint::GradientFill { opacity, .. } => {
            // Gradient fills use white as base; per-vertex gradient color
            // is applied via the gradient pipeline (future)
            (Color::WHITE.with_opacity(*opacity), false, 0.0)
        }
        ShapePaint::GradientStroke { opacity, width, .. } => {
            (Color::WHITE.with_opacity(*opacity), true, *width * 0.5)
        }
    }
}
