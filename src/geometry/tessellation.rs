//! GPU tessellation data structures (Rive-style).
//!
//! Fills:   Lyon CPU tessellation → correct concave polygon triangulation.
//! Strokes: GPU compute shader (De Casteljau) → Rive-style offset quad-strips.
//!
//! CPU work: walk contours, lyon fills, pack stroke cubics, build index buffer.
//! GPU work: evaluate stroke curves (De Casteljau), offset by normal.

use super::math::{Vec2D, Mat2D, Color, wang_cubic_segment_count};
use super::path::{RawPath, PathVerb};
use bytemuck::{Pod, Zeroable};

// ─── GPU vertex (output of compute, input to draw) ──────────

/// GPU vertex consumed by the draw shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 2],   // screen-space position
    pub uv: [f32; 2],         // texture coords / stroke AA side
    pub color: [f32; 4],      // solid color (used when paint_type=0)
    pub world_pos: [f32; 2],  // pre-transform position for gradient eval
    pub paint_index: u32,     // index into paint storage buffer
    pub _pad: u32,
}

// ─── Compute shader inputs ──────────────────────────────────

pub const MAX_GRADIENT_STOPS: usize = 16;

/// GPU paint descriptor read by the fragment shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPaint {
    pub paint_type: u32,
    pub stop_count: u32,
    pub opacity: f32,
    pub _pad: u32,
    pub grad_start: [f32; 2],
    pub grad_end: [f32; 2],
    pub stops: [f32; MAX_GRADIENT_STOPS],
    pub stop_colors: [[f32; 4]; MAX_GRADIENT_STOPS],
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

/// A cubic bezier segment packed for GPU stroke evaluation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuCubicSegment {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub p2: [f32; 2],
    pub p3: [f32; 2],
    pub color: [f32; 4],
    pub normal_offset: f32,
    pub segment_count: u32,
    pub first_vertex_idx: u32,  // absolute position in vertex buffer (after fill vertices)
    pub flags: u32,             // bit0: is_stroke, bit1: write_last
    pub paint_index: u32,
    pub _pad: [u32; 3],
}

/// Per-dispatch metadata for the compute shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TessellateUniforms {
    pub total_vertices: u32,   // total vertex buffer size (fill + stroke)
    pub total_segments: u32,
    pub _pad: [u32; 2],
}

// ─── Encoded path data ──────────────────────────────────────

/// Result of encoding paths for the GPU pipeline.
///
/// Fills: CPU-tessellated via Lyon → uploaded directly to start of vertex buffer.
/// Strokes: GPU compute (De Casteljau) → segments describe what compute shader writes.
pub struct EncodedPathData {
    /// Lyon fill vertices — uploaded directly to vertex_buffer[0..fill_vertex_count].
    pub fill_vertices: Vec<GpuVertex>,
    /// Indices for fill triangles (referencing fill_vertices region).
    pub fill_indices: Vec<u32>,
    /// Stroke cubic segments for compute shader.
    pub segments: Vec<GpuCubicSegment>,
    /// Indices for stroke triangles (referencing stroke region of vertex buffer).
    pub stroke_indices: Vec<u32>,
    /// Number of vertex slots consumed by compute shader (for strokes).
    pub stroke_vertex_count: u32,
    /// Paint descriptors (both fill and stroke).
    pub paints: Vec<GpuPaint>,
}

impl EncodedPathData {
    pub fn total_vertex_count(&self) -> u32 {
        self.fill_vertices.len() as u32 + self.stroke_vertex_count
    }
    pub fn total_index_count(&self) -> u32 {
        (self.fill_indices.len() + self.stroke_indices.len()) as u32
    }
    pub fn combined_indices(&self) -> Vec<u32> {
        let mut v = self.fill_indices.clone();
        v.extend_from_slice(&self.stroke_indices);
        v
    }
    pub fn fill_vertex_count(&self) -> u32 {
        self.fill_vertices.len() as u32
    }
}

// ─── CPU path encoder ───────────────────────────────────────

use crate::scene::layer::ShapeDrawCommand;

/// Encode a set of ShapeDrawCommands into GPU-ready tessellation data.
///
/// Fills:   Lyon CPU tessellation (correct for concave polygons).
/// Strokes: Rive-style GPU compute segments (De Casteljau on GPU).
pub fn encode_paths(
    commands: &[ShapeDrawCommand],
    world_transform: &Mat2D,
) -> EncodedPathData {
    let mut fill_vertices: Vec<GpuVertex> = Vec::new();
    let mut fill_indices: Vec<u32> = Vec::new();
    let mut segments: Vec<GpuCubicSegment> = Vec::new();
    let mut stroke_indices: Vec<u32> = Vec::new();
    let mut paints: Vec<GpuPaint> = Vec::new();
    let mut stroke_vertex_cursor: u32 = 0; // relative stroke vertex cursor

    // ── First pass: fills (lyon CPU) ────────────────────────
    for cmd in commands {
        let (color, is_stroke, _) = extract_paint_info(&cmd.paint);
        if is_stroke { continue; }

        let paint_index = paints.len() as u32;
        paints.push(build_gpu_paint(&cmd.paint, world_transform));

        let path = cmd.path.transform(world_transform);
        let fill_rule = match &cmd.paint {
            crate::scene::layer::ShapePaint::SolidFill { fill_rule, .. } => *fill_rule,
            crate::scene::layer::ShapePaint::GradientFill { fill_rule, .. } => *fill_rule,
            _ => crate::geometry::path::FillRule::NonZero,
        };
        let opacity = match &cmd.paint {
            crate::scene::layer::ShapePaint::SolidFill { opacity, .. } => *opacity,
            crate::scene::layer::ShapePaint::GradientFill { opacity, .. } => *opacity,
            _ => 1.0,
        };

        let vertex_start = fill_vertices.len() as u32;
        let (verts, idxs) = lyon_fill_path(&path, color, paint_index, opacity, fill_rule, vertex_start);
        fill_vertices.extend(verts);
        fill_indices.extend(idxs);
    }

    // ── Second pass: strokes (GPU compute) ──────────────────
    // stroke first_vertex_idx is offset by fill_vertex_count so compute shader
    // writes to the correct absolute position in the shared vertex buffer.
    let fill_count = fill_vertices.len() as u32;

    for cmd in commands {
        let (color, is_stroke, half_width) = extract_paint_info(&cmd.paint);
        if !is_stroke { continue; }

        let paint_index = paints.len() as u32;
        paints.push(build_gpu_paint(&cmd.paint, world_transform));

        let path = cmd.path.transform(world_transform);

        for contour in iter_contours(&path) {
            if contour.cubics.is_empty() { continue; }
            stroke_vertex_cursor = encode_stroke_contour(
                &contour, color, half_width, paint_index,
                fill_count,
                &mut segments, &mut stroke_indices,
                stroke_vertex_cursor,
            );
        }
    }

    if paints.is_empty() {
        paints.push(GpuPaint::default());
    }

    EncodedPathData {
        fill_vertices,
        fill_indices,
        segments,
        stroke_indices,
        stroke_vertex_count: stroke_vertex_cursor,
        paints,
    }
}

// ─── Lyon fill tessellation ─────────────────────────────────

/// CPU fill tessellation using Lyon (correct for concave polygons).
fn lyon_fill_path(
    path: &RawPath,
    color: Color,
    paint_index: u32,
    opacity: f32,
    fill_rule: crate::geometry::path::FillRule,
    vertex_start: u32,
) -> (Vec<GpuVertex>, Vec<u32>) {
    use lyon::path::Path;
    use lyon::math::point;
    use lyon::tessellation::{FillTessellator, FillOptions, VertexBuffers, BuffersBuilder, FillVertex};

    let mut builder = Path::builder();
    let mut point_idx = 0;
    let mut has_open = false;

    for verb in &path.verbs {
        match verb {
            PathVerb::Move => {
                if has_open {
                    builder.end(false);
                }
                let p = path.points[point_idx];
                builder.begin(point(p.x, p.y));
                has_open = true;
                point_idx += 1;
            }
            PathVerb::Line => {
                let p = path.points[point_idx];
                builder.line_to(point(p.x, p.y));
                point_idx += 1;
            }
            PathVerb::Quad => {
                let c = path.points[point_idx];
                let e = path.points[point_idx + 1];
                builder.quadratic_bezier_to(point(c.x, c.y), point(e.x, e.y));
                point_idx += 2;
            }
            PathVerb::Cubic => {
                let c1 = path.points[point_idx];
                let c2 = path.points[point_idx + 1];
                let e = path.points[point_idx + 2];
                builder.cubic_bezier_to(
                    point(c1.x, c1.y),
                    point(c2.x, c2.y),
                    point(e.x, e.y),
                );
                point_idx += 3;
            }
            PathVerb::Close => {
                if has_open {
                    builder.end(true);
                    has_open = false;
                }
            }
        }
    }
    if has_open {
        builder.end(false);
    }

    let lyon_path = builder.build();

    let lyon_rule = match fill_rule {
        crate::geometry::path::FillRule::EvenOdd => lyon::tessellation::FillRule::EvenOdd,
        crate::geometry::path::FillRule::NonZero => lyon::tessellation::FillRule::NonZero,
    };

    let r = color.r;
    let g = color.g;
    let b = color.b;
    let a = color.a * opacity;

    let mut tessellator = FillTessellator::new();
    let mut geometry: VertexBuffers<GpuVertex, u32> = VertexBuffers::new();

    {
        let mut buf_builder = BuffersBuilder::new(&mut geometry, |vertex: FillVertex| {
            let pos = vertex.position();
            GpuVertex {
                position: [pos.x, pos.y],
                uv: [0.0, 0.0],
                color: [r, g, b, a],
                world_pos: [pos.x, pos.y],
                paint_index,
                _pad: 0,
            }
        });
        let _ = tessellator.tessellate_path(
            &lyon_path,
            &FillOptions::default()
                .with_fill_rule(lyon_rule)
                .with_tolerance(0.5),
            &mut buf_builder,
        );
    }

    let indices: Vec<u32> = geometry.indices.iter()
        .map(|&i| i as u32 + vertex_start)
        .collect();

    (geometry.vertices, indices)
}

// ─── Contour extraction ─────────────────────────────────────

struct Contour {
    cubics: Vec<[Vec2D; 4]>,
    closed: bool,
}

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
                let p1 = current_pos.lerp(p3, 1.0 / 3.0);
                let p2 = current_pos.lerp(p3, 2.0 / 3.0);
                cubics.push([current_pos, p1, p2, p3]);
                current_pos = p3;
            }
            PathVerb::Quad => {
                let q1 = path.points[point_idx];
                let q2 = path.points[point_idx + 1];
                point_idx += 2;
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

// ─── Stroke encoding (offset quad strip, GPU compute) ───────

/// Encode a stroked contour as GPU compute segments.
/// `fill_count` is the absolute fill vertex offset in the shared buffer.
fn encode_stroke_contour(
    contour: &Contour,
    color: Color,
    half_width: f32,
    paint_index: u32,
    fill_count: u32,        // absolute offset for fill region
    segments: &mut Vec<GpuCubicSegment>,
    indices: &mut Vec<u32>,
    mut stroke_vertex_cursor: u32,  // relative stroke cursor
) -> u32 {
    let strip_start = stroke_vertex_cursor;
    let mut total_pairs: u32 = 0;

    for (i, cubic) in contour.cubics.iter().enumerate() {
        let seg_count = wang_cubic_segment_count(
            cubic[0], cubic[1], cubic[2], cubic[3], 1.0,
        ).max(1);

        let is_last = i == contour.cubics.len() - 1;
        let num_new = if is_last { seg_count + 1 } else { seg_count };

        // Absolute vertex buffer position = fill_count + stroke_vertex_cursor
        segments.push(GpuCubicSegment {
            p0: [cubic[0].x, cubic[0].y],
            p1: [cubic[1].x, cubic[1].y],
            p2: [cubic[2].x, cubic[2].y],
            p3: [cubic[3].x, cubic[3].y],
            color: [color.r, color.g, color.b, color.a],
            normal_offset: half_width,
            segment_count: seg_count,
            first_vertex_idx: fill_count + stroke_vertex_cursor,
            flags: 1 | if is_last { 2 } else { 0 },
            paint_index,
            _pad: [0; 3],
        });
        stroke_vertex_cursor += num_new;

        segments.push(GpuCubicSegment {
            p0: [cubic[0].x, cubic[0].y],
            p1: [cubic[1].x, cubic[1].y],
            p2: [cubic[2].x, cubic[2].y],
            p3: [cubic[3].x, cubic[3].y],
            color: [color.r, color.g, color.b, color.a],
            normal_offset: -half_width,
            segment_count: seg_count,
            first_vertex_idx: fill_count + stroke_vertex_cursor,
            flags: 1 | if is_last { 2 } else { 0 },
            paint_index,
            _pad: [0; 3],
        });
        stroke_vertex_cursor += num_new;

        total_pairs += num_new;
    }

    // Quad-strip indices into stroke region (absolute: fill_count + relative)
    if total_pairs >= 2 {
        for i in 0..total_pairs - 1 {
            let base = fill_count + strip_start;
            let l0 = base + i;
            let r0 = base + total_pairs + i;
            let l1 = base + i + 1;
            let r1 = base + total_pairs + i + 1;

            indices.push(l0);
            indices.push(r0);
            indices.push(l1);

            indices.push(r0);
            indices.push(r1);
            indices.push(l1);
        }
    }

    stroke_vertex_cursor
}

// ─── Paint helpers ──────────────────────────────────────────

use crate::scene::layer::ShapePaint;
use crate::lottie::model::{GradientType, GradientColors};

fn build_gpu_paint(paint: &ShapePaint, world_transform: &Mat2D) -> GpuPaint {
    match paint {
        ShapePaint::SolidFill { color, opacity, .. } => {
            let mut p = GpuPaint::default();
            p.paint_type = 0;
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

    let ws = world_transform.transform_point(start);
    let we = world_transform.transform_point(end);
    p.grad_start = [ws.x, ws.y];
    p.grad_end = [we.x, we.y];

    let count = colors.color_count.min(MAX_GRADIENT_STOPS);
    p.stop_count = count as u32;

    let opacity_start = colors.color_count * 4;
    let opacity_data = if opacity_start < colors.colors.len() {
        &colors.colors[opacity_start..]
    } else {
        &[]
    };
    let opacity_count = opacity_data.len() / 2;

    for i in 0..count {
        let base = i * 4;
        if base + 3 < colors.colors.len() {
            let stop_pos = colors.colors[base];
            p.stops[i] = stop_pos;
            let r = colors.colors[base + 1];
            let g = colors.colors[base + 2];
            let b = colors.colors[base + 3];
            let a = sample_opacity_stops(opacity_data, opacity_count, stop_pos);
            p.stop_colors[i] = [r, g, b, a];
        }
    }

    p
}

fn sample_opacity_stops(data: &[f32], count: usize, pos: f32) -> f32 {
    if count == 0 || data.len() < 2 { return 1.0; }

    let first_off = data[0];
    let first_a = data[1];
    if pos <= first_off { return first_a; }

    let last_off = data[(count - 1) * 2];
    let last_a = data[(count - 1) * 2 + 1];
    if pos >= last_off { return last_a; }

    for i in 0..count - 1 {
        let o0 = data[i * 2];
        let a0 = data[i * 2 + 1];
        let o1 = data[(i + 1) * 2];
        let a1 = data[(i + 1) * 2 + 1];
        if pos >= o0 && pos <= o1 {
            let t = if (o1 - o0) > 1e-6 { (pos - o0) / (o1 - o0) } else { 0.0 };
            return a0 + (a1 - a0) * t;
        }
    }

    last_a
}

fn extract_paint_info(paint: &ShapePaint) -> (Color, bool, f32) {
    match paint {
        ShapePaint::SolidFill { color, opacity, .. } => {
            (color.with_opacity(*opacity), false, 0.0)
        }
        ShapePaint::SolidStroke { color, opacity, width, .. } => {
            (color.with_opacity(*opacity), true, *width * 0.5)
        }
        ShapePaint::GradientFill { opacity, .. } => {
            (Color::WHITE.with_opacity(*opacity), false, 0.0)
        }
        ShapePaint::GradientStroke { opacity, width, .. } => {
            (Color::WHITE.with_opacity(*opacity), true, *width * 0.5)
        }
    }
}
