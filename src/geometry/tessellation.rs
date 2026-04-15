//! GPU tessellation data structures (Rive-style).
//!
//! Fills:   GPU cubic edge tessellation + GPU index topology.
//!          Simple fan-safe contours use midpoint fans (direct draw).
//!          Concave contours use stencil-based midpoint fans (stencil + cover).
//!          No CPU fallback — all topology is generated on the GPU.
//! Strokes: GPU compute shader (De Casteljau) → Rive-style offset quad-strips.
//!
//! Instancing:
//!   Each GpuVertex carries a `sprite_index` referencing the per-sprite transform
//!   in the GPU storage buffer.  The vertex shader applies that transform so paths
//!   are kept in LOCAL space and tessellated only ONCE per unique animation frame.
//!
//!   Synced batch (N sprites on same frame):
//!     • Tessellate sprite[0] → V verts, I indices, K paints.
//!     • Memcpy × N, updating sprite_index per copy.  Index offsets added per copy.
//!     • Only K paints uploaded (shared; gradient coords are in local space and
//!       the fragment shader transforms them using sprite_transforms[sprite_index]).
//!
//!   Unsynced batch (N sprites on different frames):
//!     • Tessellate each sprite, concatenate. Each group appends its own K paints.

use super::math::{wang_cubic_segment_count, Color, Mat2D, Vec2D};
use super::path::{PathVerb, RawPath};
use bytemuck::{Pod, Zeroable};

// ─── GPU vertex (output of compute, input to draw) ──────────

/// GPU vertex consumed by the draw shader.
/// `position` is in LOCAL (artboard) space.
/// The vertex shader applies `sprite_transforms[sprite_index]` to produce world pos.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 2],  // LOCAL space position
    pub uv: [f32; 2],        // texture coords / stroke AA side
    pub color: [f32; 4],     // solid color (used when paint_type=0)
    pub world_pos: [f32; 2], // computed by vertex shader; unused on CPU upload
    pub paint_index: u32,    // index into paint storage buffer
    pub sprite_index: u32,   // index into sprite_transforms storage buffer
}

// ─── Sprite transform (GPU storage buffer element) ──────────

/// Per-sprite 2D affine transform stored in the GPU storage buffer.
/// Values: [a, b, c, d, tx, ty, _pad, _pad] matching Mat2D::values layout.
/// Transform point:  x' = a*x + c*y + tx,  y' = b*x + d*y + ty
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SpriteTransform {
    pub values: [f32; 8],
}

impl SpriteTransform {
    pub fn from_mat2d(m: &Mat2D) -> Self {
        let v = m.values;
        Self {
            values: [v[0], v[1], v[2], v[3], v[4], v[5], 0.0, 0.0],
        }
    }

    pub fn identity() -> Self {
        Self {
            values: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        }
    }
}

// ─── Compute shader inputs ──────────────────────────────────

pub const MAX_GRADIENT_STOPS: usize = 16;

/// GPU paint descriptor read by the fragment shader.
/// Gradient `grad_start` / `grad_end` are stored in LOCAL (artboard) space.
/// The fragment shader transforms them to world space using `sprite_transforms[sprite_index]`.
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
    pub first_vertex_idx: u32,
    pub flags: u32,
    pub paint_index: u32,
    pub sprite_index: u32,
    pub _pad: [u32; 2],
}

/// Per-dispatch metadata for the compute shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TessellateUniforms {
    pub total_vertices: u32,
    pub total_segments: u32,
    pub _pad: [u32; 2],
}

// ─── GPU fill tessellation (Rive midpoint fan) ─────────────
// CPU pushes raw contour/cubic data to GPU. The compute shader tessellates
// curves and writes edge vertices. CPU builds either midpoint-fan indices for
// fan-safe contours or interior triangulation indices for concave contours.

/// Per-contour descriptor for GPU fill tessellation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuFillContour {
    pub midpoint: [f32; 2],
    pub first_cubic: u32,
    pub cubic_count: u32,
    pub first_vertex: u32,
    pub total_edge_verts: u32,
    pub first_index: u32,
    pub simple_fill: u32,
    pub color: [f32; 4],
    pub paint_index: u32,
    pub sprite_index: u32,
    pub _pad1: [u32; 2],
}

/// Per-cubic descriptor for GPU fill tessellation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuFillCubic {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub p2: [f32; 2],
    pub p3: [f32; 2],
    pub segment_count: u32,
    pub first_edge_vertex: u32,
    pub _pad: [u32; 2],
}

/// Uniforms for the GPU fill tessellation compute shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct FillTessUniforms {
    pub total_cubics: u32,
    pub total_contours: u32,
    pub _pad: [u32; 2],
}

// ─── Encoded path data ──────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComplexFillRule {
    NonZero,
    EvenOdd,
}

impl Default for ComplexFillRule {
    fn default() -> Self {
        Self::NonZero
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ComplexFillDraw {
    pub stencil_index_start: u32,
    pub stencil_index_count: u32,
    pub cover_index_start: u32,
    pub cover_index_count: u32,
    pub fill_rule: ComplexFillRule,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimpleDrawSource {
    Fill,
    Stroke,
}

#[derive(Clone, Copy, Debug)]
pub enum EncodedDraw {
    Simple {
        source: SimpleDrawSource,
        index_start: u32,
        index_count: u32,
    },
    Complex(ComplexFillDraw),
}

pub struct EncodedPathData {
    // GPU fill tessellation data (Rive midpoint fan)
    pub fill_contours: Vec<GpuFillContour>,
    pub fill_cubics: Vec<GpuFillCubic>,
    pub fill_vertex_count: u32, // total vertices the compute shader will write
    pub fill_indices: Vec<u32>, // fill indices over GPU-generated edge vertices
    pub complex_fill_indices: Vec<u32>,
    pub complex_fill_draws: Vec<ComplexFillDraw>,
    pub ordered_draws: Vec<EncodedDraw>,
    // Pre-populated fill vertices (midpoints written by CPU, edges by GPU)
    pub fill_vertices: Vec<GpuVertex>,

    // Stroke GPU tessellation (existing)
    pub segments: Vec<GpuCubicSegment>,
    pub stroke_indices: Vec<u32>,
    pub stroke_vertex_count: u32,
    pub paints: Vec<GpuPaint>,
    /// Number of GPU instances to draw. For synced sprites this is N (the sprite
    /// count); for unsynced it is 1 (transforms are already baked per-vertex).
    pub instance_count: u32,
}

impl EncodedPathData {
    pub fn empty() -> Self {
        Self {
            fill_contours: Vec::new(),
            fill_cubics: Vec::new(),
            fill_vertex_count: 0,
            fill_indices: Vec::new(),
            complex_fill_indices: Vec::new(),
            complex_fill_draws: Vec::new(),
            ordered_draws: Vec::new(),
            fill_vertices: Vec::new(),
            segments: Vec::new(),
            stroke_indices: Vec::new(),
            stroke_vertex_count: 0,
            paints: Vec::new(),
            instance_count: 1,
        }
    }

    pub fn total_vertex_count(&self) -> u32 {
        self.fill_vertex_count + self.stroke_vertex_count
    }
    pub fn total_index_count(&self) -> u32 {
        (self.fill_indices.len() + self.stroke_indices.len() + self.complex_fill_indices.len())
            as u32
    }
    pub fn combined_indices(&self) -> Vec<u32> {
        let mut v = self.fill_indices.clone();
        v.extend_from_slice(&self.stroke_indices);
        v.extend_from_slice(&self.complex_fill_indices);
        v
    }
}

// ─── Public API ─────────────────────────────────────────────

use crate::scene::layer::ShapeDrawCommand;

/// Encode draw commands from multiple sprites for GPU rendering.
///
/// `sprite_groups`: slice of `(sprite_index, draws)` where paths are in LOCAL space.
/// `transforms`: per-sprite world transforms (index matches `sprite_index`).
///
/// Synced optimisation: if all groups have the same number of draw commands (same
/// animation frame), tessellates only the first group. Instead of copying vertices
/// N times, we use GPU instancing — `draw_indexed(indices, 0, 0..N)` — and the
/// vertex shader selects the transform via `instance_index`.
///
/// Returns `EncodedPathData` with `instance_count` set for the renderer.
pub fn encode_paths_instanced(
    sprite_groups: &[(u32, Vec<ShapeDrawCommand>)],
    transforms: &[Mat2D],
    batch_synced: bool,
) -> EncodedPathData {
    if sprite_groups.is_empty() {
        return EncodedPathData::empty();
    }

    if batch_synced && transforms.len() > 1 {
        encode_synced(sprite_groups, transforms.len() as u32)
    } else {
        encode_unsynced(sprite_groups)
    }
}

/// Legacy single-sprite encode (no sprite_index needed, world transform pre-applied).
/// Kept for the old single-sprite code path; sprite_index=0 for all vertices.
pub fn encode_paths(commands: &[ShapeDrawCommand], _world_transform: &Mat2D) -> EncodedPathData {
    let group = vec![(0u32, commands.to_vec())];
    encode_paths_instanced(&group, &[Mat2D::identity()], false)
}
fn remap_local_draw(
    draw: EncodedDraw,
    fill_index_offset: u32,
    stroke_index_offset: u32,
    complex_index_offset: u32,
) -> EncodedDraw {
    match draw {
        EncodedDraw::Simple {
            source,
            index_start,
            index_count,
        } => {
            let base = match source {
                SimpleDrawSource::Fill => fill_index_offset,
                SimpleDrawSource::Stroke => stroke_index_offset,
            };
            EncodedDraw::Simple {
                source,
                index_start: index_start + base,
                index_count,
            }
        }
        EncodedDraw::Complex(mut draw) => {
            draw.stencil_index_start += complex_index_offset;
            draw.cover_index_start += complex_index_offset;
            EncodedDraw::Complex(draw)
        }
    }
}

// ─── Synced path: tessellate once, copy N times ─────────────

fn encode_synced(
    sprite_groups: &[(u32, Vec<ShapeDrawCommand>)],
    instance_count: u32,
) -> EncodedPathData {
    let (_base_idx, base_cmds) = &sprite_groups[0];

    // Tessellate only sprite[0] — sprite_index is 0; the vertex shader will
    // add instance_index to get the actual sprite transform index.
    let (
        base_verts,
        base_indices,
        base_complex_indices,
        base_complex_draws,
        base_ordered_draws,
        base_fill_contours,
        base_fill_cubics,
        base_paints,
        base_segs,
        base_stroke_idx,
        base_stroke_verts,
    ) = tessellate_sprite_fills(base_cmds, 0);

    // NO vertex copies — GPU instancing handles the N sprites.
    // The vertex shader reads: sprite_transforms[vertex.sprite_index + instance_index]

    let stroke_index_offset = base_indices.len() as u32;
    let ordered_draws = base_ordered_draws
        .into_iter()
        .map(|draw| remap_local_draw(draw, 0, stroke_index_offset, 0))
        .collect();

    let paints = if base_paints.is_empty() {
        vec![GpuPaint::default()]
    } else {
        base_paints
    };

    EncodedPathData {
        fill_contours: base_fill_contours,
        fill_cubics: base_fill_cubics,
        fill_vertex_count: base_verts.len() as u32,
        fill_vertices: base_verts,
        fill_indices: base_indices,
        complex_fill_indices: base_complex_indices,
        complex_fill_draws: base_complex_draws,
        ordered_draws,
        segments: base_segs,
        stroke_indices: base_stroke_idx,
        stroke_vertex_count: base_stroke_verts,
        paints,
        instance_count,
    }
}

// —— Unsynced path: concatenate all sprites —————————
fn encode_unsynced(sprite_groups: &[(u32, Vec<ShapeDrawCommand>)]) -> EncodedPathData {
    let mut fill_vertices: Vec<GpuVertex> = Vec::new();
    let mut fill_indices: Vec<u32> = Vec::new();
    let mut all_complex_fill_indices: Vec<u32> = Vec::new();
    let mut all_complex_fill_draws: Vec<ComplexFillDraw> = Vec::new();
    let mut all_fill_contours: Vec<GpuFillContour> = Vec::new();
    let mut all_fill_cubics: Vec<GpuFillCubic> = Vec::new();
    let mut all_paints: Vec<GpuPaint> = Vec::new();
    let mut pending_strokes: Vec<(
        Vec<GpuCubicSegment>,
        Vec<u32>,
        u32,
        u32,
        Vec<EncodedDraw>,
        u32,
        u32,
    )> = Vec::new();

    for (sprite_idx, cmds) in sprite_groups {
        let paint_offset = all_paints.len() as u32;
        let fill_vertex_offset = fill_vertices.len() as u32;
        let fill_index_offset = fill_indices.len() as u32;

        let (
            mut sv,
            si,
            cfi,
            mut cfd,
            local_draws,
            mut fill_contours,
            fill_cubics,
            sp,
            segs,
            s_idx,
            s_vcnt,
        ) = tessellate_sprite_fills(cmds, *sprite_idx);

        for v in &mut sv {
            v.paint_index += paint_offset;
        }

        for &idx in &si {
            fill_indices.push(idx + fill_vertex_offset);
        }

        let complex_index_offset = all_complex_fill_indices.len() as u32;
        for idx in cfi {
            all_complex_fill_indices.push(idx + fill_vertex_offset);
        }
        for draw in &mut cfd {
            draw.stencil_index_start += complex_index_offset;
            draw.cover_index_start += complex_index_offset;
        }
        all_complex_fill_draws.extend(cfd);

        let cubic_offset = all_fill_cubics.len() as u32;
        for contour in &mut fill_contours {
            contour.first_cubic += cubic_offset;
            contour.first_vertex += fill_vertex_offset;
            contour.paint_index += paint_offset;
        }
        all_fill_contours.extend(fill_contours);
        all_fill_cubics.extend(fill_cubics);

        let mut remapped_segs = segs;
        for seg in &mut remapped_segs {
            seg.paint_index += paint_offset;
        }
        pending_strokes.push((
            remapped_segs,
            s_idx,
            s_vcnt,
            sv.len() as u32,
            local_draws,
            fill_index_offset,
            complex_index_offset,
        ));

        fill_vertices.extend(sv);
        all_paints.extend(sp);
    }

    let total_fill_verts = fill_vertices.len() as u32;
    let stroke_index_base = fill_indices.len() as u32;
    let mut all_segs: Vec<GpuCubicSegment> = Vec::new();
    let mut all_stroke_idx: Vec<u32> = Vec::new();
    let mut total_stroke_verts: u32 = 0;
    let mut all_ordered_draws: Vec<EncodedDraw> = Vec::new();
    let mut stroke_base = total_fill_verts;

    for (
        mut segs,
        idxs,
        stroke_vert_count,
        local_fill_count,
        local_draws,
        fill_index_offset,
        complex_index_offset,
    ) in pending_strokes
    {
        for seg in &mut segs {
            seg.first_vertex_idx = stroke_base + (seg.first_vertex_idx - local_fill_count);
        }

        let stroke_index_offset = stroke_index_base + all_stroke_idx.len() as u32;
        for idx in idxs {
            all_stroke_idx.push(stroke_base + (idx - local_fill_count));
        }

        all_ordered_draws.extend(local_draws.into_iter().map(|draw| {
            remap_local_draw(
                draw,
                fill_index_offset,
                stroke_index_offset,
                complex_index_offset,
            )
        }));

        stroke_base += stroke_vert_count;
        total_stroke_verts += stroke_vert_count;
        all_segs.extend(segs);
    }

    if all_paints.is_empty() {
        all_paints.push(GpuPaint::default());
    }

    EncodedPathData {
        fill_contours: all_fill_contours,
        fill_cubics: all_fill_cubics,
        fill_vertex_count: fill_vertices.len() as u32,
        fill_vertices,
        fill_indices,
        complex_fill_indices: all_complex_fill_indices,
        complex_fill_draws: all_complex_fill_draws,
        ordered_draws: all_ordered_draws,
        segments: all_segs,
        stroke_indices: all_stroke_idx,
        stroke_vertex_count: total_stroke_verts,
        paints: all_paints,
        instance_count: 1,
    }
}

// —— Single-sprite tessellation —————————————————————————
/// Tessellate fills + strokes for one sprite's draw commands.
/// Returns (fill_verts, fill_indices, paints, stroke_segs, stroke_indices, stroke_vert_count).
/// All vertex positions are in LOCAL space.
fn tessellate_sprite_fills(
    commands: &[ShapeDrawCommand],
    sprite_idx: u32,
) -> (
    Vec<GpuVertex>,
    Vec<u32>,
    Vec<u32>,
    Vec<ComplexFillDraw>,
    Vec<EncodedDraw>,
    Vec<GpuFillContour>,
    Vec<GpuFillCubic>,
    Vec<GpuPaint>,
    Vec<GpuCubicSegment>,
    Vec<u32>,
    u32,
) {
    let mut fill_vertices: Vec<GpuVertex> = Vec::new();
    let mut fill_indices: Vec<u32> = Vec::new();
    let mut complex_fill_indices: Vec<u32> = Vec::new();
    let mut complex_fill_draws: Vec<ComplexFillDraw> = Vec::new();
    let mut ordered_draws: Vec<EncodedDraw> = Vec::new();
    let mut fill_contours: Vec<GpuFillContour> = Vec::new();
    let mut fill_cubics: Vec<GpuFillCubic> = Vec::new();
    let mut paints: Vec<GpuPaint> = Vec::new();
    let mut segments: Vec<GpuCubicSegment> = Vec::new();
    let mut stroke_indices: Vec<u32> = Vec::new();
    let mut stroke_vertex_cursor: u32 = 0;

    // —— Fills (GPU edge tess + CPU topology) ———————————————
    for cmd in commands {
        let (color, is_stroke, _) = extract_paint_info(&cmd.paint);
        if is_stroke {
            continue;
        }

        let paint_index = paints.len() as u32;
        paints.push(build_gpu_paint_local(&cmd.paint));

        let fill_rule = match &cmd.paint {
            ShapePaint::SolidFill { fill_rule, .. } => *fill_rule,
            ShapePaint::GradientFill { fill_rule, .. } => *fill_rule,
            _ => crate::geometry::path::FillRule::NonZero,
        };
        let contours = iter_contours(&cmd.path);
        let contour_data = collect_fill_contours(&contours);
        if contour_data.is_empty() {
            continue;
        }

        let fill_index_start = fill_indices.len() as u32;
        let complex_draw_start = complex_fill_draws.len();
        let same_winding = contours_have_uniform_winding(&contour_data);

        // NonZero + uniform winding: ALWAYS use simple midpoint fan.
        // Overlapping fan triangles all share the same winding direction,
        // so NonZero fill produces correct results even for concave shapes.
        // This matches ThorVG's approach and eliminates render-path oscillation
        // that causes visual trembling on animated concave shapes (e.g. chin).
        if fill_rule == crate::geometry::path::FillRule::NonZero && same_winding {
            encode_fill_midpoint_fans(
                &contour_data,
                color,
                paint_index,
                sprite_idx,
                &mut fill_vertices,
                &mut fill_indices,
                &mut fill_contours,
                &mut fill_cubics,
            );
            let index_count = fill_indices.len() as u32 - fill_index_start;
            if index_count > 0 {
                ordered_draws.push(EncodedDraw::Simple {
                    source: SimpleDrawSource::Fill,
                    index_start: fill_index_start,
                    index_count,
                });
            }
        } else {
            encode_complex_fill(
                &contour_data,
                color,
                paint_index,
                sprite_idx,
                fill_rule,
                &mut fill_vertices,
                &mut complex_fill_indices,
                &mut complex_fill_draws,
                &mut fill_contours,
                &mut fill_cubics,
            );
            if complex_fill_draws.len() > complex_draw_start {
                if let Some(draw) = complex_fill_draws.last().copied() {
                    ordered_draws.push(EncodedDraw::Complex(draw));
                }
            }
        }
    }

    // —— Strokes (GPU compute) ———————————————————————————————————
    let fill_count = fill_vertices.len() as u32;

    for cmd in commands {
        let (color, is_stroke, half_width) = extract_paint_info(&cmd.paint);
        if !is_stroke {
            continue;
        }

        let paint_index = paints.len() as u32;
        paints.push(build_gpu_paint_local(&cmd.paint));

        let stroke_index_start = stroke_indices.len() as u32;
        for contour in iter_contours(&cmd.path) {
            if contour.cubics.is_empty() {
                continue;
            }
            stroke_vertex_cursor = encode_stroke_contour(
                &contour,
                color,
                half_width,
                paint_index,
                sprite_idx,
                fill_count,
                &mut segments,
                &mut stroke_indices,
                stroke_vertex_cursor,
            );
        }

        let index_count = stroke_indices.len() as u32 - stroke_index_start;
        if index_count > 0 {
            ordered_draws.push(EncodedDraw::Simple {
                source: SimpleDrawSource::Stroke,
                index_start: stroke_index_start,
                index_count,
            });
        }
    }

    (
        fill_vertices,
        fill_indices,
        complex_fill_indices,
        complex_fill_draws,
        ordered_draws,
        fill_contours,
        fill_cubics,
        paints,
        segments,
        stroke_indices,
        stroke_vertex_cursor,
    )
}

// —— Fill topology selection ———————————————————————————————

#[derive(Clone)]
struct ContourFillData {
    contour: Contour,
    samples: Vec<Vec2D>,
    midpoint: Vec2D,
    signed_area: f32,
    fan_safe: bool,
}

const FILL_TOPOLOGY_NONE: u32 = 0;
const FILL_TOPOLOGY_SIMPLE: u32 = 1;
const FILL_TOPOLOGY_COMPLEX_FAN: u32 = 2;

fn collect_fill_contours(contours: &[Contour]) -> Vec<ContourFillData> {
    let mut result = Vec::new();

    for contour in contours {
        if !contour.closed || contour.cubics.is_empty() {
            continue;
        }

        let samples = normalized_polygon(&contour_fill_samples(contour));
        if samples.len() < 3 {
            continue;
        }

        let signed_area = polygon_signed_area(&samples);
        if signed_area.abs() < 1e-3 {
            continue;
        }

        let midpoint = contour_midpoint(contour);
        let fan_safe = point_in_polygon(&samples, midpoint)
            && point_in_polygon_kernel_approx(&samples, midpoint);

        result.push(ContourFillData {
            contour: contour.clone(),
            samples,
            midpoint,
            signed_area,
            fan_safe,
        });
    }

    result
}

fn contours_have_uniform_winding(contours: &[ContourFillData]) -> bool {
    let mut winding_sign = 0.0f32;
    for contour in contours {
        let sign = contour.signed_area.signum();
        if winding_sign == 0.0 {
            winding_sign = sign;
        } else if sign != winding_sign {
            return false;
        }
    }
    true
}

fn encode_fill_midpoint_fans(
    contours: &[ContourFillData],
    color: Color,
    paint_index: u32,
    sprite_index: u32,
    fill_vertices: &mut Vec<GpuVertex>,
    fill_indices: &mut Vec<u32>,
    fill_contours: &mut Vec<GpuFillContour>,
    fill_cubics: &mut Vec<GpuFillCubic>,
) {
    for contour in contours {
        let Some((_, total_edge_verts, _)) = append_fill_contour_geometry(
            contour,
            color,
            paint_index,
            sprite_index,
            fill_vertices,
            fill_contours,
            fill_cubics,
        ) else {
            continue;
        };

        if total_edge_verts < 3 {
            continue;
        }

        let first_index = fill_indices.len() as u32;
        fill_indices.resize(fill_indices.len() + total_edge_verts as usize * 3, 0);
        if let Some(fill_contour) = fill_contours.last_mut() {
            fill_contour.first_index = first_index;
            fill_contour.simple_fill = FILL_TOPOLOGY_SIMPLE;
        }
    }
}

fn encode_complex_fill(
    contours: &[ContourFillData],
    color: Color,
    paint_index: u32,
    sprite_index: u32,
    fill_rule: crate::geometry::path::FillRule,
    fill_vertices: &mut Vec<GpuVertex>,
    complex_fill_indices: &mut Vec<u32>,
    complex_fill_draws: &mut Vec<ComplexFillDraw>,
    fill_contours: &mut Vec<GpuFillContour>,
    fill_cubics: &mut Vec<GpuFillCubic>,
) {
    let stencil_index_start = complex_fill_indices.len() as u32;
    let mut any_geometry = false;
    let mut bounds = crate::geometry::math::AABB::empty();

    for contour in contours {
        for p in &contour.samples {
            bounds.expand_to_include(*p);
        }

        let Some((first_vertex, total_edge_verts, midpoint_idx)) = append_fill_contour_geometry(
            contour,
            color,
            paint_index,
            sprite_index,
            fill_vertices,
            fill_contours,
            fill_cubics,
        ) else {
            continue;
        };

        if total_edge_verts < 3 {
            continue;
        }

        any_geometry = true;

        // Stencil-based complex fills: midpoint fan is always correct.
        // The stencil accumulates winding, so overlapping/exterior fan
        // triangles cancel out — no need for ear-clipping or CPU fallback.
        let first_index = complex_fill_indices.len() as u32;
        complex_fill_indices.resize(
            complex_fill_indices.len() + total_edge_verts as usize * 3,
            first_vertex,
        );
        if let Some(fill_contour) = fill_contours.last_mut() {
            fill_contour.first_index = first_index;
            fill_contour.simple_fill = FILL_TOPOLOGY_COMPLEX_FAN;
        }
    }

    if !any_geometry || bounds.is_empty() {
        return;
    }

    let cover_vertex_start = fill_vertices.len() as u32;
    let cover_verts = [
        [bounds.min_x, bounds.min_y],
        [bounds.max_x, bounds.min_y],
        [bounds.max_x, bounds.max_y],
        [bounds.min_x, bounds.max_y],
    ];
    for pos in cover_verts {
        fill_vertices.push(GpuVertex {
            position: pos,
            uv: [1.0, 0.0], // uv.x=1.0 → full coverage (no edge AA on cover quad)
            color: [color.r, color.g, color.b, color.a],
            world_pos: [0.0, 0.0],
            paint_index,
            sprite_index,
        });
    }

    let cover_index_start = complex_fill_indices.len() as u32;
    complex_fill_indices.extend_from_slice(&[
        cover_vertex_start,
        cover_vertex_start + 1,
        cover_vertex_start + 2,
        cover_vertex_start,
        cover_vertex_start + 2,
        cover_vertex_start + 3,
    ]);

    complex_fill_draws.push(ComplexFillDraw {
        stencil_index_start,
        stencil_index_count: cover_index_start - stencil_index_start,
        cover_index_start,
        cover_index_count: 6,
        fill_rule: match fill_rule {
            crate::geometry::path::FillRule::EvenOdd => ComplexFillRule::EvenOdd,
            crate::geometry::path::FillRule::NonZero => ComplexFillRule::NonZero,
        },
    });
}

fn append_fill_contour_geometry(
    contour: &ContourFillData,
    color: Color,
    paint_index: u32,
    sprite_index: u32,
    fill_vertices: &mut Vec<GpuVertex>,
    fill_contours: &mut Vec<GpuFillContour>,
    fill_cubics: &mut Vec<GpuFillCubic>,
) -> Option<(u32, u32, u32)> {
    if contour.contour.cubics.is_empty() {
        return None;
    }

    let first_vertex = fill_vertices.len() as u32;
    let first_cubic = fill_cubics.len() as u32;

    let mut total_edge_verts = 0u32;
    for (i, cubic) in contour.contour.cubics.iter().enumerate() {
        let seg_count = wang_cubic_segment_count(
            cubic[0],
            cubic[1],
            cubic[2],
            cubic[3],
            RIVE_PARAMETRIC_PRECISION,
        )
        .max(MIN_FILL_CUBIC_SEGMENTS);
        total_edge_verts += if i == contour.contour.cubics.len() - 1 {
            seg_count + 1
        } else {
            seg_count
        };
    }

    if total_edge_verts < 3 {
        return None;
    }

    fill_vertices.resize(
        fill_vertices.len() + total_edge_verts as usize + 1,
        GpuVertex::default(),
    );

    let mut edge_cursor = 0u32;
    for (i, cubic) in contour.contour.cubics.iter().enumerate() {
        let seg_count = wang_cubic_segment_count(
            cubic[0],
            cubic[1],
            cubic[2],
            cubic[3],
            RIVE_PARAMETRIC_PRECISION,
        )
        .max(MIN_FILL_CUBIC_SEGMENTS);
        let edge_count = if i == contour.contour.cubics.len() - 1 {
            seg_count + 1
        } else {
            seg_count
        };
        fill_cubics.push(GpuFillCubic {
            p0: [cubic[0].x, cubic[0].y],
            p1: [cubic[1].x, cubic[1].y],
            p2: [cubic[2].x, cubic[2].y],
            p3: [cubic[3].x, cubic[3].y],
            segment_count: seg_count,
            first_edge_vertex: edge_cursor,
            _pad: [0; 2],
        });
        edge_cursor += edge_count;
    }

    let midpoint_idx = first_vertex + total_edge_verts;
    fill_contours.push(GpuFillContour {
        midpoint: [contour.midpoint.x, contour.midpoint.y],
        first_cubic,
        cubic_count: contour.contour.cubics.len() as u32,
        first_vertex,
        total_edge_verts,
        first_index: 0,
        simple_fill: FILL_TOPOLOGY_NONE,
        color: [color.r, color.g, color.b, color.a],
        paint_index,
        sprite_index,
        _pad1: [0; 2],
    });

    Some((first_vertex, total_edge_verts, midpoint_idx))
}

fn contour_fill_samples(contour: &Contour) -> Vec<Vec2D> {
    let mut points = Vec::new();
    if let Some(first) = contour.cubics.first() {
        points.push(first[0]);
    }

    for cubic in &contour.cubics {
        let seg_count = wang_cubic_segment_count(
            cubic[0],
            cubic[1],
            cubic[2],
            cubic[3],
            RIVE_PARAMETRIC_PRECISION,
        )
        .max(MIN_FILL_CUBIC_SEGMENTS);
        for i in 1..=seg_count {
            let t = i as f32 / seg_count as f32;
            points.push(super::math::cubic_eval(
                cubic[0], cubic[1], cubic[2], cubic[3], t,
            ));
        }
    }

    points
}

fn normalized_polygon(samples: &[Vec2D]) -> Vec<Vec2D> {
    let mut polygon = samples.to_vec();
    if polygon.len() >= 2 {
        let first = polygon[0];
        let last = polygon[polygon.len() - 1];
        if first.distance(last) <= 1e-3 {
            polygon.pop();
        }
    }
    polygon
}

fn contour_midpoint(contour: &Contour) -> Vec2D {
    if contour.cubics.is_empty() {
        return Vec2D::ZERO;
    }

    // Use area-weighted centroid for stability on concave animated shapes.
    // Falls back to simple average if the polygon is degenerate.
    let endpoints: Vec<Vec2D> = contour.cubics.iter().map(|c| c[0]).collect();
    let n = endpoints.len();
    if n < 3 {
        let mut sum = Vec2D::ZERO;
        for p in &endpoints {
            sum = sum + *p;
        }
        return sum * (1.0 / n.max(1) as f32);
    }

    let mut area = 0.0f32;
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    for i in 0..n {
        let p0 = endpoints[i];
        let p1 = endpoints[(i + 1) % n];
        let cross = p0.cross(p1);
        area += cross;
        cx += (p0.x + p1.x) * cross;
        cy += (p0.y + p1.y) * cross;
    }
    area *= 0.5;

    if area.abs() < 1e-5 {
        let mut sum = Vec2D::ZERO;
        for p in &endpoints {
            sum = sum + *p;
        }
        return sum * (1.0 / n as f32);
    }

    let factor = 1.0 / (6.0 * area);
    Vec2D::new(cx * factor, cy * factor)
}

fn point_in_polygon_kernel_approx(points: &[Vec2D], p: Vec2D) -> bool {
    for &target in points {
        if !segment_stays_inside_polygon(points, p, target) {
            return false;
        }
    }
    true
}

fn segment_stays_inside_polygon(points: &[Vec2D], a: Vec2D, b: Vec2D) -> bool {
    for step in 1..16 {
        let t = step as f32 / 16.0;
        let sample = a.lerp(b, t);
        if !point_in_polygon(points, sample) {
            return false;
        }
    }
    true
}

fn polygon_signed_area(points: &[Vec2D]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0f32;
    for i in 0..points.len() {
        let p0 = points[i];
        let p1 = points[(i + 1) % points.len()];
        area += p0.cross(p1);
    }
    area * 0.5
}

fn point_in_polygon(points: &[Vec2D], p: Vec2D) -> bool {
    if points.len() < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = points.len() - 1;
    for i in 0..points.len() {
        let pi = points[i];
        let pj = points[j];
        let dy = pj.y - pi.y;
        let denom = if dy.abs() < 1e-6 {
            if dy.is_sign_negative() {
                -1e-6
            } else {
                1e-6
            }
        } else {
            dy
        };
        let intersects =
            ((pi.y > p.y) != (pj.y > p.y)) && (p.x < (pj.x - pi.x) * (p.y - pi.y) / denom + pi.x);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Rive-quality tessellation precision (kParametricPrecision = 4).
/// Higher value = more segments = smoother curves.
/// 4.0 produces quarter-pixel accuracy, matching Rive's default.
const RIVE_PARAMETRIC_PRECISION: f32 = 4.0;
/// Minimum segments per cubic — Rive never drops below a reasonable floor.
/// 4 ensures even tiny near-linear cubics get enough segments for smooth AA.
const MIN_FILL_CUBIC_SEGMENTS: u32 = 4;

// ─── Contour extraction ─────────────────────────────────────

#[derive(Clone)]
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
                    contours.push(Contour {
                        cubics: std::mem::take(&mut cubics),
                        closed: false,
                    });
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
                    contours.push(Contour {
                        cubics: std::mem::take(&mut cubics),
                        closed: true,
                    });
                }
            }
        }
    }

    if !cubics.is_empty() {
        contours.push(Contour {
            cubics,
            closed: false,
        });
    }

    contours
}

// ─── Stroke encoding ─────────────────────────────────────────

fn encode_stroke_contour(
    contour: &Contour,
    color: Color,
    half_width: f32,
    paint_index: u32,
    sprite_index: u32,
    fill_count: u32,
    segments: &mut Vec<GpuCubicSegment>,
    indices: &mut Vec<u32>,
    mut stroke_vertex_cursor: u32,
) -> u32 {
    let strip_start = stroke_vertex_cursor;
    let mut total_pairs: u32 = 0;

    for (i, cubic) in contour.cubics.iter().enumerate() {
        let seg_count = wang_cubic_segment_count(
            cubic[0],
            cubic[1],
            cubic[2],
            cubic[3],
            RIVE_PARAMETRIC_PRECISION,
        )
        .max(1);

        let is_last = i == contour.cubics.len() - 1;
        let num_new = if is_last { seg_count + 1 } else { seg_count };

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
            sprite_index,
            _pad: [0; 2],
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
            sprite_index,
            _pad: [0; 2],
        });
        stroke_vertex_cursor += num_new;

        total_pairs += num_new;
    }

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

use crate::lottie::model::{GradientColors, GradientType};
use crate::scene::layer::ShapePaint;

/// Build a GPU paint from a ShapePaint.
/// Gradient coordinates are stored in LOCAL space; the fragment shader
/// will transform them to world space via sprite_transforms[sprite_index].
fn build_gpu_paint_local(paint: &ShapePaint) -> GpuPaint {
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
        ShapePaint::GradientFill {
            gradient_type,
            start,
            end,
            colors,
            opacity,
            ..
        } => build_gradient_paint_local(*gradient_type, *start, *end, colors, *opacity),
        ShapePaint::GradientStroke {
            gradient_type,
            start,
            end,
            colors,
            opacity,
            ..
        } => build_gradient_paint_local(*gradient_type, *start, *end, colors, *opacity),
    }
}

fn build_gradient_paint_local(
    gradient_type: GradientType,
    start: Vec2D,
    end: Vec2D,
    colors: &GradientColors,
    opacity: f32,
) -> GpuPaint {
    let mut p = GpuPaint::default();
    p.paint_type = match gradient_type {
        GradientType::Linear => 1,
        GradientType::Radial => 2,
    };
    p.opacity = opacity;
    // Keep in LOCAL space — vertex/fragment shader applies sprite_transforms[sprite_index]
    p.grad_start = [start.x, start.y];
    p.grad_end = [end.x, end.y];

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
    if count == 0 || data.len() < 2 {
        return 1.0;
    }

    let first_off = data[0];
    let first_a = data[1];
    if pos <= first_off {
        return first_a;
    }

    let last_off = data[(count - 1) * 2];
    let last_a = data[(count - 1) * 2 + 1];
    if pos >= last_off {
        return last_a;
    }

    for i in 0..count - 1 {
        let o0 = data[i * 2];
        let a0 = data[i * 2 + 1];
        let o1 = data[(i + 1) * 2];
        let a1 = data[(i + 1) * 2 + 1];
        if pos >= o0 && pos <= o1 {
            let t = if (o1 - o0) > 1e-6 {
                (pos - o0) / (o1 - o0)
            } else {
                0.0
            };
            return a0 + (a1 - a0) * t;
        }
    }

    last_a
}

fn extract_paint_info(paint: &ShapePaint) -> (Color, bool, f32) {
    match paint {
        ShapePaint::SolidFill { color, opacity, .. } => (color.with_opacity(*opacity), false, 0.0),
        ShapePaint::SolidStroke {
            color,
            opacity,
            width,
            ..
        } => (color.with_opacity(*opacity), true, *width * 0.5),
        ShapePaint::GradientFill { opacity, .. } => {
            (Color::WHITE.with_opacity(*opacity), false, 0.0)
        }
        ShapePaint::GradientStroke { opacity, width, .. } => {
            (Color::WHITE.with_opacity(*opacity), true, *width * 0.5)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::path::FillRule;
    use crate::lottie::model::BlendMode;
    use crate::scene::layer::{ShapeDrawCommand, ShapePaint};
    use std::mem::{offset_of, size_of};

    fn solid_fill(path: RawPath) -> ShapeDrawCommand {
        ShapeDrawCommand {
            path,
            paint: ShapePaint::SolidFill {
                color: Color::WHITE,
                opacity: 1.0,
                fill_rule: FillRule::NonZero,
            },
            blend_mode: BlendMode::Normal,
            sprite_index: 0,
        }
    }

    fn concave_path() -> RawPath {
        let mut path = RawPath::new();
        path.move_to(0.0, 0.0);
        path.line_to(4.0, 0.0);
        path.line_to(4.0, 2.0);
        path.line_to(2.0, 1.0);
        path.line_to(4.0, 4.0);
        path.line_to(0.0, 4.0);
        path.close();
        path
    }

    #[test]
    fn gpu_fill_contour_layout_matches_wgsl() {
        assert_eq!(size_of::<GpuFillContour>(), 64);
        assert_eq!(offset_of!(GpuFillContour, midpoint), 0);
        assert_eq!(offset_of!(GpuFillContour, first_cubic), 8);
        assert_eq!(offset_of!(GpuFillContour, cubic_count), 12);
        assert_eq!(offset_of!(GpuFillContour, first_vertex), 16);
        assert_eq!(offset_of!(GpuFillContour, total_edge_verts), 20);
        assert_eq!(offset_of!(GpuFillContour, first_index), 24);
        assert_eq!(offset_of!(GpuFillContour, simple_fill), 28);
        assert_eq!(offset_of!(GpuFillContour, color), 32);
        assert_eq!(offset_of!(GpuFillContour, paint_index), 48);
        assert_eq!(offset_of!(GpuFillContour, sprite_index), 52);
        assert_eq!(offset_of!(GpuFillContour, _pad1), 56);
    }

    #[test]
    fn gpu_struct_sizes_match_shader_expectations() {
        assert_eq!(size_of::<GpuVertex>(), 48);
        assert_eq!(size_of::<GpuCubicSegment>(), 80);
        assert_eq!(size_of::<GpuFillCubic>(), 48);
        assert_eq!(size_of::<GpuPaint>(), 352);
        assert_eq!(size_of::<SpriteTransform>(), 32);
        assert_eq!(size_of::<TessellateUniforms>(), 16);
        assert_eq!(size_of::<FillTessUniforms>(), 16);
    }

    #[test]
    fn rejects_concave_polygon_as_fan_safe() {
        let polygon = vec![
            Vec2D::new(0.0, 0.0),
            Vec2D::new(4.0, 0.0),
            Vec2D::new(4.0, 2.0),
            Vec2D::new(2.0, 1.0),
            Vec2D::new(4.0, 4.0),
            Vec2D::new(0.0, 4.0),
        ];

        let midpoint = Vec2D::new(2.3333333, 1.8333334);
        assert!(point_in_polygon(&polygon, midpoint));
        assert!(!point_in_polygon_kernel_approx(&polygon, midpoint));
    }

    #[test]
    fn preserves_draw_order_across_simple_and_complex_fills() {
        let mut simple = RawPath::new();
        simple.add_rect(10.0, 10.0, 12.0, 12.0);

        let encoded = encode_paths_instanced(
            &[(
                0,
                vec![solid_fill(concave_path()), solid_fill(simple.clone())],
            )],
            &[Mat2D::identity()],
            false,
        );

        assert_eq!(encoded.ordered_draws.len(), 2);
        assert!(matches!(encoded.ordered_draws[0], EncodedDraw::Complex(_)));
        assert!(matches!(
            encoded.ordered_draws[1],
            EncodedDraw::Simple {
                source: SimpleDrawSource::Fill,
                index_count,
                ..
            } if index_count > 0
        ));

        let reversed = encode_paths_instanced(
            &[(0, vec![solid_fill(simple), solid_fill(concave_path())])],
            &[Mat2D::identity()],
            false,
        );

        assert_eq!(reversed.ordered_draws.len(), 2);
        assert!(matches!(
            reversed.ordered_draws[0],
            EncodedDraw::Simple {
                source: SimpleDrawSource::Fill,
                index_count,
                ..
            } if index_count > 0
        ));
        assert!(matches!(reversed.ordered_draws[1], EncodedDraw::Complex(_)));
    }
}
