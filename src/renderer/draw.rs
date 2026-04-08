use crate::geometry::math::{Color, Mat2D, Vec2D};
use crate::geometry::tessellation::{Tessellator, TessellatedMesh};
use crate::lottie::model::{BlendMode, GradientType, GradientColors};
use crate::scene::layer::{ShapeDrawCommand, ShapePaint};

// ─── Rive-style Draw types ───────────────────────────────────

/// Draw type (matches Rive DrawType)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawType {
    MidpointFanPatches,
    OuterCurvePatches,
    InteriorTriangles,
    ImageRect,
    ImageMesh,
}

/// Draw contents classification (matches Rive DrawContents)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawContents {
    Opaque,
    StrokeOpaque,
    Feathered,
    StrokeFeathered,
}

/// A single draw command before batching
#[derive(Debug)]
pub struct Draw {
    pub path_id: u32,
    pub draw_type: DrawType,
    pub draw_contents: DrawContents,
    pub blend_mode: BlendMode,
    pub z_index: u32,
    pub sort_key: u64,
}

/// A batched draw call ready for GPU submission (Rive DrawBatch equivalent)
#[derive(Debug)]
pub struct DrawBatch {
    pub mesh: TessellatedMesh,
    pub draw_type: DrawType,
    pub blend_mode: BlendMode,
    pub element_count: u32,
    pub base_element: u32,
}

// ─── Sort key builder (Rive-style 63-bit sort key) ───────────

/// Build a sort key for draw ordering and batching.
/// Higher priority fields occupy higher bits.
/// Format: [blendMode(4) | drawContents(2) | zIndex(16) | drawType(3) | padding]
pub fn build_sort_key(
    blend_mode: BlendMode,
    draw_contents: DrawContents,
    z_index: u32,
    draw_type: DrawType,
) -> u64 {
    let bm = (blend_mode as u64) & 0xF;
    let dc = (draw_contents as u64) & 0x3;
    let zi = (z_index as u64) & 0xFFFF;
    let dt = (draw_type as u64) & 0x7;

    (bm << 21) | (dc << 19) | (zi << 3) | dt
}

// ─── Draw batcher (Rive-style) ──────────────────────────────

/// Process scene draw commands into GPU-ready batched draws.
/// Implements Rive-style sort-key batching for minimum draw calls.
pub struct DrawBatcher {
    tessellator: Tessellator,
}

impl DrawBatcher {
    pub fn new() -> Self {
        Self {
            tessellator: Tessellator::new(),
        }
    }

    /// Convert shape draw commands into batched GPU meshes.
    /// Sorts by sort key, merges adjacent compatible draws.
    pub fn batch(
        &self,
        commands: &[ShapeDrawCommand],
        world_transform: &Mat2D,
    ) -> Vec<DrawBatch> {
        if commands.is_empty() {
            return Vec::new();
        }

        // Phase 1: Tessellate all commands into individual meshes with metadata
        let mut draw_items: Vec<(u64, BlendMode, DrawType, TessellatedMesh)> = Vec::new();

        for (i, cmd) in commands.iter().enumerate() {
            let (mesh, blend_mode, draw_type) = self.tessellate_command(cmd, world_transform, i as u32);
            if mesh.is_empty() {
                continue;
            }
            let sort_key = build_sort_key(blend_mode, DrawContents::Opaque, i as u32, draw_type);
            draw_items.push((sort_key, blend_mode, draw_type, mesh));
        }

        // Phase 2: Sort by sort key
        draw_items.sort_by_key(|(key, _, _, _)| *key);

        // Phase 3: Merge adjacent compatible draws into batches
        let mut batches: Vec<DrawBatch> = Vec::new();

        for (_, blend_mode, draw_type, mesh) in draw_items {
            // Try to merge with last batch if same blend mode and draw type
            let can_merge = batches.last().map_or(false, |last| {
                last.blend_mode == blend_mode && last.draw_type == draw_type
            });

            if can_merge {
                let last = batches.last_mut().unwrap();
                last.element_count += mesh.index_count();
                last.mesh.append(&mesh);
            } else {
                let element_count = mesh.index_count();
                batches.push(DrawBatch {
                    mesh,
                    draw_type,
                    blend_mode,
                    element_count,
                    base_element: 0,
                });
            }
        }

        batches
    }

    fn tessellate_command(
        &self,
        cmd: &ShapeDrawCommand,
        world_transform: &Mat2D,
        _index: u32,
    ) -> (TessellatedMesh, BlendMode, DrawType) {
        let cmd_blend_mode = cmd.blend_mode;
        match &cmd.paint {
            ShapePaint::SolidFill { color, opacity, fill_rule } => {
                let c = color.with_opacity(*opacity);
                let mesh = self.tessellator.tessellate_fill(
                    &cmd.path, *fill_rule, c, world_transform,
                );
                (mesh, cmd_blend_mode, DrawType::MidpointFanPatches)
            }
            ShapePaint::SolidStroke { color, opacity, width, cap, join, miter_limit } => {
                let c = color.with_opacity(*opacity);
                let mesh = self.tessellator.tessellate_stroke(
                    &cmd.path, *width, *cap, *join, *miter_limit, c, world_transform,
                );
                (mesh, cmd_blend_mode, DrawType::OuterCurvePatches)
            }
            ShapePaint::GradientFill { gradient_type, start, end, colors, opacity, fill_rule } => {
                // Tessellate with per-vertex gradient colors by sampling the gradient
                // based on each vertex's position relative to gradient start/end
                let mesh = self.tessellator.tessellate_fill(
                    &cmd.path,
                    *fill_rule,
                    Color::WHITE, // placeholder, will be overwritten
                    world_transform,
                );
                let mesh = apply_gradient_to_mesh(
                    mesh, *gradient_type, *start, *end, colors, *opacity, world_transform,
                );
                (mesh, cmd_blend_mode, DrawType::MidpointFanPatches)
            }
            ShapePaint::GradientStroke { gradient_type, start, end, colors, opacity, width, cap, join, miter_limit } => {
                let mesh = self.tessellator.tessellate_stroke(
                    &cmd.path, *width, *cap, *join, *miter_limit,
                    Color::WHITE, // placeholder
                    world_transform,
                );
                let mesh = apply_gradient_to_mesh(
                    mesh, *gradient_type, *start, *end, colors, *opacity, world_transform,
                );
                (mesh, cmd_blend_mode, DrawType::OuterCurvePatches)
            }
        }
    }
}

/// Sample a Lottie gradient at parameter t (0..1).
/// Format: [offset, R, G, B, offset, R, G, B, ...] repeated color_count times.
fn sample_gradient(colors: &GradientColors, t: f32) -> Color {
    let stride = 4; // offset + R + G + B
    let count = colors.color_count;
    if count == 0 || colors.colors.len() < stride {
        return Color::WHITE;
    }

    let t = t.clamp(0.0, 1.0);

    // Find surrounding stops
    let mut prev_idx = 0;
    let mut next_idx = 0;
    for i in 0..count {
        let base = i * stride;
        if base >= colors.colors.len() { break; }
        let offset = colors.colors[base];
        if offset <= t {
            prev_idx = i;
        }
        if offset >= t {
            next_idx = i;
            break;
        }
        next_idx = i;
    }

    let prev_base = prev_idx * stride;
    let next_base = next_idx * stride;

    if prev_idx == next_idx || next_base + 3 >= colors.colors.len() || prev_base + 3 >= colors.colors.len() {
        let base = prev_base.min(colors.colors.len().saturating_sub(stride));
        if base + 3 < colors.colors.len() {
            return Color::new(colors.colors[base + 1], colors.colors[base + 2], colors.colors[base + 3], 1.0);
        }
        return Color::WHITE;
    }

    let prev_offset = colors.colors[prev_base];
    let next_offset = colors.colors[next_base];
    let segment_t = if (next_offset - prev_offset).abs() < 1e-6 {
        0.0
    } else {
        (t - prev_offset) / (next_offset - prev_offset)
    };

    Color::new(
        colors.colors[prev_base + 1] + (colors.colors[next_base + 1] - colors.colors[prev_base + 1]) * segment_t,
        colors.colors[prev_base + 2] + (colors.colors[next_base + 2] - colors.colors[prev_base + 2]) * segment_t,
        colors.colors[prev_base + 3] + (colors.colors[next_base + 3] - colors.colors[prev_base + 3]) * segment_t,
        1.0,
    )
}

/// Apply gradient colors to tessellated mesh vertices based on their position.
/// ThorVG-style: each vertex gets colored based on its gradient parameter.
fn apply_gradient_to_mesh(
    mut mesh: TessellatedMesh,
    gradient_type: GradientType,
    start: Vec2D,
    end: Vec2D,
    colors: &GradientColors,
    opacity: f32,
    world_transform: &Mat2D,
) -> TessellatedMesh {
    if mesh.is_empty() {
        return mesh;
    }

    // Transform gradient control points by the same world transform
    let ws = world_transform.transform_point(start);
    let we = world_transform.transform_point(end);

    for vertex in &mut mesh.vertices {
        let pos = Vec2D::new(vertex.position[0], vertex.position[1]);

        let t = match gradient_type {
            GradientType::Linear => {
                let dir = Vec2D::new(we.x - ws.x, we.y - ws.y);
                let len_sq = dir.dot(dir);
                if len_sq < 1e-6 {
                    0.0
                } else {
                    let d = Vec2D::new(pos.x - ws.x, pos.y - ws.y);
                    d.dot(dir) / len_sq
                }
            }
            GradientType::Radial => {
                let radius = ws.distance(we);
                if radius < 1e-6 {
                    0.0
                } else {
                    pos.distance(ws) / radius
                }
            }
        };

        let color = sample_gradient(colors, t);
        vertex.color = [color.r, color.g, color.b, opacity];
    }

    mesh
}
