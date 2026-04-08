use crate::geometry::math::Mat2D;
use crate::geometry::path::RawPath;
use crate::lottie::model::*;
use crate::lottie::property::*;
use crate::lottie::modifiers;
use super::transform::{ComputedTransform, evaluate_transform};

/// Runtime layer in the scene graph
#[derive(Clone, Debug)]
pub struct SceneLayer {
    pub name: String,
    pub layer_type: LayerType,
    pub index: Option<i32>,
    pub parent_index: Option<i32>,
    pub in_point: f32,
    pub out_point: f32,
    pub start_time: f32,
    pub stretch: f32,
    pub blend_mode: BlendMode,
    pub matte_type: MatteType,
    pub hidden: bool,
    pub transform_data: LottieTransform,
    pub shapes: Vec<ShapeItem>,
    pub masks: Vec<LottieMask>,
    pub effects: Vec<LottieEffect>,
    pub ref_id: Option<String>,
    pub computed: ComputedTransform,
    // Children for precomp layers
    pub children: Vec<SceneLayer>,
}

impl SceneLayer {
    pub fn from_lottie(layer: &LottieLayer) -> Self {
        Self {
            name: layer.name.clone(),
            layer_type: layer.layer_type,
            index: layer.index,
            parent_index: layer.parent_index,
            in_point: layer.in_point,
            out_point: layer.out_point,
            start_time: layer.start_time,
            stretch: layer.stretch,
            blend_mode: layer.blend_mode,
            matte_type: layer.matte_type,
            hidden: layer.hidden,
            transform_data: layer.transform.clone(),
            shapes: layer.shapes.clone(),
            masks: layer.masks.clone(),
            effects: layer.effects.clone(),
            ref_id: layer.ref_id.clone(),
            computed: ComputedTransform::default(),
            children: Vec::new(),
        }
    }

    /// Check if layer is visible at a given frame
    pub fn is_visible_at(&self, frame: f32) -> bool {
        !self.hidden && frame >= self.in_point && frame < self.out_point
    }

    /// Get the local frame considering start_time and stretch
    pub fn local_frame(&self, frame: f32) -> f32 {
        (frame - self.start_time) / self.stretch
    }

    /// Update the computed transform for this layer at a given frame
    pub fn update_transform(&mut self, frame: f32, parent_world: &Mat2D, parent_opacity: f32) {
        let local_frame = self.local_frame(frame);
        let (local_mat, opacity) = evaluate_transform(&self.transform_data, local_frame);

        self.computed.local = local_mat;
        self.computed.world = Mat2D::multiply(parent_world, &local_mat);
        self.computed.opacity = parent_opacity * opacity;
    }

    /// Generate draw paths from shapes at a given frame
    pub fn generate_paths(&self, frame: f32) -> Vec<ShapeDrawCommand> {
        let local_frame = self.local_frame(frame);
        let mut commands = Vec::new();
        collect_shape_draws(&self.shapes, local_frame, &mut commands);
        commands
    }
}

/// A resolved draw command from shapes
#[derive(Clone, Debug)]
pub struct ShapeDrawCommand {
    pub path: RawPath,
    pub paint: ShapePaint,
    pub blend_mode: crate::lottie::model::BlendMode,
}

/// Paint style for a shape draw
#[derive(Clone, Debug)]
pub enum ShapePaint {
    SolidFill {
        color: crate::geometry::math::Color,
        opacity: f32,
        fill_rule: crate::geometry::path::FillRule,
    },
    SolidStroke {
        color: crate::geometry::math::Color,
        opacity: f32,
        width: f32,
        cap: crate::geometry::path::StrokeCap,
        join: crate::geometry::path::StrokeJoin,
        miter_limit: f32,
    },
    GradientFill {
        gradient_type: GradientType,
        start: crate::geometry::math::Vec2D,
        end: crate::geometry::math::Vec2D,
        colors: GradientColors,
        opacity: f32,
        fill_rule: crate::geometry::path::FillRule,
    },
    GradientStroke {
        gradient_type: GradientType,
        start: crate::geometry::math::Vec2D,
        end: crate::geometry::math::Vec2D,
        colors: GradientColors,
        opacity: f32,
        width: f32,
        cap: crate::geometry::path::StrokeCap,
        join: crate::geometry::path::StrokeJoin,
        miter_limit: f32,
    },
}

impl ShapePaint {
    /// Multiply layer opacity into paint opacity
    pub fn apply_opacity(&mut self, layer_opacity: f32) {
        match self {
            ShapePaint::SolidFill { opacity, .. } => *opacity *= layer_opacity,
            ShapePaint::SolidStroke { opacity, .. } => *opacity *= layer_opacity,
            ShapePaint::GradientFill { opacity, .. } => *opacity *= layer_opacity,
            ShapePaint::GradientStroke { opacity, .. } => *opacity *= layer_opacity,
        }
    }
}

/// Collect draw commands from a shape list.
/// Lottie shapes work bottom-up: geometry shapes appear first, then paints.
/// ThorVG-style: geometry → modifiers → paints.
/// Extract the group-level transform from a "tr" shape item (ThorVG-style).
fn extract_group_transform(shapes: &[ShapeItem], frame: f32) -> Option<(Mat2D, f32)> {
    for shape in shapes {
        if let ShapeItem::Transform(tr) = shape {
            let (mat, opacity) = evaluate_transform(tr, frame);
            let is_identity = (mat.values[0] - 1.0).abs() < 1e-5
                && mat.values[1].abs() < 1e-5
                && mat.values[2].abs() < 1e-5
                && (mat.values[3] - 1.0).abs() < 1e-5
                && mat.values[4].abs() < 1e-5
                && mat.values[5].abs() < 1e-5
                && (opacity - 1.0).abs() < 1e-5;
            if !is_identity {
                return Some((mat, opacity));
            }
            return None;
        }
    }
    None
}

fn collect_shape_draws(shapes: &[ShapeItem], frame: f32, commands: &mut Vec<ShapeDrawCommand>) {
    // Collect modifiers from this group level
    let mods = modifiers::collect_modifiers(shapes);

    // First pass: collect all geometry paths
    let mut paths: Vec<RawPath> = Vec::new();

    for shape in shapes {
        match shape {
            ShapeItem::Group(group) => {
                if !group.hidden {
                    // ThorVG-style: apply group transform to children
                    let group_transform = extract_group_transform(&group.items, frame);
                    let mut group_commands = Vec::new();
                    collect_shape_draws(&group.items, frame, &mut group_commands);
                    if let Some((mat, opacity)) = group_transform {
                        for mut cmd in group_commands {
                            cmd.path = cmd.path.transform(&mat);
                            cmd.paint.apply_opacity(opacity);
                            commands.push(cmd);
                        }
                    } else {
                        commands.extend(group_commands);
                    }
                }
            }
            ShapeItem::Rect(rect) if !rect.hidden => {
                let mut path = RawPath::new();
                let pos = eval_vec2d(&rect.position, frame);
                let size = eval_vec2d(&rect.size, frame);
                let roundness = eval_f32(&rect.roundness, frame);
                let x = pos.x - size.x * 0.5;
                let y = pos.y - size.y * 0.5;
                if roundness > 0.0 {
                    path.add_rounded_rect(x, y, size.x, size.y, roundness);
                } else {
                    path.add_rect(x, y, size.x, size.y);
                }
                paths.push(path);
            }
            ShapeItem::Ellipse(ellipse) if !ellipse.hidden => {
                let mut path = RawPath::new();
                let pos = eval_vec2d(&ellipse.position, frame);
                let size = eval_vec2d(&ellipse.size, frame);
                path.add_ellipse(pos.x, pos.y, size.x * 0.5, size.y * 0.5);
                paths.push(path);
            }
            ShapeItem::Path(shape_path) if !shape_path.hidden => {
                let bezier = eval_bezier_path(&shape_path.shape, frame);
                let path = bezier_to_raw_path(&bezier);
                paths.push(path);
            }
            ShapeItem::Polystar(star) if !star.hidden => {
                let mut path = RawPath::new();
                let pos = eval_vec2d(&star.position, frame);
                let points = eval_f32(&star.points, frame);
                let rotation = eval_f32(&star.rotation, frame) * std::f32::consts::PI / 180.0;
                let outer_r = eval_f32(&star.outer_radius, frame);
                let outer_round = eval_f32(&star.outer_roundness, frame) / 100.0;
                let inner_r = eval_f32(&star.inner_radius, frame);
                let inner_round = eval_f32(&star.inner_roundness, frame) / 100.0;
                let is_star = star.star_type == PolystarType::Star;
                path.add_polystar(
                    pos.x, pos.y, points as u32, outer_r, inner_r,
                    outer_round, inner_round, rotation, is_star,
                );
                paths.push(path);
            }
            ShapeItem::Fill(fill) if !fill.hidden => {
                // Apply ThorVG-style modifiers before painting
                let modified_paths = if !mods.is_empty() {
                    modifiers::apply_modifiers(&paths, &mods, frame)
                } else {
                    paths.clone()
                };
                let color = eval_color(&fill.color, frame);
                let opacity = eval_f32(&fill.opacity, frame) / 100.0;
                for p in &modified_paths {
                    commands.push(ShapeDrawCommand {
                        path: p.clone(),
                        paint: ShapePaint::SolidFill {
                            color,
                            opacity,
                            fill_rule: fill.fill_rule,
                        },
                        blend_mode: BlendMode::Normal,
                    });
                }
            }
            ShapeItem::Stroke(stroke) if !stroke.hidden => {
                let modified_paths = if !mods.is_empty() {
                    modifiers::apply_modifiers(&paths, &mods, frame)
                } else {
                    paths.clone()
                };
                let color = eval_color(&stroke.color, frame);
                let opacity = eval_f32(&stroke.opacity, frame) / 100.0;
                let width = eval_f32(&stroke.width, frame);
                for p in &modified_paths {
                    commands.push(ShapeDrawCommand {
                        path: p.clone(),
                        paint: ShapePaint::SolidStroke {
                            color,
                            opacity,
                            width,
                            cap: stroke.line_cap,
                            join: stroke.line_join,
                            miter_limit: stroke.miter_limit,
                        },
                        blend_mode: BlendMode::Normal,
                    });
                }
            }
            ShapeItem::GradientFill(gf) if !gf.hidden => {
                let modified_paths = if !mods.is_empty() {
                    modifiers::apply_modifiers(&paths, &mods, frame)
                } else {
                    paths.clone()
                };
                let start = eval_vec2d(&gf.start_point, frame);
                let end = eval_vec2d(&gf.end_point, frame);
                let colors = evaluate(&gf.colors, frame);
                let opacity = eval_f32(&gf.opacity, frame) / 100.0;
                for p in &modified_paths {
                    commands.push(ShapeDrawCommand {
                        path: p.clone(),
                        paint: ShapePaint::GradientFill {
                            gradient_type: gf.gradient_type,
                            start,
                            end,
                            colors: colors.clone(),
                            opacity,
                            fill_rule: gf.fill_rule,
                        },
                        blend_mode: BlendMode::Normal,
                    });
                }
            }
            ShapeItem::GradientStroke(gs) if !gs.hidden => {
                let modified_paths = if !mods.is_empty() {
                    modifiers::apply_modifiers(&paths, &mods, frame)
                } else {
                    paths.clone()
                };
                let start = eval_vec2d(&gs.start_point, frame);
                let end = eval_vec2d(&gs.end_point, frame);
                let colors = evaluate(&gs.colors, frame);
                let opacity = eval_f32(&gs.opacity, frame) / 100.0;
                let width = eval_f32(&gs.width, frame);
                for p in &modified_paths {
                    commands.push(ShapeDrawCommand {
                        path: p.clone(),
                        paint: ShapePaint::GradientStroke {
                            gradient_type: gs.gradient_type,
                            start,
                            end,
                            colors: colors.clone(),
                            opacity,
                            width,
                            cap: gs.line_cap,
                            join: gs.line_join,
                            miter_limit: gs.miter_limit,
                        },
                        blend_mode: BlendMode::Normal,
                    });
                }
            }
            _ => {}
        }
    }
}

/// Convert a Lottie BezierPath to a RawPath
pub fn bezier_to_raw_path(bezier: &BezierPath) -> RawPath {
    let mut path = RawPath::new();
    let n = bezier.vertices.len();
    if n == 0 {
        return path;
    }

    let v0 = bezier.vertices[0];
    path.move_to(v0.x, v0.y);

    for i in 1..n {
        let prev = i - 1;
        let ot = if prev < bezier.out_tangents.len() {
            bezier.out_tangents[prev]
        } else {
            crate::geometry::math::Vec2D::ZERO
        };
        let it = if i < bezier.in_tangents.len() {
            bezier.in_tangents[i]
        } else {
            crate::geometry::math::Vec2D::ZERO
        };

        let cp1x = bezier.vertices[prev].x + ot.x;
        let cp1y = bezier.vertices[prev].y + ot.y;
        let cp2x = bezier.vertices[i].x + it.x;
        let cp2y = bezier.vertices[i].y + it.y;

        // If tangents are zero, use line_to
        if ot.x.abs() < 1e-4 && ot.y.abs() < 1e-4 && it.x.abs() < 1e-4 && it.y.abs() < 1e-4 {
            path.line_to(bezier.vertices[i].x, bezier.vertices[i].y);
        } else {
            path.cubic_to(cp1x, cp1y, cp2x, cp2y, bezier.vertices[i].x, bezier.vertices[i].y);
        }
    }

    // Close path
    if bezier.closed && n > 1 {
        let last = n - 1;
        let ot = if last < bezier.out_tangents.len() {
            bezier.out_tangents[last]
        } else {
            crate::geometry::math::Vec2D::ZERO
        };
        let it = if !bezier.in_tangents.is_empty() {
            bezier.in_tangents[0]
        } else {
            crate::geometry::math::Vec2D::ZERO
        };

        let cp1x = bezier.vertices[last].x + ot.x;
        let cp1y = bezier.vertices[last].y + ot.y;
        let cp2x = bezier.vertices[0].x + it.x;
        let cp2y = bezier.vertices[0].y + it.y;

        if ot.x.abs() < 1e-4 && ot.y.abs() < 1e-4 && it.x.abs() < 1e-4 && it.y.abs() < 1e-4 {
            // No tangents, just close
        } else {
            path.cubic_to(cp1x, cp1y, cp2x, cp2y, bezier.vertices[0].x, bezier.vertices[0].y);
        }
        path.close();
    }

    path
}
