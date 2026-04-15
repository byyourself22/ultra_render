use super::animation::AnimationPlayer;
use super::layer::{bezier_to_raw_path, ShapeDrawCommand, ShapePaint};
use super::transform::{evaluate_transform, ComputedTransform};
/// Rive-style Artboard — main container for animation playback and rendering.
///
/// Mirrors Rive's Artboard architecture:
/// - Dirty flag system (ComponentDirt) to skip unnecessary recalculations
/// - DAG-ordered component updates (topological sort by parent hierarchy)
/// - advance() / draw() cycle
/// - Drawable list for efficient rendering
use crate::geometry::math::{Mat2D, AABB};
use crate::geometry::path::RawPath;
use crate::lottie::model::*;
use crate::lottie::modifiers;
use crate::lottie::property::*;

// ─── Dirty Flags (Rive ComponentDirt) ───────────────────────

bitflags::bitflags! {
    /// Rive-style dirty flags for tracking what needs recalculation.
    /// Each flag corresponds to a specific aspect of the component.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ComponentDirt: u32 {
        /// Dependents need update
        const DEPENDENTS    = 1 << 0;
        /// Transform changed
        const TRANSFORM     = 1 << 1;
        /// World transform needs recalculation
        const WORLD_TRANSFORM = 1 << 2;
        /// Path geometry changed
        const PATH          = 1 << 3;
        /// Paint properties changed
        const PAINT         = 1 << 4;
        /// Blend mode changed
        const BLEND_MODE    = 1 << 5;
        /// Render opacity changed
        const RENDER_OPACITY = 1 << 6;
        /// Clip changed
        const CLIP          = 1 << 7;
        /// Draw order changed
        const DRAW_ORDER    = 1 << 8;
        /// Stops changed (gradient)
        const STOPS         = 1 << 9;
        /// Layout needs recalculation
        const LAYOUT        = 1 << 10;
    }
}

// ─── Artboard Node ──────────────────────────────────────────

/// A node in the artboard's component tree.
/// Rive uses a flat array of components with parent indices for the DAG.
#[derive(Clone, Debug)]
pub struct ArtboardNode {
    pub name: String,
    pub node_type: NodeType,
    pub parent_idx: Option<usize>, // Transform parent in the node array
    pub draw_parent_idx: Option<usize>, // Draw-order owner (root or precomp)
    pub lottie_parent: Option<i32>, // Lottie parent layer index
    pub lottie_index: Option<i32>, // Lottie layer index
    pub dirt: ComponentDirt,
    pub depth: u32, // DAG depth for topological ordering
    pub transform: ComputedTransform,
    pub layer_data: Option<LayerData>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeType {
    Root,
    Layer,
    Shape,
}

/// Per-layer runtime data
#[derive(Clone, Debug)]
pub struct LayerData {
    pub layer_type: LayerType,
    pub in_point: f32,
    pub out_point: f32,
    pub start_time: f32,
    pub stretch: f32,
    pub blend_mode: BlendMode,
    pub hidden: bool,
    pub transform_data: LottieTransform,
    pub shapes: Vec<ShapeItem>,
    // Solid layer fields (LayerType::Solid)
    pub solid_color: Option<String>,
    pub solid_width: Option<f32>,
    pub solid_height: Option<f32>,
}

// ─── Artboard ───────────────────────────────────────────────

/// Rive-style Artboard — top-level container.
///
/// The artboard owns:
/// - A flat node array (components) with parent refs forming a DAG
/// - A DAG-ordered update list (topological sort)
/// - Dirty flags per node for incremental updates
/// - An animation player
/// - Cached draw commands (only regenerated when dirty)
#[derive(Clone, Debug)]
pub struct Artboard {
    pub name: String,
    pub width: f32,
    pub height: f32,
    pub nodes: Vec<ArtboardNode>,
    pub update_order: Vec<usize>, // DAG topological order
    pub player: AnimationPlayer,
    pub draw_cache: Vec<ShapeDrawCommand>,
    pub draw_dirty: bool,
    pub draw_version: u64,
    pub last_frame: f32,
    // Cached children list — rebuilt only when node count changes
    children_cache: Vec<Vec<usize>>,
    children_dirty: bool,
}

impl Artboard {
    /// Build an artboard from a Lottie composition
    pub fn from_lottie(comp: &LottieComposition) -> Self {
        let mut nodes = Vec::new();

        // Root node
        nodes.push(ArtboardNode {
            name: comp.name.clone(),
            node_type: NodeType::Root,
            parent_idx: None,
            draw_parent_idx: None,
            lottie_parent: None,
            lottie_index: None,
            dirt: ComponentDirt::all(),
            depth: 0,
            transform: ComputedTransform::default(),
            layer_data: None,
        });

        // Add layers as nodes
        for layer in &comp.layers {
            let node = ArtboardNode {
                name: layer.name.clone(),
                node_type: if layer.layer_type == LayerType::Shape {
                    NodeType::Shape
                } else {
                    NodeType::Layer
                },
                parent_idx: None, // resolved after all nodes are added
                draw_parent_idx: Some(0),
                lottie_parent: layer.parent_index,
                lottie_index: layer.index,
                dirt: ComponentDirt::all(),
                depth: 1,
                transform: ComputedTransform::default(),
                layer_data: Some(LayerData {
                    layer_type: layer.layer_type,
                    in_point: layer.in_point,
                    out_point: layer.out_point,
                    start_time: layer.start_time,
                    stretch: layer.stretch,
                    blend_mode: layer.blend_mode,
                    hidden: layer.hidden,
                    transform_data: layer.transform.clone(),
                    shapes: layer.shapes.clone(),
                    solid_color: layer.solid_color.clone(),
                    solid_width: layer.solid_width,
                    solid_height: layer.solid_height,
                }),
            };
            nodes.push(node);
        }

        // Add precomp children
        let mut precomp_children = Vec::new();
        for (i, layer) in comp.layers.iter().enumerate() {
            if let Some(ref_id) = &layer.ref_id {
                if let Some(asset) = comp.assets.iter().find(|a| a.id == *ref_id) {
                    if !asset.is_image {
                        let parent_node_idx = i + 1; // +1 because root is at 0
                        for child_layer in &asset.layers {
                            precomp_children.push((parent_node_idx, child_layer.clone()));
                        }
                    }
                }
            }
        }

        for (parent_idx, child_layer) in precomp_children {
            nodes.push(ArtboardNode {
                name: child_layer.name.clone(),
                node_type: if child_layer.layer_type == LayerType::Shape {
                    NodeType::Shape
                } else {
                    NodeType::Layer
                },
                parent_idx: Some(parent_idx),
                draw_parent_idx: Some(parent_idx),
                lottie_parent: child_layer.parent_index,
                lottie_index: child_layer.index,
                dirt: ComponentDirt::all(),
                depth: 2,
                transform: ComputedTransform::default(),
                layer_data: Some(LayerData {
                    layer_type: child_layer.layer_type,
                    in_point: child_layer.in_point,
                    out_point: child_layer.out_point,
                    start_time: child_layer.start_time,
                    stretch: child_layer.stretch,
                    blend_mode: child_layer.blend_mode,
                    hidden: child_layer.hidden,
                    transform_data: child_layer.transform.clone(),
                    shapes: child_layer.shapes.clone(),
                    solid_color: child_layer.solid_color.clone(),
                    solid_width: child_layer.solid_width,
                    solid_height: child_layer.solid_height,
                }),
            });
        }

        // Resolve parent indices from Lottie parent references
        resolve_parent_indices(&mut nodes);

        // Compute DAG depths
        compute_depths(&mut nodes);

        // Build topological update order (sorted by depth)
        let mut update_order: Vec<usize> = (0..nodes.len()).collect();
        update_order.sort_by_key(|&i| nodes[i].depth);

        let player = AnimationPlayer::new(comp.frame_rate, comp.in_point, comp.out_point);

        Artboard {
            name: comp.name.clone(),
            width: comp.width,
            height: comp.height,
            nodes,
            update_order,
            player,
            draw_cache: Vec::new(),
            draw_dirty: true,
            draw_version: 0,
            last_frame: -1.0,
            children_cache: Vec::new(),
            children_dirty: true,
        }
    }

    /// Advance animation by delta_seconds (Rive Artboard::advance)
    pub fn advance(&mut self, delta_seconds: f32) -> bool {
        let was_playing = self.player.advance(delta_seconds);
        let frame = self.player.frame;

        // Only mark dirty if frame actually changed (use tiny epsilon for sub-frame precision)
        if (frame - self.last_frame).abs() > f32::EPSILON {
            self.last_frame = frame;
            // Mark all nodes with transform dirt
            for node in &mut self.nodes {
                node.dirt
                    .insert(ComponentDirt::TRANSFORM | ComponentDirt::WORLD_TRANSFORM);
            }
            self.draw_dirty = true;
        }

        was_playing
    }

    /// Update all components in DAG order (Rive Artboard::update)
    pub fn update(&mut self) {
        let frame = self.player.frame;
        for &idx in &self.update_order {
            if !self.nodes[idx]
                .dirt
                .intersects(ComponentDirt::TRANSFORM | ComponentDirt::WORLD_TRANSFORM)
            {
                continue; // Clean node, skip
            }

            let parent_world;
            let parent_opacity;

            if let Some(parent_idx) = self.nodes[idx].parent_idx {
                parent_world = self.nodes[parent_idx].transform.world;
                parent_opacity = self.nodes[parent_idx].transform.opacity;
            } else {
                parent_world = Mat2D::identity();
                parent_opacity = 1.0;
            }

            let Some((start_time, stretch, transform_data)) =
                self.nodes[idx].layer_data.as_ref().map(|layer_data| {
                    (
                        layer_data.start_time,
                        layer_data.stretch,
                        layer_data.transform_data.clone(),
                    )
                })
            else {
                self.nodes[idx]
                    .dirt
                    .remove(ComponentDirt::TRANSFORM | ComponentDirt::WORLD_TRANSFORM);
                continue;
            };

            // Compute effective frame considering parent precomp timing.
            // ONLY precomp parents create a new timing context (Lottie spec / ThorVG).
            // Regular transform parents (via `parent` field) share the same timeline —
            // they do NOT remap time. This fixes gfunny.json where chin (parent=upper_face)
            // was incorrectly getting time-remapped through a non-precomp parent.
            let eff_frame = {
                let mut eff = frame;
                let mut current = idx;
                while let Some(pi) = self.nodes[current].parent_idx {
                    if pi == 0 {
                        break;
                    }
                    if let Some(pd) = &self.nodes[pi].layer_data {
                        if pd.layer_type == LayerType::Precomp {
                            eff = (eff - pd.start_time) / pd.stretch;
                        }
                    }
                    current = pi;
                }
                eff
            };

            let local_frame = (eff_frame - start_time) / stretch;
            let (local_mat, opacity) = evaluate_transform(&transform_data, local_frame);

            self.nodes[idx].transform.local = local_mat;
            self.nodes[idx].transform.world = Mat2D::multiply(&parent_world, &local_mat);
            self.nodes[idx].transform.opacity = parent_opacity * opacity;

            // Clear dirt
            self.nodes[idx]
                .dirt
                .remove(ComponentDirt::TRANSFORM | ComponentDirt::WORLD_TRANSFORM);
        }
    }

    /// Collect draw commands (Rive Artboard::draw)
    /// Uses dirty flag to avoid redundant regeneration.
    /// ThorVG-style: respects Lottie layer tree — precomp children render within
    /// their parent's context, not interleaved with main layers.
    pub fn draw(&mut self) -> &[ShapeDrawCommand] {
        if !self.draw_dirty {
            return &self.draw_cache;
        }

        self.draw_cache.clear();
        let frame = self.player.frame;

        // Compute effective frame for each node considering parent precomp timing.
        // Walk up the parent chain to accumulate time remapping for nested precomps.
        let effective_frames: Vec<f32> = (0..self.nodes.len())
            .map(|idx| self.compute_effective_frame(idx, frame))
            .collect();

        // Build/reuse children list (only rebuilds when structure changes)
        if self.children_dirty || self.children_cache.len() != self.nodes.len() {
            self.children_cache = vec![Vec::new(); self.nodes.len()];
            for i in 1..self.nodes.len() {
                if let Some(draw_parent_idx) = self.nodes[i].draw_parent_idx {
                    self.children_cache[draw_parent_idx].push(i);
                }
            }
            self.children_dirty = false;
        }

        let root_children = &self.children_cache[0];

        // Debug: log node info on first draw
        static DRAW_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !DRAW_LOGGED.load(std::sync::atomic::Ordering::Relaxed) {
            DRAW_LOGGED.store(true, std::sync::atomic::Ordering::Relaxed);
            log::info!(
                "Artboard draw: frame={:.1}, {} nodes, root_children={:?}",
                frame,
                self.nodes.len(),
                root_children
            );
            for (i, node) in self.nodes.iter().enumerate() {
                if let Some(ref ld) = node.layer_data {
                    let eff = effective_frames[i];
                    let vis = !ld.hidden && eff >= ld.in_point && eff < ld.out_point;
                    log::info!("  node[{}] '{}' {:?} parent={:?} draw_parent={:?} eff_frame={:.1} in={:.0}..{:.0} hidden={} shapes={} vis={}",
                        i, node.name, ld.layer_type, node.parent_idx, node.draw_parent_idx, eff, ld.in_point, ld.out_point, ld.hidden, ld.shapes.len(), vis);
                }
            }
        }

        // Collect draws tree-recursively, back-to-front (ThorVG layer ordering)
        let children = &self.children_cache;
        let mut cache = Vec::new();
        self.collect_draws_recursive(root_children, children, &effective_frames, &mut cache);
        self.draw_cache = cache;

        self.draw_version = self.draw_version.wrapping_add(1);
        self.draw_dirty = false;
        &self.draw_cache
    }

    pub fn draw_version(&self) -> u64 {
        self.draw_version
    }

    /// Compute effective frame for a node, walking up the parent chain
    /// for nested precomp time remapping.
    fn compute_effective_frame(&self, idx: usize, global_frame: f32) -> f32 {
        let mut eff = global_frame;
        let mut current = idx;
        // Walk up to find precomp parents and apply their time remapping
        while let Some(parent_idx) = self.nodes[current].parent_idx {
            if parent_idx == 0 {
                break;
            }
            if let Some(parent_data) = &self.nodes[parent_idx].layer_data {
                if parent_data.layer_type == LayerType::Precomp {
                    eff = (eff - parent_data.start_time) / parent_data.stretch;
                }
            }
            current = parent_idx;
        }
        eff
    }

    /// Recursively collect draws in correct Lottie layer order (back-to-front = reversed).
    /// Only precomp ownership affects draw tree; normal parenting only affects transforms.
    fn collect_draws_recursive(
        &self,
        layer_indices: &[usize],
        children: &[Vec<usize>],
        effective_frames: &[f32],
        commands: &mut Vec<ShapeDrawCommand>,
    ) {
        for &idx in layer_indices.iter().rev() {
            let node = &self.nodes[idx];
            let layer_data = match &node.layer_data {
                Some(d) => d,
                None => continue,
            };

            if layer_data.hidden {
                continue;
            }

            let eff_frame = effective_frames[idx];
            if eff_frame < layer_data.in_point || eff_frame >= layer_data.out_point {
                continue;
            }

            match layer_data.layer_type {
                LayerType::Shape => {
                    let local_frame = (eff_frame - layer_data.start_time) / layer_data.stretch;
                    let world = node.transform.world;
                    let opacity = node.transform.opacity;
                    let layer_blend = layer_data.blend_mode;
                    let draws = generate_shape_draws(&layer_data.shapes, local_frame);

                    for mut draw in draws {
                        draw.path = draw.path.transform(&world);
                        draw.paint.apply_transform(&world);
                        draw.paint.apply_opacity(opacity);
                        if layer_blend != BlendMode::Normal {
                            draw.blend_mode = layer_blend;
                        }
                        commands.push(draw);
                    }
                }
                LayerType::Precomp => {
                    let precomp_children = &children[idx];
                    if !precomp_children.is_empty() {
                        self.collect_draws_recursive(
                            precomp_children,
                            children,
                            effective_frames,
                            commands,
                        );
                    }
                }
                LayerType::Solid => {
                    if let (Some(color_hex), Some(w), Some(h)) = (
                        &layer_data.solid_color,
                        layer_data.solid_width,
                        layer_data.solid_height,
                    ) {
                        let color = parse_hex_color(color_hex);
                        let opacity = node.transform.opacity;
                        let world = node.transform.world;

                        let mut path = crate::geometry::path::RawPath::new();
                        path.add_rect(0.0, 0.0, w, h);
                        let path = path.transform(&world);

                        commands.push(ShapeDrawCommand {
                            path,
                            paint: super::layer::ShapePaint::SolidFill {
                                color,
                                opacity,
                                fill_rule: crate::geometry::path::FillRule::NonZero,
                            },
                            blend_mode: layer_data.blend_mode,
                            sprite_index: 0,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    /// Start playing
    pub fn play(&mut self) {
        self.player.play();
    }

    /// Get current frame
    pub fn frame(&self) -> f32 {
        self.player.frame
    }

    /// Copy playback state from another artboard without rebuilding draw data.
    /// Used for synced sprite instances where only the master artboard is evaluated.
    pub fn sync_playback_from(&mut self, other: &Artboard) {
        self.player = other.player.clone();
        self.last_frame = other.last_frame;
    }

    /// Seek to a frame and force transforms/draws to rebuild on the next update.
    pub fn seek_frame(&mut self, frame: f32) {
        self.player.seek(frame);
        self.last_frame = -1.0;
        self.draw_dirty = true;
        for node in &mut self.nodes {
            node.dirt
                .insert(ComponentDirt::TRANSFORM | ComponentDirt::WORLD_TRANSFORM);
        }
    }

    /// Mark a specific node dirty
    pub fn mark_dirty(&mut self, node_idx: usize, flags: ComponentDirt) {
        if node_idx < self.nodes.len() {
            self.nodes[node_idx].dirt.insert(flags);
            self.draw_dirty = true;
        }
    }

    /// Get artboard bounds
    pub fn bounds(&self) -> AABB {
        AABB::new(0.0, 0.0, self.width, self.height)
    }
}

/// Extract the group-level transform from a "tr" shape item (ThorVG-style).
/// In Lottie, each shape group ("gr") contains a "tr" item that transforms all
/// geometry within the group. ThorVG applies this before painting.
fn extract_group_transform(shapes: &[ShapeItem], frame: f32) -> Option<(Mat2D, f32)> {
    for shape in shapes {
        if let ShapeItem::Transform(tr) = shape {
            let (mat, opacity) = evaluate_transform(tr, frame);
            // Only return if not identity (optimization)
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

// ─── DAG helpers ────────────────────────────────────────────

/// Resolve lottie_parent indices to node array indices.
/// Parent lookup is scoped to the draw owner (root or precomp), not global layer order.
fn resolve_parent_indices(nodes: &mut [ArtboardNode]) {
    let scope_parents: Vec<Option<usize>> = nodes.iter().map(|n| n.draw_parent_idx).collect();

    for i in 1..nodes.len() {
        let fallback_parent = scope_parents[i].unwrap_or(0);

        if let Some(lottie_parent) = nodes[i].lottie_parent {
            let my_scope = scope_parents[i];
            let found = nodes.iter().enumerate().find(|(j, n)| {
                *j != i && n.lottie_index == Some(lottie_parent) && scope_parents[*j] == my_scope
            });
            nodes[i].parent_idx = Some(
                found
                    .map(|(parent_node_idx, _)| parent_node_idx)
                    .unwrap_or(fallback_parent),
            );
        } else {
            nodes[i].parent_idx = Some(fallback_parent);
        }
    }
}

/// Compute depths for topological ordering
fn compute_depths(nodes: &mut [ArtboardNode]) {
    // Iteratively compute depths from parent chain
    let max_iter = nodes.len();
    for _ in 0..max_iter {
        let mut changed = false;
        for i in 0..nodes.len() {
            if let Some(parent_idx) = nodes[i].parent_idx {
                let new_depth = nodes[parent_idx].depth + 1;
                if nodes[i].depth != new_depth {
                    nodes[i].depth = new_depth;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

// ─── Shape draw generation ──────────────────────────────────

/// Generate draw commands from shapes at a given frame.
/// ThorVG-style: geometry → modifiers → paints.
fn generate_shape_draws(shapes: &[ShapeItem], frame: f32) -> Vec<ShapeDrawCommand> {
    let mut commands = Vec::new();
    collect_shape_draws_artboard(shapes, frame, &mut commands);
    commands
}

fn collect_shape_draws_artboard(
    shapes: &[ShapeItem],
    frame: f32,
    commands: &mut Vec<ShapeDrawCommand>,
) {
    let mods = modifiers::collect_modifiers(shapes);
    let mut paths: Vec<RawPath> = Vec::new();

    for shape in shapes {
        match shape {
            ShapeItem::Group(group) if !group.hidden => {
                // ThorVG-style: extract group transform from "tr" item and apply to children
                let group_transform = extract_group_transform(&group.items, frame);
                let mut group_commands = Vec::new();
                collect_shape_draws_artboard(&group.items, frame, &mut group_commands);
                // Apply group transform to all paths in the group's draw commands
                if let Some((mat, opacity)) = group_transform {
                    for mut cmd in group_commands {
                        cmd.path = cmd.path.transform(&mat);
                        cmd.paint.apply_transform(&mat);
                        cmd.paint.apply_opacity(opacity);
                        commands.push(cmd);
                    }
                } else {
                    commands.extend(group_commands);
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
                    pos.x,
                    pos.y,
                    points as u32,
                    outer_r,
                    inner_r,
                    outer_round,
                    inner_round,
                    rotation,
                    is_star,
                );
                paths.push(path);
            }
            ShapeItem::Fill(fill) if !fill.hidden => {
                let modified = if !mods.is_empty() {
                    modifiers::apply_modifiers(&paths, &mods, frame)
                } else {
                    paths.clone()
                };
                let color = eval_color(&fill.color, frame);
                let opacity = eval_f32(&fill.opacity, frame) / 100.0;
                for p in &modified {
                    commands.push(ShapeDrawCommand {
                        path: p.clone(),
                        paint: ShapePaint::SolidFill {
                            color,
                            opacity,
                            fill_rule: fill.fill_rule,
                        },
                        blend_mode: BlendMode::Normal,
                        sprite_index: 0,
                    });
                }
            }
            ShapeItem::Stroke(stroke) if !stroke.hidden => {
                let modified = if !mods.is_empty() {
                    modifiers::apply_modifiers(&paths, &mods, frame)
                } else {
                    paths.clone()
                };
                let color = eval_color(&stroke.color, frame);
                let opacity = eval_f32(&stroke.opacity, frame) / 100.0;
                let width = eval_f32(&stroke.width, frame);
                for p in &modified {
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
                        sprite_index: 0,
                    });
                }
            }
            ShapeItem::GradientFill(gf) if !gf.hidden => {
                let modified = if !mods.is_empty() {
                    modifiers::apply_modifiers(&paths, &mods, frame)
                } else {
                    paths.clone()
                };
                let start = eval_vec2d(&gf.start_point, frame);
                let end = eval_vec2d(&gf.end_point, frame);
                let colors = evaluate(&gf.colors, frame);
                let opacity = eval_f32(&gf.opacity, frame) / 100.0;
                for p in &modified {
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
                        sprite_index: 0,
                    });
                }
            }
            ShapeItem::GradientStroke(gs) if !gs.hidden => {
                let modified = if !mods.is_empty() {
                    modifiers::apply_modifiers(&paths, &mods, frame)
                } else {
                    paths.clone()
                };
                let start = eval_vec2d(&gs.start_point, frame);
                let end = eval_vec2d(&gs.end_point, frame);
                let colors = evaluate(&gs.colors, frame);
                let opacity = eval_f32(&gs.opacity, frame) / 100.0;
                let width = eval_f32(&gs.width, frame);
                for p in &modified {
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
                        sprite_index: 0,
                    });
                }
            }
            _ => {}
        }
    }
}

/// Parse a Lottie solid layer hex color string (e.g. "#ff8800") to Color.
fn parse_hex_color(hex: &str) -> crate::geometry::math::Color {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex.get(0..2).unwrap_or("00"), 16).unwrap_or(0);
    let g = u8::from_str_radix(&hex.get(2..4).unwrap_or("00"), 16).unwrap_or(0);
    let b = u8::from_str_radix(&hex.get(4..6).unwrap_or("00"), 16).unwrap_or(0);
    crate::geometry::math::Color::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::math::Vec2D;

    fn make_test_composition() -> LottieComposition {
        LottieComposition {
            version: "5.0.0".into(),
            width: 100.0,
            height: 100.0,
            frame_rate: 30.0,
            in_point: 0.0,
            out_point: 30.0,
            name: "test".into(),
            is_3d: false,
            assets: vec![],
            layers: vec![],
            markers: vec![],
        }
    }

    fn make_shape_layer(index: i32, shapes: Vec<ShapeItem>) -> LottieLayer {
        LottieLayer {
            layer_type: LayerType::Shape,
            index: Some(index),
            in_point: 0.0,
            out_point: 30.0,
            shapes,
            ..Default::default()
        }
    }

    fn make_rect(x: f32, y: f32, w: f32, h: f32) -> ShapeItem {
        ShapeItem::Rect(ShapeRect {
            name: "rect".into(),
            position: AnimatedValue::Static(Vec2D::new(x + w / 2.0, y + h / 2.0)),
            size: AnimatedValue::Static(Vec2D::new(w, h)),
            roundness: AnimatedValue::Static(0.0),
            hidden: false,
        })
    }

    fn make_fill(r: f32, g: f32, b: f32) -> ShapeItem {
        ShapeItem::Fill(ShapeFill {
            name: "fill".into(),
            color: AnimatedValue::Static(crate::geometry::math::Color::new(r, g, b, 1.0)),
            opacity: AnimatedValue::Static(100.0),
            fill_rule: crate::geometry::path::FillRule::NonZero,
            hidden: false,
        })
    }

    fn make_group_transform(tx: f32, ty: f32) -> ShapeItem {
        ShapeItem::Transform(LottieTransform {
            position: AnimatedValue::Static(Vec2D::new(tx, ty)),
            ..Default::default()
        })
    }

    #[test]
    fn test_group_transform_applied() {
        // A shape group with a transform should offset its children
        let group = ShapeItem::Group(ShapeGroup {
            name: "group".into(),
            items: vec![
                make_rect(0.0, 0.0, 10.0, 10.0),
                make_fill(1.0, 0.0, 0.0),
                make_group_transform(50.0, 50.0),
            ],
            blend_mode: BlendMode::Normal,
            hidden: false,
        });

        let layer = make_shape_layer(1, vec![group]);
        let mut comp = make_test_composition();
        comp.layers = vec![layer];

        let mut artboard = Artboard::from_lottie(&comp);
        artboard.play();
        artboard.advance(0.0);
        artboard.update();
        let draws = artboard.draw();

        assert!(!draws.is_empty(), "Should produce draw commands");
        // The rect is at (0,0)-(10,10) but group transform shifts by (50,50)
        // So all path points should be offset by (50,50)
        let bounds = draws[0].path.bounds();
        assert!(
            bounds.min_x >= 45.0,
            "X should be shifted by group transform, got {}",
            bounds.min_x
        );
        assert!(
            bounds.min_y >= 45.0,
            "Y should be shifted by group transform, got {}",
            bounds.min_y
        );
    }

    #[test]
    fn test_layer_ordering_back_to_front() {
        // Layer 0 (red) should render before Layer 1 (blue) — back-to-front
        let red_layer = make_shape_layer(
            0,
            vec![make_rect(0.0, 0.0, 10.0, 10.0), make_fill(1.0, 0.0, 0.0)],
        );
        let blue_layer = make_shape_layer(
            1,
            vec![make_rect(0.0, 0.0, 10.0, 10.0), make_fill(0.0, 0.0, 1.0)],
        );

        let mut comp = make_test_composition();
        // Lottie layers are front-to-back: first in array renders on top
        comp.layers = vec![blue_layer, red_layer];

        let mut artboard = Artboard::from_lottie(&comp);
        artboard.play();
        artboard.advance(0.0);
        artboard.update();
        let draws = artboard.draw();

        assert_eq!(draws.len(), 2, "Should have 2 draws");
        // Back-to-front: red (index 0) should be first in draw order
        match &draws[0].paint {
            ShapePaint::SolidFill { color, .. } => {
                assert!(color.r > 0.9, "First draw should be red (back layer)");
            }
            _ => panic!("Expected SolidFill"),
        }
        match &draws[1].paint {
            ShapePaint::SolidFill { color, .. } => {
                assert!(color.b > 0.9, "Second draw should be blue (front layer)");
            }
            _ => panic!("Expected SolidFill"),
        }
    }

    #[test]
    fn test_layer_visibility_timing() {
        // Layer only visible in frames 10..20
        let mut layer = make_shape_layer(
            0,
            vec![make_rect(0.0, 0.0, 10.0, 10.0), make_fill(1.0, 0.0, 0.0)],
        );
        layer.in_point = 10.0;
        layer.out_point = 20.0;

        let mut comp = make_test_composition();
        comp.layers = vec![layer];

        let mut artboard = Artboard::from_lottie(&comp);
        artboard.play();

        // Frame 0: not visible
        artboard.player.seek(0.0);
        artboard.draw_dirty = true;
        for node in &mut artboard.nodes {
            node.dirt = ComponentDirt::all();
        }
        artboard.update();
        let draws = artboard.draw();
        assert!(draws.is_empty(), "Layer should not be visible at frame 0");

        // Frame 15: visible
        artboard.player.seek(15.0);
        artboard.draw_dirty = true;
        artboard.last_frame = -1.0;
        for node in &mut artboard.nodes {
            node.dirt = ComponentDirt::all();
        }
        artboard.update();
        let draws = artboard.draw();
        assert!(!draws.is_empty(), "Layer should be visible at frame 15");
    }

    #[test]
    fn test_parent_transform_hierarchy() {
        // Parent layer (null) at position (30, 30), child layer with shape
        let parent_layer = LottieLayer {
            layer_type: LayerType::Null,
            index: Some(0),
            in_point: 0.0,
            out_point: 30.0,
            transform: LottieTransform {
                position: AnimatedValue::Static(Vec2D::new(30.0, 30.0)),
                ..Default::default()
            },
            ..Default::default()
        };

        let mut child_layer = make_shape_layer(
            1,
            vec![make_rect(0.0, 0.0, 10.0, 10.0), make_fill(1.0, 0.0, 0.0)],
        );
        child_layer.parent_index = Some(0); // parent is layer index 0

        let mut comp = make_test_composition();
        comp.layers = vec![parent_layer, child_layer];

        let mut artboard = Artboard::from_lottie(&comp);
        artboard.play();
        artboard.advance(0.0);
        artboard.update();
        let draws = artboard.draw();

        assert!(!draws.is_empty(), "Should produce draws");
        // Child rect is at (0,0)-(10,10), parent shifts by (30,30)
        let bounds = draws[0].path.bounds();
        assert!(
            bounds.min_x >= 25.0,
            "Child should inherit parent position, got min_x={}",
            bounds.min_x
        );
        assert!(
            bounds.min_y >= 25.0,
            "Child should inherit parent position, got min_y={}",
            bounds.min_y
        );
    }

    #[test]
    fn test_opacity_inheritance() {
        // Parent at 50% opacity, child fill at 80% opacity -> effective ~40%
        let parent_layer = LottieLayer {
            layer_type: LayerType::Null,
            index: Some(0),
            in_point: 0.0,
            out_point: 30.0,
            transform: LottieTransform {
                opacity: AnimatedValue::Static(50.0), // 50%
                ..Default::default()
            },
            ..Default::default()
        };

        let mut child_layer = make_shape_layer(
            1,
            vec![
                make_rect(0.0, 0.0, 10.0, 10.0),
                ShapeItem::Fill(ShapeFill {
                    name: "fill".into(),
                    color: AnimatedValue::Static(crate::geometry::math::Color::new(
                        1.0, 0.0, 0.0, 1.0,
                    )),
                    opacity: AnimatedValue::Static(80.0), // 80%
                    fill_rule: crate::geometry::path::FillRule::NonZero,
                    hidden: false,
                }),
            ],
        );
        child_layer.parent_index = Some(0);

        let mut comp = make_test_composition();
        comp.layers = vec![parent_layer, child_layer];

        let mut artboard = Artboard::from_lottie(&comp);
        artboard.play();
        artboard.advance(0.0);
        artboard.update();
        let draws = artboard.draw();

        assert!(!draws.is_empty());
        match &draws[0].paint {
            ShapePaint::SolidFill { opacity, .. } => {
                // 0.8 (fill) * 0.5 (parent) = 0.4
                assert!(
                    (opacity - 0.4).abs() < 0.05,
                    "Effective opacity should be ~0.4, got {}",
                    opacity
                );
            }
            _ => panic!("Expected SolidFill"),
        }
    }

    #[test]
    fn test_precomp_children_render() {
        // Precomp layer referencing an asset with a shape layer
        let asset = LottieAsset {
            id: "precomp_1".into(),
            name: "precomp".into(),
            layers: vec![make_shape_layer(
                0,
                vec![make_rect(0.0, 0.0, 20.0, 20.0), make_fill(0.0, 1.0, 0.0)],
            )],
            width: Some(100.0),
            height: Some(100.0),
            path: None,
            filename: None,
            is_image: false,
        };

        let precomp_layer = LottieLayer {
            layer_type: LayerType::Precomp,
            index: Some(0),
            in_point: 0.0,
            out_point: 30.0,
            ref_id: Some("precomp_1".into()),
            ..Default::default()
        };

        let mut comp = make_test_composition();
        comp.assets = vec![asset];
        comp.layers = vec![precomp_layer];

        let mut artboard = Artboard::from_lottie(&comp);
        artboard.play();
        artboard.advance(0.0);
        artboard.update();
        let draws = artboard.draw();

        assert!(!draws.is_empty(), "Precomp children should produce draws");
        match &draws[0].paint {
            ShapePaint::SolidFill { color, .. } => {
                assert!(color.g > 0.9, "Should be green from precomp child");
            }
            _ => panic!("Expected SolidFill from precomp"),
        }
    }

    #[test]
    fn test_parented_layers_keep_global_draw_order() {
        let mut child = make_shape_layer(
            1,
            vec![make_rect(0.0, 0.0, 10.0, 10.0), make_fill(0.0, 1.0, 0.0)],
        );
        child.parent_index = Some(3);

        let sibling = make_shape_layer(
            2,
            vec![make_rect(0.0, 0.0, 10.0, 10.0), make_fill(0.0, 0.0, 1.0)],
        );

        let parent = make_shape_layer(
            3,
            vec![make_rect(0.0, 0.0, 10.0, 10.0), make_fill(1.0, 0.0, 0.0)],
        );

        let mut comp = make_test_composition();
        comp.layers = vec![child, sibling, parent];

        let mut artboard = Artboard::from_lottie(&comp);
        artboard.play();
        artboard.advance(0.0);
        artboard.update();
        let draws = artboard.draw();

        let colors: Vec<(u8, u8, u8)> = draws
            .iter()
            .filter_map(|draw| match &draw.paint {
                ShapePaint::SolidFill { color, .. } => Some((
                    (color.r * 255.0).round() as u8,
                    (color.g * 255.0).round() as u8,
                    (color.b * 255.0).round() as u8,
                )),
                _ => None,
            })
            .collect();

        assert_eq!(colors, vec![(255, 0, 0), (0, 0, 255), (0, 255, 0)]);
    }

    #[test]
    fn test_draw_sort_key() {
        use crate::renderer::draw::*;
        // Verify sort key ordering: higher blend mode → higher key
        let k1 = build_sort_key(
            BlendMode::Normal,
            DrawContents::Opaque,
            0,
            DrawType::MidpointFanFill,
        );
        let k2 = build_sort_key(
            BlendMode::Multiply,
            DrawContents::Opaque,
            0,
            DrawType::MidpointFanFill,
        );
        assert!(k2 > k1);
    }
}
