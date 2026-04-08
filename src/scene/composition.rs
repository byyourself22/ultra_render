use crate::geometry::math::Mat2D;
use crate::lottie::model::LottieComposition;
use super::layer::SceneLayer;
use super::animation::AnimationPlayer;

/// Root scene composition — the runtime representation of a Lottie animation
#[derive(Clone, Debug)]
pub struct SceneComposition {
    pub name: String,
    pub width: f32,
    pub height: f32,
    pub layers: Vec<SceneLayer>,
    pub player: AnimationPlayer,
}

impl SceneComposition {
    /// Create a scene from a parsed Lottie composition
    pub fn from_lottie(comp: &LottieComposition) -> Self {
        let mut layers: Vec<SceneLayer> = comp.layers.iter()
            .map(|l| SceneLayer::from_lottie(l))
            .collect();

        // Resolve precomp references
        for layer in &mut layers {
            if let Some(ref_id) = &layer.ref_id {
                if let Some(asset) = comp.assets.iter().find(|a| a.id == *ref_id) {
                    if !asset.is_image {
                        layer.children = asset.layers.iter()
                            .map(|l| SceneLayer::from_lottie(l))
                            .collect();
                    }
                }
            }
        }

        let player = AnimationPlayer::new(comp.frame_rate, comp.in_point, comp.out_point);

        Self {
            name: comp.name.clone(),
            width: comp.width,
            height: comp.height,
            layers,
            player,
        }
    }

    /// Advance animation and update all layer transforms
    pub fn update(&mut self, delta_seconds: f32) {
        self.player.advance(delta_seconds);
        let frame = self.player.frame;
        let identity = Mat2D::identity();
        update_layer_transforms(&mut self.layers, frame, &identity, 1.0);
    }

    /// Get current frame
    pub fn frame(&self) -> f32 {
        self.player.frame
    }

    /// Start playing
    pub fn play(&mut self) {
        self.player.play();
    }

    /// Collect all visible draw commands at current frame
    pub fn collect_draws(&self) -> Vec<super::layer::ShapeDrawCommand> {
        let frame = self.player.frame;
        let mut all_commands = Vec::new();
        collect_layer_draws(&self.layers, frame, &mut all_commands);
        all_commands
    }
}

fn update_layer_transforms(layers: &mut [SceneLayer], frame: f32, parent_world: &Mat2D, parent_opacity: f32) {
    // Build parent index map for resolving hierarchy
    let parent_transforms: Vec<(Option<i32>, Mat2D, f32)> = layers.iter()
        .map(|l| (l.index, l.computed.world, l.computed.opacity))
        .collect();

    for i in 0..layers.len() {
        let parent_idx = layers[i].parent_index;
        let (world, opacity) = if let Some(pid) = parent_idx {
            parent_transforms.iter()
                .find(|(idx, _, _)| *idx == Some(pid))
                .map(|(_, w, o)| (*w, *o))
                .unwrap_or((*parent_world, parent_opacity))
        } else {
            (*parent_world, parent_opacity)
        };

        layers[i].update_transform(frame, &world, opacity);

        // Update children (precomp layers)
        if !layers[i].children.is_empty() {
            let child_world = layers[i].computed.world;
            let child_opacity = layers[i].computed.opacity;
            update_layer_transforms(&mut layers[i].children, frame, &child_world, child_opacity);
        }
    }
}

fn collect_layer_draws(layers: &[SceneLayer], frame: f32, commands: &mut Vec<super::layer::ShapeDrawCommand>) {
    // Iterate in reverse order (Lottie layers are ordered back-to-front)
    for layer in layers.iter().rev() {
        if !layer.is_visible_at(frame) {
            continue;
        }

        match layer.layer_type {
            crate::lottie::model::LayerType::Shape => {
                let mut draws = layer.generate_paths(frame);
                commands.append(&mut draws);
            }
            crate::lottie::model::LayerType::Precomp => {
                collect_layer_draws(&layer.children, frame, commands);
            }
            _ => {}
        }
    }
}
