use crate::geometry::math::{Mat2D, AABB};
use crate::lottie;
use crate::scene::artboard::Artboard;
use crate::scene::sprite::Sprite;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Main canvas that manages multiple sprites
pub struct UltraCanvas {
    pub sprites: Vec<Sprite>,
    next_id: u64,
    pub view_bounds: AABB,
    pub viewport_width: f32,
    pub viewport_height: f32,
}

impl UltraCanvas {
    pub fn new(viewport_width: f32, viewport_height: f32) -> Self {
        Self {
            sprites: Vec::new(),
            next_id: 1,
            view_bounds: AABB::new(0.0, 0.0, viewport_width, viewport_height),
            viewport_width,
            viewport_height,
        }
    }

    /// Load a Lottie animation and add it as a sprite (native only)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn add_animation(&mut self, path: &str) -> Result<u64, String> {
        let comp = lottie::load_animation(path)?;
        let artboard = Artboard::from_lottie(&comp);
        let id = self.next_id;
        self.next_id += 1;

        let mut sprite = Sprite::new(id, artboard);
        sprite.play();
        self.sprites.push(sprite);
        Ok(id)
    }

    /// Load a Lottie animation from a JSON string (web/embedded)
    pub fn add_animation_from_json(&mut self, json: &str, _name: &str) -> Result<u64, String> {
        let comp = lottie::parse_lottie(json)?;
        let artboard = Artboard::from_lottie(&comp);
        let id = self.next_id;
        self.next_id += 1;

        let mut sprite = Sprite::new(id, artboard);
        sprite.play();
        self.sprites.push(sprite);
        Ok(id)
    }

    /// Add a pre-built sprite
    pub fn add_sprite(&mut self, mut sprite: Sprite) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        sprite.id = id;
        self.sprites.push(sprite);
        id
    }

    pub fn get_sprite(&self, id: u64) -> Option<&Sprite> {
        self.sprites.iter().find(|s| s.id == id)
    }

    pub fn get_sprite_mut(&mut self, id: u64) -> Option<&mut Sprite> {
        self.sprites.iter_mut().find(|s| s.id == id)
    }

    pub fn remove_sprite(&mut self, id: u64) -> bool {
        if let Some(pos) = self.sprites.iter().position(|s| s.id == id) {
            self.sprites.remove(pos);
            true
        } else {
            false
        }
    }

    /// Advance all sprites. When multiple sprites share the same animation,
    /// the master sprite is advanced and all others sync from it (Rive-style).
    /// This ensures identical geometry → single tessellation → single draw call.
    pub fn update(&mut self, delta_seconds: f32) {
        if self.sprites.is_empty() {
            return;
        }

        if self.sprites.len() > 1 && self.supports_synced_update() {
            let master_idx = self
                .sprites
                .iter()
                .position(|sprite| sprite.visible)
                .unwrap_or(0);
            let (before_master, master_and_after) = self.sprites.split_at_mut(master_idx);
            let (master, after_master) = master_and_after.split_first_mut().expect("non-empty");

            master.artboard.advance(delta_seconds);
            master.artboard.update();

            for sprite in before_master.iter_mut().chain(after_master.iter_mut()) {
                sprite.artboard.sync_playback_from(&master.artboard);
            }
            return;
        }

        for sprite in &mut self.sprites {
            sprite.update(delta_seconds);
        }
    }

    fn supports_synced_update(&self) -> bool {
        if self.sprites.len() < 2 {
            return false;
        }

        let first = &self.sprites[0];
        let first_name = first.artboard.name.as_str();
        let first_node_count = first.artboard.nodes.len();
        let first_width = first.composition_width.to_bits();
        let first_height = first.composition_height.to_bits();
        let first_frame_rate = first.artboard.player.frame_rate.to_bits();
        let first_in_point = first.artboard.player.in_point.to_bits();
        let first_out_point = first.artboard.player.out_point.to_bits();

        self.sprites.iter().skip(1).all(|sprite| {
            sprite.artboard.name == first_name
                && sprite.artboard.nodes.len() == first_node_count
                && sprite.composition_width.to_bits() == first_width
                && sprite.composition_height.to_bits() == first_height
                && sprite.artboard.player.frame_rate.to_bits() == first_frame_rate
                && sprite.artboard.player.in_point.to_bits() == first_in_point
                && sprite.artboard.player.out_point.to_bits() == first_out_point
        })
    }

    pub fn visible_sprites(&self) -> Vec<&Sprite> {
        let mut visible: Vec<&Sprite> = self
            .sprites
            .iter()
            .filter(|s| s.is_in_view(&self.view_bounds))
            .collect();
        visible.sort_by_key(|s| s.z_order);
        visible
    }

    /// Collect draw commands grouped by sprite, keeping paths in LOCAL artboard space.
    ///
    /// Returns:
    ///   `groups`:     Vec of `(sprite_index, Vec<ShapeDrawCommand>)` — one per visible sprite.
    ///   `transforms`: Vec of world transforms indexed by sprite_index.
    ///
    /// Synced sprites (same animation frame) produce identical `groups[i].1` lengths,
    /// which `encode_paths_instanced` detects and tessellates only once.
    ///
    /// Rive-style optimization: for synced sprites, only collect draw commands from
    /// the first sprite — the rest share the same geometry via GPU instancing.
    /// Collect draw groups. `zoom_transform` is applied to every sprite's
    /// world transform (Rive-style: zoom = scale, not camera).
    pub fn collect_draw_groups(
        &mut self,
        zoom_transform: &Mat2D,
    ) -> (
        Vec<(u32, Vec<crate::scene::layer::ShapeDrawCommand>)>,
        Vec<Mat2D>,
        bool,
        u64,
    ) {
        let mut groups: Vec<(u32, Vec<crate::scene::layer::ShapeDrawCommand>)> = Vec::new();
        let mut transforms: Vec<Mat2D> = Vec::new();
        let mut geometry_stamp = DefaultHasher::new();

        let mut visible_indices: Vec<usize> = self
            .sprites
            .iter()
            .enumerate()
            .filter_map(|(idx, sprite)| sprite.is_in_view(&self.view_bounds).then_some(idx))
            .collect();
        visible_indices.sort_by_key(|&idx| (self.sprites[idx].z_order, idx));

        let synced = if visible_indices.len() > 1 {
            let first = &self.sprites[visible_indices[0]];
            let first_frame = first.artboard.player.frame;
            let first_name = first.artboard.name.as_str();
            let first_node_count = first.artboard.nodes.len();
            let first_width = first.composition_width.to_bits();
            let first_height = first.composition_height.to_bits();

            visible_indices.iter().all(|&idx| {
                let sprite = &self.sprites[idx];
                (sprite.artboard.player.frame - first_frame).abs() < 0.01
                    && sprite.artboard.name == first_name
                    && sprite.artboard.nodes.len() == first_node_count
                    && sprite.composition_width.to_bits() == first_width
                    && sprite.composition_height.to_bits() == first_height
            })
        } else {
            false
        };
        synced.hash(&mut geometry_stamp);

        for &sprite_list_idx in &visible_indices {
            let sprite = &mut self.sprites[sprite_list_idx];
            let world = Mat2D::multiply(zoom_transform, &sprite.world_transform());
            let sprite_idx = transforms.len() as u32;
            transforms.push(world);

            if synced && !groups.is_empty() {
                continue;
            }

            let cmds = sprite.artboard.draw().to_vec();
            sprite.z_order.hash(&mut geometry_stamp);
            sprite.artboard.draw_version().hash(&mut geometry_stamp);
            cmds.len().hash(&mut geometry_stamp);
            groups.push((sprite_idx, cmds));
        }

        if !synced {
            transforms.len().hash(&mut geometry_stamp);
            groups.len().hash(&mut geometry_stamp);
        }

        (groups, transforms, synced, geometry_stamp.finish())
    }

    pub fn resize(&mut self, width: f32, height: f32) {
        self.viewport_width = width;
        self.viewport_height = height;
        self.view_bounds = AABB::new(0.0, 0.0, width, height);
    }

    pub fn sprite_count(&self) -> usize {
        self.sprites.len()
    }
}
