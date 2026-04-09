use crate::geometry::math::AABB;
use crate::lottie;
use crate::scene::artboard::Artboard;
use crate::scene::sprite::Sprite;

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

    /// Load a Lottie animation and add it as a sprite (now uses Artboard)
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

    /// Load a Lottie animation from a JSON string (for web/embedded)
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

    /// Get a sprite by ID
    pub fn get_sprite(&self, id: u64) -> Option<&Sprite> {
        self.sprites.iter().find(|s| s.id == id)
    }

    /// Get a mutable sprite by ID
    pub fn get_sprite_mut(&mut self, id: u64) -> Option<&mut Sprite> {
        self.sprites.iter_mut().find(|s| s.id == id)
    }

    /// Remove a sprite by ID
    pub fn remove_sprite(&mut self, id: u64) -> bool {
        if let Some(pos) = self.sprites.iter().position(|s| s.id == id) {
            self.sprites.remove(pos);
            true
        } else {
            false
        }
    }

    /// Update all sprites
    pub fn update(&mut self, delta_seconds: f32) {
        for sprite in &mut self.sprites {
            sprite.update(delta_seconds);
        }
    }

    /// Get visible sprites sorted by z_order
    pub fn visible_sprites(&self) -> Vec<&Sprite> {
        let mut visible: Vec<&Sprite> = self.sprites.iter()
            .filter(|s| s.is_in_view(&self.view_bounds))
            .collect();
        visible.sort_by_key(|s| s.z_order);
        visible
    }

    /// Collect all draw commands from all visible sprites into a flat list.
    /// Each sprite's world transform is pre-applied to its paths so a single
    /// render call (identity transform) can draw everything in one frame.
    pub fn collect_all_draws(&mut self) -> Vec<crate::scene::layer::ShapeDrawCommand> {
        let mut all = Vec::new();
        for sprite in &mut self.sprites {
            if !sprite.is_in_view(&self.view_bounds) {
                continue;
            }
            let world = sprite.world_transform();
            for draw in sprite.artboard.draw() {
                let mut d = draw.clone();
                d.path = d.path.transform(&world);
                all.push(d);
            }
        }
        all
    }

    /// Resize viewport
    pub fn resize(&mut self, width: f32, height: f32) {
        self.viewport_width = width;
        self.viewport_height = height;
        self.view_bounds = AABB::new(0.0, 0.0, width, height);
    }

    pub fn sprite_count(&self) -> usize {
        self.sprites.len()
    }
}
