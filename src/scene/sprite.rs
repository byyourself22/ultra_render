use super::artboard::Artboard;
use crate::geometry::math::{Mat2D, Vec2D, AABB};

/// A sprite instance on the canvas.
/// Each sprite references an Artboard (Rive-style) and has its own transform/playback state.
#[derive(Clone, Debug)]
pub struct Sprite {
    pub id: u64,
    pub artboard: Artboard,
    pub position: Vec2D,
    pub scale: Vec2D,
    pub rotation: f32, // degrees
    pub opacity: f32,
    pub visible: bool,
    pub z_order: i32,

    // Legacy accessors for compatibility
    pub composition_width: f32,
    pub composition_height: f32,
}

impl Sprite {
    pub fn new(id: u64, artboard: Artboard) -> Self {
        let w = artboard.width;
        let h = artboard.height;
        Self {
            id,
            artboard,
            position: Vec2D::ZERO,
            scale: Vec2D::new(1.0, 1.0),
            rotation: 0.0,
            opacity: 1.0,
            visible: true,
            z_order: 0,
            composition_width: w,
            composition_height: h,
        }
    }

    /// Get the world transform for this sprite
    pub fn world_transform(&self) -> Mat2D {
        Mat2D::from_components(
            self.position.x,
            self.position.y,
            self.scale.x,
            self.scale.y,
            self.rotation * std::f32::consts::PI / 180.0,
        )
    }

    /// Get the bounding box in world space
    pub fn bounds(&self) -> AABB {
        let local_aabb = AABB::new(0.0, 0.0, self.composition_width, self.composition_height);
        local_aabb.transform(&self.world_transform())
    }

    /// Update the sprite's animation (Rive-style advance + update cycle)
    pub fn update(&mut self, delta_seconds: f32) {
        if self.visible {
            self.artboard.advance(delta_seconds);
            self.artboard.update();
        }
    }

    /// Play the animation
    pub fn play(&mut self) {
        self.artboard.play();
    }

    /// Check if this sprite is within a view bounds (for culling)
    pub fn is_in_view(&self, view_bounds: &AABB) -> bool {
        if !self.visible {
            return false;
        }
        self.bounds().intersects(view_bounds)
    }
}
