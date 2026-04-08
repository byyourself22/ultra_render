use crate::geometry::math::Mat2D;
use crate::scene::sprite::Sprite;

/// Groups sprites that share the same animation for batch rendering.
/// Sprites using the same Lottie source can share tessellation data.
pub struct SpriteBatch {
    pub animation_name: String,
    pub sprite_ids: Vec<u64>,
    pub transforms: Vec<Mat2D>,
}

/// Analyze sprites and group them into batches
pub fn create_batches(sprites: &[&Sprite]) -> Vec<SpriteBatch> {
    let mut batches: Vec<SpriteBatch> = Vec::new();

    for sprite in sprites {
        let name = &sprite.artboard.name;
        if let Some(batch) = batches.iter_mut().find(|b| b.animation_name == *name) {
            batch.sprite_ids.push(sprite.id);
            batch.transforms.push(sprite.world_transform());
        } else {
            batches.push(SpriteBatch {
                animation_name: name.clone(),
                sprite_ids: vec![sprite.id],
                transforms: vec![sprite.world_transform()],
            });
        }
    }

    batches
}
