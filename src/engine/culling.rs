use crate::geometry::math::AABB;
use crate::scene::sprite::Sprite;

/// Frustum culling: filter sprites to only those visible in the view
pub fn cull_sprites<'a>(sprites: &'a [Sprite], view: &AABB) -> Vec<&'a Sprite> {
    sprites.iter().filter(|s| s.is_in_view(view)).collect()
}

/// Simple grid-based spatial index for fast culling with many sprites
pub struct SpatialGrid {
    pub cell_size: f32,
    pub cells: std::collections::HashMap<(i32, i32), Vec<u64>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: std::collections::HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }

    pub fn insert(&mut self, id: u64, bounds: &AABB) {
        let min_cx = (bounds.min_x / self.cell_size).floor() as i32;
        let min_cy = (bounds.min_y / self.cell_size).floor() as i32;
        let max_cx = (bounds.max_x / self.cell_size).ceil() as i32;
        let max_cy = (bounds.max_y / self.cell_size).ceil() as i32;

        for cx in min_cx..=max_cx {
            for cy in min_cy..=max_cy {
                self.cells.entry((cx, cy)).or_default().push(id);
            }
        }
    }

    pub fn query(&self, bounds: &AABB) -> Vec<u64> {
        let min_cx = (bounds.min_x / self.cell_size).floor() as i32;
        let min_cy = (bounds.min_y / self.cell_size).floor() as i32;
        let max_cx = (bounds.max_x / self.cell_size).ceil() as i32;
        let max_cy = (bounds.max_y / self.cell_size).ceil() as i32;

        let mut result: Vec<u64> = Vec::new();
        for cx in min_cx..=max_cx {
            for cy in min_cy..=max_cy {
                if let Some(ids) = self.cells.get(&(cx, cy)) {
                    for id in ids {
                        if !result.contains(id) {
                            result.push(*id);
                        }
                    }
                }
            }
        }
        result
    }
}
