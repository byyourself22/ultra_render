use wgpu::*;

/// Manages GPU textures for the renderer
pub struct TextureManager {
    pub gradient_texture: Option<Texture>,
    pub gradient_view: Option<TextureView>,
}

impl TextureManager {
    pub fn new() -> Self {
        Self {
            gradient_texture: None,
            gradient_view: None,
        }
    }

    /// Create a 1D gradient ramp texture
    pub fn create_gradient_ramp(
        &mut self,
        device: &Device,
        queue: &Queue,
        colors: &[f32],
        color_count: usize,
        width: u32,
    ) {
        let mut pixels = vec![0u8; (width * 4) as usize];

        if color_count < 2 || colors.len() < color_count * 4 {
            // Fill with white
            for chunk in pixels.chunks_mut(4) {
                chunk.copy_from_slice(&[255, 255, 255, 255]);
            }
        } else {
            // Interpolate gradient stops
            for x in 0..width {
                let t = x as f32 / (width - 1) as f32;
                let (r, g, b, a) = sample_gradient(colors, color_count, t);
                let idx = (x * 4) as usize;
                pixels[idx] = (r * 255.0).clamp(0.0, 255.0) as u8;
                pixels[idx + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
                pixels[idx + 2] = (b * 255.0).clamp(0.0, 255.0) as u8;
                pixels[idx + 3] = (a * 255.0).clamp(0.0, 255.0) as u8;
            }
        }

        let texture = device.create_texture(&TextureDescriptor {
            label: Some("gradient ramp"),
            size: Extent3d {
                width,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &pixels,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(1),
            },
            Extent3d {
                width,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        self.gradient_view = Some(texture.create_view(&TextureViewDescriptor::default()));
        self.gradient_texture = Some(texture);
    }
}

/// Sample a gradient at position t, given color stops [offset, r, g, b, ...]
fn sample_gradient(colors: &[f32], color_count: usize, t: f32) -> (f32, f32, f32, f32) {
    if color_count == 0 {
        return (1.0, 1.0, 1.0, 1.0);
    }

    let stride = 4; // offset + r + g + b

    // Find the two stops surrounding t
    let mut prev_idx = 0;
    let mut next_idx = 0;

    for i in 0..color_count {
        let offset = colors[i * stride];
        if offset <= t {
            prev_idx = i;
        }
        if offset >= t {
            next_idx = i;
            break;
        }
        next_idx = i;
    }

    if prev_idx == next_idx {
        let base = prev_idx * stride;
        return (colors[base + 1], colors[base + 2], colors[base + 3], 1.0);
    }

    let prev_base = prev_idx * stride;
    let next_base = next_idx * stride;
    let prev_offset = colors[prev_base];
    let next_offset = colors[next_base];

    let segment_t = if (next_offset - prev_offset).abs() < 1e-6 {
        0.0
    } else {
        (t - prev_offset) / (next_offset - prev_offset)
    };

    let r = colors[prev_base + 1] + (colors[next_base + 1] - colors[prev_base + 1]) * segment_t;
    let g = colors[prev_base + 2] + (colors[next_base + 2] - colors[prev_base + 2]) * segment_t;
    let b = colors[prev_base + 3] + (colors[next_base + 3] - colors[prev_base + 3]) * segment_t;

    (r, g, b, 1.0)
}
