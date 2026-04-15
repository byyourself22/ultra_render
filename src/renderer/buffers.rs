use crate::geometry::tessellation::{
    ComplexFillDraw, EncodedDraw, EncodedPathData, FillTessUniforms, GpuVertex, SpriteTransform,
    TessellateUniforms,
};
use bytemuck::{Pod, Zeroable};
use wgpu::*;

// ─── Frame uniforms (draw pass) ─────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view_proj: [f32; 16],
    pub resolution: [f32; 2],
    pub time: f32,
    pub is_srgb: u32,
}

// ─── GPU Buffer Manager ─────────────────────────────────────

pub struct GpuBuffers {
    // Draw pass
    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub paint_buffer: Option<Buffer>,
    pub sprite_transform_buffer: Option<Buffer>,
    pub uniform_buffer: Buffer,
    pub index_count: u32,
    pub simple_index_count: u32,
    /// GPU instancing: how many instances to draw (>1 for synced sprites).
    pub instance_count: u32,
    pub complex_fill_draws: Vec<ComplexFillDraw>,
    pub ordered_draws: Vec<EncodedDraw>,

    // Tessellation compute pass (strokes only)
    pub segment_buffer: Option<Buffer>,
    pub tess_uniform_buffer: Buffer,
    pub segment_count: u32,
    pub total_vertices: u32,

    // Fill tessellation compute pass
    pub fill_contour_buffer: Option<Buffer>,
    pub fill_cubic_buffer: Option<Buffer>,
    pub fill_tess_uniform_buffer: Buffer,
    pub fill_contour_count: u32,
    pub fill_cubic_count: u32,

    // Capacity tracking
    vertex_cap: u64,
    segment_cap: u64,
    index_cap: u64,
    paint_cap: u64,
    sprite_transform_cap: u64,
    fill_contour_cap: u64,
    fill_cubic_cap: u64,
}

impl GpuBuffers {
    pub fn new(device: &Device) -> Self {
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("frame uniform buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tess_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("tess uniform buffer"),
            size: std::mem::size_of::<TessellateUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fill_tess_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("fill tess uniform buffer"),
            size: std::mem::size_of::<FillTessUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer: None,
            index_buffer: None,
            paint_buffer: None,
            sprite_transform_buffer: None,
            uniform_buffer,
            index_count: 0,
            simple_index_count: 0,
            instance_count: 1,
            complex_fill_draws: Vec::new(),
            ordered_draws: Vec::new(),
            segment_buffer: None,
            tess_uniform_buffer,
            segment_count: 0,
            total_vertices: 0,
            fill_contour_buffer: None,
            fill_cubic_buffer: None,
            fill_tess_uniform_buffer,
            fill_contour_count: 0,
            fill_cubic_count: 0,
            vertex_cap: 0,
            segment_cap: 0,
            index_cap: 0,
            paint_cap: 0,
            sprite_transform_cap: 0,
            fill_contour_cap: 0,
            fill_cubic_cap: 0,
        }
    }

    /// Upload encoded path data (fill vertices + stroke segments + paints).
    pub fn upload_encoded_paths(&mut self, device: &Device, queue: &Queue, data: &EncodedPathData) {
        let total_verts = data.total_vertex_count();
        let total_idxs = data.total_index_count();

        if total_verts == 0 || total_idxs == 0 {
            self.segment_count = 0;
            self.total_vertices = 0;
            self.index_count = 0;
            self.simple_index_count = 0;
            self.instance_count = 1;
            self.complex_fill_draws.clear();
            self.ordered_draws.clear();
            self.fill_contour_count = 0;
            self.fill_cubic_count = 0;
            return;
        }

        // Vertex buffer (fill + stroke, COPY_DST for fill pre-population / CPU fallback fills)
        let vertex_size = std::mem::size_of::<GpuVertex>() as u64;
        let needed_vertex_bytes = vertex_size * total_verts as u64;
        if needed_vertex_bytes > self.vertex_cap {
            let new_cap = Self::next_storage_capacity(needed_vertex_bytes, 4096);
            self.vertex_buffer = Some(device.create_buffer(&BufferDescriptor {
                label: Some("vertex buffer"),
                size: new_cap,
                usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.vertex_cap = new_cap;
        }
        if !data.fill_vertices.is_empty() {
            let fill_bytes = bytemuck::cast_slice::<GpuVertex, u8>(&data.fill_vertices);
            if let Some(vb) = &self.vertex_buffer {
                queue.write_buffer(vb, 0, fill_bytes);
            }
        }

        // Fill contour / cubic buffers. Complex contour topology lives in the
        // tail of the combined index buffer, so patch those offsets before upload.
        let mut fill_contours = data.fill_contours.clone();
        let complex_index_offset = (data.fill_indices.len() + data.stroke_indices.len()) as u32;
        for contour in &mut fill_contours {
            if contour.simple_fill >= 2 {
                contour.first_index += complex_index_offset;
            }
        }
        if !fill_contours.is_empty() {
            let contour_bytes = bytemuck::cast_slice::<_, u8>(&fill_contours);
            Self::ensure_storage_buffer(
                device,
                queue,
                &mut self.fill_contour_buffer,
                &mut self.fill_contour_cap,
                contour_bytes,
                BufferUsages::STORAGE,
                "fill contour buffer",
            );
        }
        if !data.fill_cubics.is_empty() {
            let cubic_bytes = bytemuck::cast_slice::<_, u8>(&data.fill_cubics);
            Self::ensure_storage_buffer(
                device,
                queue,
                &mut self.fill_cubic_buffer,
                &mut self.fill_cubic_cap,
                cubic_bytes,
                BufferUsages::STORAGE,
                "fill cubic buffer",
            );
        }

        // Stroke segment buffer
        if !data.segments.is_empty() {
            let seg_bytes = bytemuck::cast_slice::<_, u8>(&data.segments);
            Self::ensure_storage_buffer(
                device,
                queue,
                &mut self.segment_buffer,
                &mut self.segment_cap,
                seg_bytes,
                BufferUsages::STORAGE,
                "segment buffer",
            );
        }

        // Combined index buffer (fill_indices + stroke_indices)
        let combined = data.combined_indices();
        if !combined.is_empty() {
            let idx_bytes = bytemuck::cast_slice::<_, u8>(&combined);
            Self::ensure_index_buffer(
                device,
                queue,
                &mut self.index_buffer,
                &mut self.index_cap,
                idx_bytes,
            );
        }

        // Paint buffer
        if !data.paints.is_empty() {
            let paint_bytes = bytemuck::cast_slice::<_, u8>(&data.paints);
            Self::ensure_storage_buffer(
                device,
                queue,
                &mut self.paint_buffer,
                &mut self.paint_cap,
                paint_bytes,
                BufferUsages::STORAGE,
                "paint buffer",
            );
        }

        // Tess uniforms
        let tess_uniforms = TessellateUniforms {
            total_vertices: total_verts,
            total_segments: data.segments.len() as u32,
            _pad: [0; 2],
        };
        queue.write_buffer(
            &self.tess_uniform_buffer,
            0,
            bytemuck::bytes_of(&tess_uniforms),
        );

        let fill_tess_uniforms = FillTessUniforms {
            total_cubics: data.fill_cubics.len() as u32,
            total_contours: data.fill_contours.len() as u32,
            _pad: [0; 2],
        };
        queue.write_buffer(
            &self.fill_tess_uniform_buffer,
            0,
            bytemuck::bytes_of(&fill_tess_uniforms),
        );

        self.segment_count = data.segments.len() as u32;
        self.total_vertices = total_verts;
        self.index_count = total_idxs;
        self.simple_index_count = (data.fill_indices.len() + data.stroke_indices.len()) as u32;
        self.instance_count = data.instance_count;
        self.complex_fill_draws = data
            .complex_fill_draws
            .iter()
            .map(|draw| ComplexFillDraw {
                stencil_index_start: draw.stencil_index_start + self.simple_index_count,
                cover_index_start: draw.cover_index_start + self.simple_index_count,
                ..*draw
            })
            .collect();
        self.ordered_draws = data
            .ordered_draws
            .iter()
            .map(|draw| match *draw {
                EncodedDraw::Simple { .. } => *draw,
                EncodedDraw::Complex(draw) => EncodedDraw::Complex(ComplexFillDraw {
                    stencil_index_start: draw.stencil_index_start + self.simple_index_count,
                    cover_index_start: draw.cover_index_start + self.simple_index_count,
                    ..draw
                }),
            })
            .collect();
        self.fill_contour_count = data.fill_contours.len() as u32;
        self.fill_cubic_count = data.fill_cubics.len() as u32;
    }

    /// Upload per-sprite transforms to the GPU storage buffer.
    /// Must be called before render if sprite count changed.
    pub fn upload_sprite_transforms(
        &mut self,
        device: &Device,
        queue: &Queue,
        transforms: &[SpriteTransform],
    ) {
        let data = bytemuck::cast_slice::<SpriteTransform, u8>(transforms);
        Self::ensure_storage_buffer(
            device,
            queue,
            &mut self.sprite_transform_buffer,
            &mut self.sprite_transform_cap,
            data,
            BufferUsages::STORAGE,
            "sprite transforms buffer",
        );
    }

    fn next_storage_capacity(needed: u64, min_size: u64) -> u64 {
        ((needed * 3 / 2).max(min_size) + 3) & !3
    }
    fn ensure_storage_buffer(
        device: &Device,
        queue: &Queue,
        buf: &mut Option<Buffer>,
        cap: &mut u64,
        data: &[u8],
        extra_usage: BufferUsages,
        label: &str,
    ) {
        let needed = data.len() as u64;
        if needed == 0 {
            return;
        }
        if buf.is_none() || *cap < needed {
            let new_cap = Self::next_storage_capacity(needed, 256);
            *buf = Some(device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: new_cap,
                usage: extra_usage | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            *cap = new_cap;
        }
        if let Some(b) = buf.as_ref() {
            queue.write_buffer(b, 0, data);
        }
    }

    fn ensure_index_buffer(
        device: &Device,
        queue: &Queue,
        buf: &mut Option<Buffer>,
        cap: &mut u64,
        data: &[u8],
    ) {
        let needed = data.len() as u64;
        if needed == 0 {
            return;
        }
        if buf.is_none() || *cap < needed {
            let new_cap = Self::next_storage_capacity(needed, 256);
            *buf = Some(device.create_buffer(&BufferDescriptor {
                label: Some("index buffer"),
                size: new_cap,
                usage: BufferUsages::INDEX | BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            *cap = new_cap;
        }
        if let Some(b) = buf.as_ref() {
            queue.write_buffer(b, 0, data);
        }
    }

    pub fn update_uniforms(&self, queue: &Queue, uniforms: &FrameUniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    pub fn create_uniform_bind_group(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: self.uniform_buffer.as_entire_binding(),
            }],
        })
    }

    pub fn create_paint_bind_group(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
    ) -> Option<BindGroup> {
        let paint_buf = self.paint_buffer.as_ref()?;
        Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("paint bind group"),
            layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: paint_buf.as_entire_binding(),
            }],
        }))
    }

    pub fn create_sprite_transforms_bind_group(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
    ) -> Option<BindGroup> {
        let buf = self.sprite_transform_buffer.as_ref()?;
        Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("sprite transforms bind group"),
            layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        }))
    }

    pub fn create_tessellate_bind_group(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
    ) -> Option<BindGroup> {
        let seg_buf = self.segment_buffer.as_ref()?;
        let vert_buf = self.vertex_buffer.as_ref()?;

        Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("tessellate bind group"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.tess_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: seg_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: vert_buf.as_entire_binding(),
                },
            ],
        }))
    }

    pub fn create_fill_tessellate_bind_group(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
    ) -> Option<BindGroup> {
        let contour_buf = self.fill_contour_buffer.as_ref()?;
        let cubic_buf = self.fill_cubic_buffer.as_ref()?;
        let vert_buf = self.vertex_buffer.as_ref()?;
        let idx_buf = self.index_buffer.as_ref()?;

        Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("fill tessellate bind group"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.fill_tess_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: contour_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: cubic_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: vert_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: idx_buf.as_entire_binding(),
                },
            ],
        }))
    }
}

pub fn ortho_projection(width: f32, height: f32) -> [f32; 16] {
    [
        2.0 / width,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.0 / height,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -1.0,
        1.0,
        0.0,
        1.0,
    ]
}

/// Orthographic projection with zoom and pan.
///
/// `zoom`: scale factor (>1 = closer). `pan_x`/`pan_y`: camera offset in world pixels.
///
/// Camera center in world space = (w/2 + pan_x, h/2 + pan_y).
/// Visible area = w/zoom × h/zoom pixels centered on that point.
pub fn ortho_projection_zoom(
    width: f32,
    height: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
) -> [f32; 16] {
    let sx = zoom * 2.0 / width;
    let sy = -zoom * 2.0 / height;
    let tx = -zoom * (1.0 + 2.0 * pan_x / width);
    let ty = zoom * (1.0 + 2.0 * pan_y / height);
    [
        sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, 0.0, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_capacities_are_4_byte_aligned() {
        for (needed, min_size) in [(1, 256), (3, 256), (257, 256), (101034, 256), (4097, 4096)] {
            let cap = GpuBuffers::next_storage_capacity(needed, min_size);
            assert_eq!(cap % 4, 0);
            assert!(cap >= needed.max(min_size));
        }
    }
}
