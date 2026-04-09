use wgpu::*;
use bytemuck::{Pod, Zeroable};
use crate::geometry::tessellation::{GpuVertex, TessellateUniforms, EncodedPathData};

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

/// Manages all GPU buffers for the hybrid fill+stroke pipeline.
///
/// Fill:   Lyon vertices uploaded directly at start of vertex_buffer.
/// Stroke: Compute shader writes after fill region.
pub struct GpuBuffers {
    // Draw pass
    pub vertex_buffer: Option<Buffer>,    // fill region (CPU) + stroke region (compute)
    pub index_buffer: Option<Buffer>,
    pub paint_buffer: Option<Buffer>,
    pub uniform_buffer: Buffer,
    pub index_count: u32,

    // Tessellation compute pass (strokes only)
    pub segment_buffer: Option<Buffer>,
    pub tess_uniform_buffer: Buffer,
    pub segment_count: u32,
    pub total_vertices: u32,

    // Capacity tracking
    vertex_cap: u64,
    segment_cap: u64,
    index_cap: u64,
    paint_cap: u64,
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

        Self {
            vertex_buffer: None,
            index_buffer: None,
            paint_buffer: None,
            uniform_buffer,
            index_count: 0,
            segment_buffer: None,
            tess_uniform_buffer,
            segment_count: 0,
            total_vertices: 0,
            vertex_cap: 0,
            segment_cap: 0,
            index_cap: 0,
            paint_cap: 0,
        }
    }

    /// Upload encoded path data.
    ///
    /// 1. Allocate vertex_buffer for fill + stroke vertices.
    /// 2. Pre-populate fill vertices at offset 0.
    /// 3. Upload stroke segments (compute will write stroke vertices after fill region).
    /// 4. Upload combined index buffer.
    pub fn upload_encoded_paths(&mut self, device: &Device, queue: &Queue, data: &EncodedPathData) {
        let total_verts = data.total_vertex_count();
        let total_idxs = data.total_index_count();

        if total_verts == 0 || total_idxs == 0 {
            self.segment_count = 0;
            self.total_vertices = 0;
            self.index_count = 0;
            return;
        }

        // ── Vertex buffer (fill + stroke, COPY_DST for fill pre-population) ─
        let vertex_size = std::mem::size_of::<GpuVertex>() as u64;
        let needed_vertex_bytes = vertex_size * total_verts as u64;
        if needed_vertex_bytes > self.vertex_cap {
            let new_cap = (needed_vertex_bytes * 3 / 2).max(4096);
            self.vertex_buffer = Some(device.create_buffer(&BufferDescriptor {
                label: Some("vertex buffer"),
                size: new_cap,
                // STORAGE: compute shader writes stroke vertices
                // VERTEX:  draw shader reads all vertices
                // COPY_DST: queue.write_buffer for fill vertices
                usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.vertex_cap = new_cap;
        }

        // Pre-populate fill vertices at the start of the buffer
        if !data.fill_vertices.is_empty() {
            let fill_bytes = bytemuck::cast_slice::<GpuVertex, u8>(&data.fill_vertices);
            if let Some(vb) = &self.vertex_buffer {
                queue.write_buffer(vb, 0, fill_bytes);
            }
        }

        // ── Segment buffer (compute input, strokes only) ─────────────────
        if !data.segments.is_empty() {
            let seg_bytes = bytemuck::cast_slice::<_, u8>(&data.segments);
            Self::ensure_storage_buffer(
                device, queue,
                &mut self.segment_buffer, &mut self.segment_cap,
                seg_bytes,
                BufferUsages::STORAGE,
                "segment buffer",
            );
        }

        // ── Index buffer (fill_indices + stroke_indices combined) ─────────
        let combined = data.combined_indices();
        if !combined.is_empty() {
            let idx_bytes = bytemuck::cast_slice::<_, u8>(&combined);
            Self::ensure_index_buffer(
                device, queue,
                &mut self.index_buffer, &mut self.index_cap,
                idx_bytes,
            );
        }

        // ── Paint buffer ─────────────────────────────────────────────────
        if !data.paints.is_empty() {
            let paint_bytes = bytemuck::cast_slice::<_, u8>(&data.paints);
            Self::ensure_storage_buffer(
                device, queue,
                &mut self.paint_buffer, &mut self.paint_cap,
                paint_bytes,
                BufferUsages::STORAGE,
                "paint buffer",
            );
        }

        // ── Tess uniforms (total_vertices = full buffer size for bounds check) ─
        let tess_uniforms = TessellateUniforms {
            total_vertices: total_verts,  // full buffer, not just stroke count
            total_segments: data.segments.len() as u32,
            _pad: [0; 2],
        };
        queue.write_buffer(&self.tess_uniform_buffer, 0, bytemuck::bytes_of(&tess_uniforms));

        self.segment_count = data.segments.len() as u32;
        self.total_vertices = total_verts;
        self.index_count = total_idxs;
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
        if needed == 0 { return; }
        if buf.is_none() || *cap < needed {
            let new_cap = (needed * 3 / 2).max(256);
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
        if needed == 0 { return; }
        if buf.is_none() || *cap < needed {
            let new_cap = (needed * 3 / 2).max(256);
            *buf = Some(device.create_buffer(&BufferDescriptor {
                label: Some("index buffer"),
                size: new_cap,
                usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
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

    pub fn create_uniform_bind_group(&self, device: &Device, layout: &BindGroupLayout) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: self.uniform_buffer.as_entire_binding(),
            }],
        })
    }

    pub fn create_paint_bind_group(&self, device: &Device, layout: &BindGroupLayout) -> Option<BindGroup> {
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

    pub fn create_tessellate_bind_group(&self, device: &Device, layout: &BindGroupLayout) -> Option<BindGroup> {
        let seg_buf = self.segment_buffer.as_ref()?;
        let vert_buf = self.vertex_buffer.as_ref()?;

        Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("tessellate bind group"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: self.tess_uniform_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: seg_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: vert_buf.as_entire_binding() },
            ],
        }))
    }
}

pub fn ortho_projection(width: f32, height: f32) -> [f32; 16] {
    [
        2.0 / width,  0.0,           0.0, 0.0,
        0.0,          -2.0 / height, 0.0, 0.0,
        0.0,          0.0,           1.0, 0.0,
        -1.0,         1.0,           0.0, 1.0,
    ]
}
