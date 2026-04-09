use wgpu::*;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use crate::geometry::tessellation::{GpuVertex, TessellateUniforms, EncodedPathData};

// ─── Frame uniforms (draw pass) ─────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view_proj: [f32; 16],
    pub resolution: [f32; 2],
    pub time: f32,
    pub _pad: f32,
}

// ─── GPU Buffer Manager ─────────────────────────────────────

/// Manages all GPU buffers for the Rive-style tessellation pipeline.
///
/// Flow:
///   CPU encode_paths() → segments[], indices[], paints[]
///   GPU compute (tessellate) → vertex_buffer (GpuVertex[])
///   GPU draw → reads vertex_buffer + index_buffer + paint_buffer
pub struct GpuBuffers {
    // Draw pass
    pub vertex_buffer: Option<Buffer>,    // output of compute, input to draw
    pub index_buffer: Option<Buffer>,
    pub paint_buffer: Option<Buffer>,     // GpuPaint[] for fragment shader
    pub uniform_buffer: Buffer,
    pub index_count: u32,

    // Tessellation compute pass
    pub segment_buffer: Option<Buffer>,   // GpuCubicSegment[] (input)
    pub tess_uniform_buffer: Buffer,      // TessellateUniforms
    pub segment_count: u32,
    pub total_vertices: u32,
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
        }
    }

    /// Upload encoded path data for the GPU tessellation pipeline.
    ///
    /// Creates:
    /// - segment_buffer: compute shader input (GpuCubicSegment[])
    /// - vertex_buffer: compute shader output / draw shader input (GpuVertex[])
    /// - index_buffer: draw shader index buffer
    /// - paint_buffer: fragment shader paint descriptors (GpuPaint[])
    pub fn upload_encoded_paths(&mut self, device: &Device, queue: &Queue, data: &EncodedPathData) {
        if data.segments.is_empty() || data.total_vertices == 0 {
            self.segment_count = 0;
            self.total_vertices = 0;
            self.index_count = 0;
            return;
        }

        // Segment buffer (compute input)
        self.segment_buffer = Some(device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("segment buffer"),
            contents: bytemuck::cast_slice(&data.segments),
            usage: BufferUsages::STORAGE,
        }));

        // Vertex buffer (compute output → draw input)
        let vertex_size = std::mem::size_of::<GpuVertex>() as u64;
        self.vertex_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("tess vertex buffer"),
            size: vertex_size * data.total_vertices as u64,
            usage: BufferUsages::STORAGE | BufferUsages::VERTEX,
            mapped_at_creation: false,
        }));

        // Index buffer
        if !data.indices.is_empty() {
            self.index_buffer = Some(device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("index buffer"),
                contents: bytemuck::cast_slice(&data.indices),
                usage: BufferUsages::INDEX,
            }));
        }

        // Paint buffer (fragment shader reads gradient params)
        if !data.paints.is_empty() {
            self.paint_buffer = Some(device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("paint buffer"),
                contents: bytemuck::cast_slice(&data.paints),
                usage: BufferUsages::STORAGE,
            }));
        }

        // Tess uniforms
        let tess_uniforms = TessellateUniforms {
            total_vertices: data.total_vertices,
            total_segments: data.segments.len() as u32,
            _pad: [0; 2],
        };
        queue.write_buffer(&self.tess_uniform_buffer, 0, bytemuck::bytes_of(&tess_uniforms));

        self.segment_count = data.segments.len() as u32;
        self.total_vertices = data.total_vertices;
        self.index_count = data.indices.len() as u32;
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

    /// Create bind group for the paint storage buffer (fragment shader).
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

    /// Create bind group for the tessellation compute shader.
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
}

/// Orthographic projection matrix for 2D rendering
pub fn ortho_projection(width: f32, height: f32) -> [f32; 16] {
    [
        2.0 / width,  0.0,           0.0, 0.0,
        0.0,          -2.0 / height, 0.0, 0.0,
        0.0,          0.0,           1.0, 0.0,
        -1.0,         1.0,           0.0, 1.0,
    ]
}
