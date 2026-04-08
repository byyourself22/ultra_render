use wgpu::*;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use crate::geometry::math::{Vec2D, Mat2D, Color};
use crate::geometry::tessellation::TessellatedMesh;

// ─── Rive-aligned GPU data structures ────────────────────────

/// Per-path data stored in GPU storage buffer (Rive PathData equivalent)
/// 256-byte aligned for cache efficiency
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PathData {
    pub matrix: [f32; 6],         // 2D affine transform (a,b,c,d,tx,ty)
    pub stroke_radius: f32,       // 0 = filled path
    pub feather_radius: f32,      // Feather width for AA
    pub z_index: u32,             // Z-ordering
    pub _pad: [u32; 7],          // Pad to 64 bytes
}

/// Per-contour data stored in GPU storage buffer (Rive ContourData equivalent)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ContourData {
    pub midpoint: [f32; 2],       // Center point for midpoint fan
    pub path_id: u32,             // Back-reference to parent PathData
    pub vertex_index0: u32,       // First tessellation vertex index
}

/// Per-path paint data (Rive PaintData equivalent)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PaintData {
    pub params: u32,              // [clipID(10) | flags(6) | paintType(3)]
    pub color_or_grad_y: u32,     // Packed RGBA8 color, or gradient texture Y
}

/// Paint auxiliary data for gradients/images (Rive PaintAuxData equivalent)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PaintAuxData {
    pub inv_matrix: [f32; 6],            // viewMatrix^-1 for gradient/image
    pub grad_texture_span: [f32; 2],     // Gradient horizontal span
    pub clip_rect_inv_matrix: [f32; 6],  // Clip rect inverse mapping
    pub inverse_fwidth: [f32; 2],        // Anti-aliasing factor
}

/// Tessellation vertex span — the core GPU tessellation input (Rive TessVertexSpan)
/// Each span is a 1px-tall rectangle containing cubic bezier data.
/// The GPU fragment shader evaluates it via parametric+polar merge + De Casteljau.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TessVertexSpan {
    pub pts: [[f32; 2]; 4],            // p0, p1, p2, p3 cubic control points
    pub join_tangent: [f32; 2],        // Ending tangent for stroke joins
    pub y: f32,                        // Y coordinate of span center
    pub reflection_y: f32,             // Y for reflection (NaN to disable)
    pub x0x1: i32,                     // Packed [x0(16-bit), x1(16-bit)]
    pub reflection_x0x1: i32,          // Packed reflection x0x1
    pub segment_counts: u32,           // [(joinSeg << 20) | (polarSeg << 10) | parametricSeg]
    pub contour_id_with_flags: u32,    // Contour ID + rendering flags
}

/// Paint type constants (matches Rive)
pub const SOLID_COLOR_PAINT_TYPE: u32 = 1;
pub const LINEAR_GRADIENT_PAINT_TYPE: u32 = 2;
pub const RADIAL_GRADIENT_PAINT_TYPE: u32 = 3;
pub const IMAGE_PAINT_TYPE: u32 = 4;

/// Contour flags (matches Rive)
pub const ROUND_JOIN: u32 = 1 << 26;
pub const BEVEL_JOIN: u32 = 2 << 26;
pub const MITER_JOIN: u32 = 3 << 26;

// ─── Flush uniforms (matches Rive FlushUniforms) ─────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FlushUniforms {
    pub render_target_inv_viewport: [f32; 2],  // 2/width, 2/height
    pub render_target_size: [f32; 2],          // width, height
    pub tess_inv_viewport_y: f32,              // For tessellation texture
    pub grad_inv_viewport_y: f32,              // For gradient texture
    pub time: f32,
    pub feather_texture_stddevs: f32,          // Gaussian sigma
    pub view_proj: [f32; 16],                  // Orthographic projection
}

// ─── Frame uniforms (simplified for current pipeline) ────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view_proj: [f32; 16],
    pub resolution: [f32; 2],
    pub time: f32,
    pub _pad: f32,
}

// ─── GPU Buffer Manager ──────────────────────────────────────

/// Manages all GPU buffers for the Rive-style render pipeline
pub struct GpuBuffers {
    // Legacy vertex/index buffers (used until full GPU tessellation)
    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub uniform_buffer: Buffer,
    pub vertex_count: u32,
    pub index_count: u32,

    // Rive-style storage buffers
    pub path_buffer: Option<Buffer>,
    pub contour_buffer: Option<Buffer>,
    pub paint_buffer: Option<Buffer>,
    pub paint_aux_buffer: Option<Buffer>,
    pub tess_vertex_span_buffer: Option<Buffer>,

    // Counts for Rive-style buffers
    pub path_count: u32,
    pub contour_count: u32,
    pub tess_span_count: u32,
}

impl GpuBuffers {
    pub fn new(device: &Device) -> Self {
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("uniform buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer: None,
            index_buffer: None,
            uniform_buffer,
            vertex_count: 0,
            index_count: 0,
            path_buffer: None,
            contour_buffer: None,
            paint_buffer: None,
            paint_aux_buffer: None,
            tess_vertex_span_buffer: None,
            path_count: 0,
            contour_count: 0,
            tess_span_count: 0,
        }
    }

    /// Upload a tessellated mesh to GPU (legacy path, until GPU tessellation is done)
    pub fn upload_mesh(&mut self, device: &Device, _queue: &Queue, mesh: &TessellatedMesh) {
        if mesh.is_empty() {
            self.vertex_count = 0;
            self.index_count = 0;
            return;
        }

        let vertex_data = bytemuck::cast_slice(&mesh.vertices);
        let index_data = bytemuck::cast_slice(&mesh.indices);

        self.vertex_buffer = Some(device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: vertex_data,
            usage: BufferUsages::VERTEX,
        }));

        self.index_buffer = Some(device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("index buffer"),
            contents: index_data,
            usage: BufferUsages::INDEX,
        }));

        self.vertex_count = mesh.vertex_count();
        self.index_count = mesh.index_count();
    }

    /// Upload Rive-style path data
    pub fn upload_paths(&mut self, device: &Device, paths: &[PathData]) {
        if paths.is_empty() {
            self.path_count = 0;
            return;
        }
        self.path_buffer = Some(device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("path buffer"),
            contents: bytemuck::cast_slice(paths),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        }));
        self.path_count = paths.len() as u32;
    }

    /// Upload Rive-style contour data
    pub fn upload_contours(&mut self, device: &Device, contours: &[ContourData]) {
        if contours.is_empty() {
            self.contour_count = 0;
            return;
        }
        self.contour_buffer = Some(device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("contour buffer"),
            contents: bytemuck::cast_slice(contours),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        }));
        self.contour_count = contours.len() as u32;
    }

    /// Upload Rive-style paint data
    pub fn upload_paints(&mut self, device: &Device, paints: &[PaintData], aux: &[PaintAuxData]) {
        if !paints.is_empty() {
            self.paint_buffer = Some(device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("paint buffer"),
                contents: bytemuck::cast_slice(paints),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }));
        }
        if !aux.is_empty() {
            self.paint_aux_buffer = Some(device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("paint aux buffer"),
                contents: bytemuck::cast_slice(aux),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }));
        }
    }

    /// Upload tessellation vertex spans for GPU tessellation
    pub fn upload_tess_spans(&mut self, device: &Device, spans: &[TessVertexSpan]) {
        if spans.is_empty() {
            self.tess_span_count = 0;
            return;
        }
        self.tess_vertex_span_buffer = Some(device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("tess vertex span buffer"),
            contents: bytemuck::cast_slice(spans),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        }));
        self.tess_span_count = spans.len() as u32;
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
}

// ─── Helper constructors ─────────────────────────────────────

impl PathData {
    pub fn from_mat2d(mat: &Mat2D, stroke_radius: f32, feather_radius: f32, z_index: u32) -> Self {
        Self {
            matrix: mat.values,
            stroke_radius,
            feather_radius,
            z_index,
            _pad: [0; 7],
        }
    }
}

impl ContourData {
    pub fn new(midpoint: Vec2D, path_id: u32, vertex_index0: u32) -> Self {
        Self {
            midpoint: [midpoint.x, midpoint.y],
            path_id,
            vertex_index0,
        }
    }
}

impl PaintData {
    pub fn solid_color(color: Color) -> Self {
        let r = (color.r * 255.0).clamp(0.0, 255.0) as u32;
        let g = (color.g * 255.0).clamp(0.0, 255.0) as u32;
        let b = (color.b * 255.0).clamp(0.0, 255.0) as u32;
        let a = (color.a * 255.0).clamp(0.0, 255.0) as u32;
        Self {
            params: SOLID_COLOR_PAINT_TYPE,
            color_or_grad_y: (a << 24) | (b << 16) | (g << 8) | r,
        }
    }
}

impl TessVertexSpan {
    /// Pack segment counts into a single u32 (Rive format)
    pub fn pack_segment_counts(parametric: u32, polar: u32, join: u32) -> u32 {
        (join & 0x3FF) << 20 | (polar & 0x3FF) << 10 | (parametric & 0x3FF)
    }

    /// Pack two i16 x-coordinates into one i32
    pub fn pack_x0x1(x0: i16, x1: i16) -> i32 {
        ((x1 as i32) << 16) | (x0 as u16 as i32)
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
