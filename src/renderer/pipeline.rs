use wgpu::*;
use super::context::MSAA_SAMPLES;

// ─── Draw pipeline (renders tessellated vertices) ───────────

/// Creates the main render pipeline for path drawing.
/// Reads GpuVertex data (position + uv + color + world_pos + paint_index)
/// from vertex buffer. Bind group 0 = frame uniforms, group 1 = paint storage.
pub fn create_path_pipeline(
    device: &Device,
    format: TextureFormat,
) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("draw_path shader"),
        source: ShaderSource::Wgsl(include_str!("shaders/draw_path.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("path pipeline layout"),
        bind_group_layouts: &[
            &create_uniform_bind_group_layout(device),
            &create_paint_bind_group_layout(device),
        ],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("path pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_buffer_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(ColorTargetState {
                format,
                blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: MultisampleState {
            count: MSAA_SAMPLES,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

// ─── GPU tessellation compute pipeline ──────────────────────

/// Creates the compute pipeline that evaluates cubic bezier segments
/// on the GPU (Rive-style De Casteljau tessellation).
pub fn create_tessellate_pipeline(device: &Device) -> ComputePipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("tessellate compute shader"),
        source: ShaderSource::Wgsl(include_str!("shaders/tessellate.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("tessellate pipeline layout"),
        bind_group_layouts: &[&create_tessellate_bind_group_layout(device)],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("tessellate compute pipeline"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("cs_tessellate"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Bind group layout for the tessellation compute shader.
/// binding 0: TessUniforms (uniform)
/// binding 1: GpuCubicSegment[] (storage, read)
/// binding 2: GpuVertex[] (storage, read_write)
pub fn create_tessellate_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("tessellate bind group layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

// ─── Shared layouts ─────────────────────────────────────────

pub fn create_uniform_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("uniform bind group layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Bind group layout for the paint storage buffer (read by fragment shader).
pub fn create_paint_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("paint bind group layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn vertex_buffer_layout() -> VertexBufferLayout<'static> {
    VertexBufferLayout {
        array_stride: std::mem::size_of::<crate::geometry::tessellation::GpuVertex>() as BufferAddress,
        step_mode: VertexStepMode::Vertex,
        attributes: &[
            // position: vec2<f32>
            VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: VertexFormat::Float32x2,
            },
            // uv: vec2<f32>
            VertexAttribute {
                offset: 8,
                shader_location: 1,
                format: VertexFormat::Float32x2,
            },
            // color: vec4<f32>
            VertexAttribute {
                offset: 16,
                shader_location: 2,
                format: VertexFormat::Float32x4,
            },
            // world_pos: vec2<f32>
            VertexAttribute {
                offset: 32,
                shader_location: 3,
                format: VertexFormat::Float32x2,
            },
            // paint_index: u32
            VertexAttribute {
                offset: 40,
                shader_location: 4,
                format: VertexFormat::Uint32,
            },
        ],
    }
}
