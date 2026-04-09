use wgpu::*;
use crate::geometry::math::Mat2D;
use crate::geometry::tessellation::encode_paths;
use super::context::RenderContext;
use super::pipeline;
use super::buffers::{GpuBuffers, FrameUniforms, ortho_projection};

/// Main render canvas — orchestrates GPU tessellation + rendering.
///
/// Pipeline:
///   1. CPU: encode_paths() packs cubic segments + indices + paints
///   2. GPU compute: tessellate cubics → vertex buffer (De Casteljau)
///   3. GPU draw: render vertices with paint storage (gradients) + MSAA
pub struct RenderCanvas {
    pub path_pipeline: RenderPipeline,
    pub tessellate_pipeline: ComputePipeline,
    pub gpu_buffers: GpuBuffers,
    pub uniform_bind_group_layout: BindGroupLayout,
    pub paint_bind_group_layout: BindGroupLayout,
    pub tessellate_bind_group_layout: BindGroupLayout,
    pub clear_color: wgpu::Color,
}

impl RenderCanvas {
    pub fn new(ctx: &RenderContext) -> Self {
        let path_pipeline = pipeline::create_path_pipeline(&ctx.device, ctx.format);
        let tessellate_pipeline = pipeline::create_tessellate_pipeline(&ctx.device);
        let gpu_buffers = GpuBuffers::new(&ctx.device);
        let uniform_bind_group_layout = pipeline::create_uniform_bind_group_layout(&ctx.device);
        let paint_bind_group_layout = pipeline::create_paint_bind_group_layout(&ctx.device);
        let tessellate_bind_group_layout = pipeline::create_tessellate_bind_group_layout(&ctx.device);

        Self {
            path_pipeline,
            tessellate_pipeline,
            gpu_buffers,
            uniform_bind_group_layout,
            paint_bind_group_layout,
            tessellate_bind_group_layout,
            clear_color: wgpu::Color {
                r: 0.1,
                g: 0.1,
                b: 0.12,
                a: 1.0,
            },
        }
    }

    /// Render a frame using Rive-style GPU tessellation.
    pub fn render(
        &mut self,
        ctx: &RenderContext,
        draws: &[crate::scene::layer::ShapeDrawCommand],
        world_transform: &Mat2D,
        time: f32,
    ) -> Result<(), SurfaceError> {
        let surface = ctx.surface.as_ref().expect("No surface");
        let output = surface.get_current_texture()?;
        let view = output.texture.create_view(&TextureViewDescriptor::default());

        // Update frame uniforms
        let uniforms = FrameUniforms {
            view_proj: ortho_projection(ctx.width as f32, ctx.height as f32),
            resolution: [ctx.width as f32, ctx.height as f32],
            time,
            _pad: 0.0,
        };
        self.gpu_buffers.update_uniforms(&ctx.queue, &uniforms);

        // Phase 1: CPU — encode paths into GPU tessellation data + paints
        let encoded = encode_paths(draws, world_transform);

        // Diagnostic: log stats on first frame with content
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.load(std::sync::atomic::Ordering::Relaxed) {
            LOGGED.store(true, std::sync::atomic::Ordering::Relaxed);
            log::info!(
                "GPU tess: {} draw cmds → {} segments, {} vertices, {} indices, {} paints",
                draws.len(), encoded.segments.len(), encoded.total_vertices,
                encoded.indices.len(), encoded.paints.len()
            );
            for (i, d) in draws.iter().enumerate().take(10) {
                let paint_desc = match &d.paint {
                    crate::scene::layer::ShapePaint::SolidFill { color, opacity, .. } =>
                        format!("fill({:.2},{:.2},{:.2},a={:.2})", color.r, color.g, color.b, opacity),
                    crate::scene::layer::ShapePaint::SolidStroke { color, opacity, width, .. } =>
                        format!("stroke({:.2},{:.2},{:.2},a={:.2},w={:.1})", color.r, color.g, color.b, opacity, width),
                    crate::scene::layer::ShapePaint::GradientFill { gradient_type, .. } =>
                        format!("grad_fill({:?})", gradient_type),
                    crate::scene::layer::ShapePaint::GradientStroke { gradient_type, .. } =>
                        format!("grad_stroke({:?})", gradient_type),
                };
                log::info!("  draw[{}]: {} verbs, {} pts, paint={}", i, d.path.verbs.len(), d.path.points.len(), paint_desc);
            }
            if draws.len() > 10 { log::info!("  ... and {} more", draws.len() - 10); }
        }

        self.gpu_buffers.upload_encoded_paths(&ctx.device, &ctx.queue, &encoded);

        let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("render encoder"),
        });

        // Phase 2: GPU compute — tessellate cubics into vertices
        if self.gpu_buffers.segment_count > 0 {
            if let Some(tess_bind_group) = self.gpu_buffers.create_tessellate_bind_group(
                &ctx.device,
                &self.tessellate_bind_group_layout,
            ) {
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("tessellate pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.tessellate_pipeline);
                compute_pass.set_bind_group(0, &tess_bind_group, &[]);

                // Dispatch: one thread per segment, workgroup size 64
                let workgroups = (self.gpu_buffers.segment_count + 63) / 64;
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        // Phase 3: GPU draw — render tessellated vertices (MSAA → resolve)
        {
            let (color_view, resolve) = if let Some(msaa_view) = &ctx.msaa_view {
                (msaa_view, Some(&view))
            } else {
                (&view, None)
            };
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("main render pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: resolve,
                    ops: Operations {
                        load: LoadOp::Clear(self.clear_color),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if self.gpu_buffers.index_count > 0 {
                let uniform_bind_group = self.gpu_buffers.create_uniform_bind_group(
                    &ctx.device,
                    &self.uniform_bind_group_layout,
                );

                render_pass.set_pipeline(&self.path_pipeline);
                render_pass.set_bind_group(0, &uniform_bind_group, &[]);

                // Set paint bind group (group 1)
                if let Some(paint_bind_group) = self.gpu_buffers.create_paint_bind_group(
                    &ctx.device,
                    &self.paint_bind_group_layout,
                ) {
                    render_pass.set_bind_group(1, &paint_bind_group, &[]);
                }

                if let Some(vb) = &self.gpu_buffers.vertex_buffer {
                    render_pass.set_vertex_buffer(0, vb.slice(..));
                }
                if let Some(ib) = &self.gpu_buffers.index_buffer {
                    render_pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                }
                render_pass.draw_indexed(0..self.gpu_buffers.index_count, 0, 0..1);
            }
        }

        ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
