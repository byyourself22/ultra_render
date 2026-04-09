use wgpu::*;
use crate::geometry::math::Mat2D;
use crate::geometry::tessellation::encode_paths;
use super::context::RenderContext;
use super::pipeline;
use super::buffers::{GpuBuffers, FrameUniforms, ortho_projection};

/// Main render canvas — orchestrates GPU tessellation + rendering.
///
/// Pipeline:
///   1. CPU fills:   lyon tessellate → vertex buffer[0..fill_count]
///   2. CPU strokes: encode cubic segments
///   3. GPU compute: tessellate stroke cubics → vertex_buffer[fill_count..]
///   4. GPU draw:    render all vertices (MSAA)
pub struct RenderCanvas {
    pub path_pipeline: RenderPipeline,
    pub tessellate_pipeline: ComputePipeline,
    pub gpu_buffers: GpuBuffers,
    pub uniform_bind_group_layout: BindGroupLayout,
    pub paint_bind_group_layout: BindGroupLayout,
    pub tessellate_bind_group_layout: BindGroupLayout,
    pub clear_color: wgpu::Color,
    needs_tessellation: bool,
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
            clear_color: wgpu::Color { r: 0.1, g: 0.1, b: 0.12, a: 1.0 },
            needs_tessellation: false,
        }
    }

    /// Render all draw commands in a single frame.
    ///
    /// `world_transform` is applied on top of every draw command's existing transform
    /// (pass `Mat2D::identity()` if paths are already in screen space).
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

        // Frame uniforms
        let uniforms = FrameUniforms {
            view_proj: ortho_projection(ctx.width as f32, ctx.height as f32),
            resolution: [ctx.width as f32, ctx.height as f32],
            time,
            is_srgb: ctx.is_srgb as u32,
        };
        self.gpu_buffers.update_uniforms(&ctx.queue, &uniforms);

        // Encode paths: fills (lyon CPU) + strokes (compute GPU)
        let encoded = encode_paths(draws, world_transform);

        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.load(std::sync::atomic::Ordering::Relaxed) && encoded.total_vertex_count() > 0 {
            LOGGED.store(true, std::sync::atomic::Ordering::Relaxed);
            log::info!(
                "GPU pipeline: {} draws → {} fill_verts, {} fill_idx, {} stroke_segs, {} stroke_idx, {} paints",
                draws.len(),
                encoded.fill_vertices.len(),
                encoded.fill_indices.len(),
                encoded.segments.len(),
                encoded.stroke_indices.len(),
                encoded.paints.len(),
            );
        }

        self.gpu_buffers.upload_encoded_paths(&ctx.device, &ctx.queue, &encoded);
        self.needs_tessellation = true;

        let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("render encoder"),
        });

        // Compute pass: tessellate stroke cubics (only if strokes present)
        if self.needs_tessellation && self.gpu_buffers.segment_count > 0 {
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
                let workgroups = (self.gpu_buffers.segment_count + 63) / 64;
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            }
            self.needs_tessellation = false;
        }

        // Draw pass
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

                if let Some(paint_bg) = self.gpu_buffers.create_paint_bind_group(
                    &ctx.device,
                    &self.paint_bind_group_layout,
                ) {
                    render_pass.set_bind_group(1, &paint_bg, &[]);
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
