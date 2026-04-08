use wgpu::*;
use crate::geometry::math::Mat2D;
use crate::geometry::tessellation::TessellatedMesh;
use super::context::RenderContext;
use super::pipeline;
use super::buffers::{GpuBuffers, FrameUniforms, ortho_projection};
use super::draw::DrawBatcher;

/// Main render canvas — orchestrates GPU rendering
pub struct RenderCanvas {
    pub path_pipeline: RenderPipeline,
    pub gpu_buffers: GpuBuffers,
    pub batcher: DrawBatcher,
    pub uniform_bind_group_layout: BindGroupLayout,
    pub clear_color: wgpu::Color,
}

impl RenderCanvas {
    pub fn new(ctx: &RenderContext) -> Self {
        let path_pipeline = pipeline::create_path_pipeline(&ctx.device, ctx.format);
        let gpu_buffers = GpuBuffers::new(&ctx.device);
        let batcher = DrawBatcher::new();
        let uniform_bind_group_layout = pipeline::create_uniform_bind_group_layout(&ctx.device);

        Self {
            path_pipeline,
            gpu_buffers,
            batcher,
            uniform_bind_group_layout,
            clear_color: wgpu::Color {
                r: 0.1,
                g: 0.1,
                b: 0.12,
                a: 1.0,
            },
        }
    }

    /// Render a frame
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

        // Update uniforms
        let uniforms = FrameUniforms {
            view_proj: ortho_projection(ctx.width as f32, ctx.height as f32),
            resolution: [ctx.width as f32, ctx.height as f32],
            time,
            _pad: 0.0,
        };
        self.gpu_buffers.update_uniforms(&ctx.queue, &uniforms);

        // Batch and tessellate draws
        let batches = self.batcher.batch(draws, world_transform);

        // Merge all solid batches into one mesh for efficiency
        let mut combined_mesh = TessellatedMesh::new();
        for batch in &batches {
            combined_mesh.append(&batch.mesh);
        }

        // Upload to GPU
        self.gpu_buffers.upload_mesh(&ctx.device, &ctx.queue, &combined_mesh);

        // Create bind group
        let uniform_bind_group = self.gpu_buffers.create_uniform_bind_group(
            &ctx.device,
            &self.uniform_bind_group_layout,
        );

        // Encode render commands
        let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("render encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("main render pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
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
                render_pass.set_pipeline(&self.path_pipeline);
                render_pass.set_bind_group(0, &uniform_bind_group, &[]);

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
