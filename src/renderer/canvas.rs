use super::buffers::{ortho_projection, FrameUniforms, GpuBuffers};
use super::context::RenderContext;
use super::pipeline;
use crate::geometry::math::Mat2D;
use crate::geometry::tessellation::{encode_paths_instanced, EncodedDraw, SpriteTransform};
use crate::scene::layer::ShapeDrawCommand;
use wgpu::*;

/// Main render canvas — orchestrates GPU tessellation + rendering.
///
/// Pipeline:
///   1. CPU topology: build fill indices over sampled LOCAL-space contours
///   2. GPU compute:  tessellate fill/stroke cubics into the shared vertex buffer
///   3. GPU draw:     render all vertices with per-sprite transforms from storage buffer
///
/// Synced sprites (same animation frame) share one tessellation — only position
/// offsets differ, handled by the sprite_transforms storage buffer in the vertex shader.
pub struct RenderCanvas {
    pub path_pipeline: RenderPipeline,
    pub complex_stencil_nz_pipeline: RenderPipeline,
    pub complex_stencil_eo_pipeline: RenderPipeline,
    pub complex_cover_pipeline: RenderPipeline,
    pub fill_tessellate_pipeline: ComputePipeline,
    pub fill_topology_pipeline: ComputePipeline,
    pub tessellate_pipeline: ComputePipeline,
    pub gpu_buffers: GpuBuffers,
    pub uniform_bind_group_layout: BindGroupLayout,
    pub paint_bind_group_layout: BindGroupLayout,
    pub fill_tess_bind_group_layout: BindGroupLayout,
    pub tessellate_bind_group_layout: BindGroupLayout,
    pub sprite_transforms_bg_layout: BindGroupLayout,
    pub clear_color: wgpu::Color,
    needs_tessellation: bool,
    /// Geometry stamp for the last encoded submission.
    last_geometry_stamp: u64,
}

impl RenderCanvas {
    pub fn new(ctx: &RenderContext) -> Self {
        let path_pipeline = pipeline::create_path_pipeline(&ctx.device, ctx.format);
        let complex_stencil_nz_pipeline = pipeline::create_complex_stencil_pipeline(
            &ctx.device,
            ctx.format,
            pipeline::ComplexStencilMode::NonZero,
        );
        let complex_stencil_eo_pipeline = pipeline::create_complex_stencil_pipeline(
            &ctx.device,
            ctx.format,
            pipeline::ComplexStencilMode::EvenOdd,
        );
        let complex_cover_pipeline =
            pipeline::create_complex_cover_pipeline(&ctx.device, ctx.format);
        let fill_tessellate_pipeline = pipeline::create_fill_tessellate_pipeline(&ctx.device);
        let fill_topology_pipeline = pipeline::create_fill_topology_pipeline(&ctx.device);
        let tessellate_pipeline = pipeline::create_tessellate_pipeline(&ctx.device);
        let gpu_buffers = GpuBuffers::new(&ctx.device);
        let uniform_bind_group_layout = pipeline::create_uniform_bind_group_layout(&ctx.device);
        let paint_bind_group_layout = pipeline::create_paint_bind_group_layout(&ctx.device);
        let fill_tess_bind_group_layout =
            pipeline::create_fill_tessellate_bind_group_layout(&ctx.device);
        let tessellate_bind_group_layout =
            pipeline::create_tessellate_bind_group_layout(&ctx.device);
        let sprite_transforms_bg_layout =
            pipeline::create_sprite_transforms_bind_group_layout(&ctx.device);

        Self {
            path_pipeline,
            complex_stencil_nz_pipeline,
            complex_stencil_eo_pipeline,
            complex_cover_pipeline,
            fill_tessellate_pipeline,
            fill_topology_pipeline,
            tessellate_pipeline,
            gpu_buffers,
            uniform_bind_group_layout,
            paint_bind_group_layout,
            fill_tess_bind_group_layout,
            tessellate_bind_group_layout,
            sprite_transforms_bg_layout,
            clear_color: wgpu::Color {
                r: 0.1,
                g: 0.1,
                b: 0.12,
                a: 1.0,
            },
            needs_tessellation: false,
            last_geometry_stamp: 0,
        }
    }

    pub fn gpu_draw_call_count(&self) -> u32 {
        self.gpu_buffers
            .ordered_draws
            .iter()
            .map(|draw| match draw {
                EncodedDraw::Simple { .. } => 1,
                EncodedDraw::Complex(_) => 2,
            })
            .sum()
    }

    /// Render a frame.
    ///
    /// `sprite_groups`: one entry per visible sprite — `(sprite_index, local-space draws)`.
    /// `transforms`:    per-sprite world transforms indexed by sprite_index.
    /// `time`:          elapsed seconds (for shader effects).
    pub fn render(
        &mut self,
        ctx: &RenderContext,
        sprite_groups: &[(u32, Vec<ShapeDrawCommand>)],
        transforms: &[Mat2D],
        batch_synced: bool,
        geometry_stamp: u64,
        visible_sprite_count: u32,
        time: f32,
    ) -> Result<(), SurfaceError> {
        let surface = ctx.surface.as_ref().expect("No surface");
        let output = surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        // Frame uniforms
        let uniforms = FrameUniforms {
            view_proj: ortho_projection(ctx.width as f32, ctx.height as f32),
            resolution: [ctx.width as f32, ctx.height as f32],
            time,
            is_srgb: ctx.is_srgb as u32,
        };
        self.gpu_buffers.update_uniforms(&ctx.queue, &uniforms);

        // Upload per-sprite transforms (identity for single-sprite case)
        let sprite_xforms: Vec<SpriteTransform> = if transforms.is_empty() {
            vec![SpriteTransform::identity()]
        } else {
            transforms.iter().map(SpriteTransform::from_mat2d).collect()
        };
        self.gpu_buffers
            .upload_sprite_transforms(&ctx.device, &ctx.queue, &sprite_xforms);

        let geometry_changed = geometry_stamp != self.last_geometry_stamp;
        if geometry_changed {
            self.last_geometry_stamp = geometry_stamp;

            // Encode paths (tessellate; synced sprites share geometry)
            let encoded = encode_paths_instanced(sprite_groups, transforms, batch_synced);

            static LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.load(std::sync::atomic::Ordering::Relaxed)
                && encoded.total_vertex_count() > 0
            {
                LOGGED.store(true, std::sync::atomic::Ordering::Relaxed);
                let n_sprites = sprite_groups.len();
                let draws: usize = sprite_groups.iter().map(|(_, c)| c.len()).sum();
                log::info!(
                    "GPU pipeline: {} sprites, {} total draws -> {} fill_verts, {} fill_idx, {} stroke_segs, {} stroke_idx, {} paints | instance_count={} synced={}",
                    n_sprites, draws,
                    encoded.fill_vertices.len(),
                    encoded.fill_indices.len(),
                    encoded.segments.len(),
                    encoded.stroke_indices.len(),
                    encoded.paints.len(),
                    encoded.instance_count,
                    batch_synced,
                );
            }

            self.gpu_buffers
                .upload_encoded_paths(&ctx.device, &ctx.queue, &encoded);
            self.needs_tessellation = true;
        } else if batch_synced {
            self.gpu_buffers.instance_count = visible_sprite_count.max(1);
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        // Compute passes: fill tessellation first, then fill topology, then
        // strokes, so the shared buffers are fully populated before drawing.
        if self.needs_tessellation {
            if self.gpu_buffers.fill_contour_count > 0 {
                if let Some(fill_bg) = self.gpu_buffers.create_fill_tessellate_bind_group(
                    &ctx.device,
                    &self.fill_tess_bind_group_layout,
                ) {
                    if self.gpu_buffers.fill_cubic_count > 0 {
                        let mut cp = encoder.begin_compute_pass(&ComputePassDescriptor {
                            label: Some("fill tessellate pass"),
                            timestamp_writes: None,
                        });
                        cp.set_pipeline(&self.fill_tessellate_pipeline);
                        cp.set_bind_group(0, &fill_bg, &[]);
                        let workgroups = (self.gpu_buffers.fill_cubic_count + 63) / 64;
                        cp.dispatch_workgroups(workgroups, 1, 1);
                    }

                    if !self.gpu_buffers.complex_fill_draws.is_empty() {
                        let mut cp = encoder.begin_compute_pass(&ComputePassDescriptor {
                            label: Some("fill topology pass"),
                            timestamp_writes: None,
                        });
                        cp.set_pipeline(&self.fill_topology_pipeline);
                        cp.set_bind_group(0, &fill_bg, &[]);
                        cp.dispatch_workgroups(self.gpu_buffers.fill_contour_count, 1, 1);
                    }
                }
            }

            if self.gpu_buffers.segment_count > 0 {
                if let Some(tess_bg) = self
                    .gpu_buffers
                    .create_tessellate_bind_group(&ctx.device, &self.tessellate_bind_group_layout)
                {
                    let mut cp = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("stroke tessellate pass"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(&self.tessellate_pipeline);
                    cp.set_bind_group(0, &tess_bg, &[]);
                    let workgroups = (self.gpu_buffers.segment_count + 63) / 64;
                    cp.dispatch_workgroups(workgroups, 1, 1);
                }
            }
            self.needs_tessellation = false;
        }

        let uniform_bg = self
            .gpu_buffers
            .create_uniform_bind_group(&ctx.device, &self.uniform_bind_group_layout);
        let paint_bg = self
            .gpu_buffers
            .create_paint_bind_group(&ctx.device, &self.paint_bind_group_layout);
        let sprite_bg = self
            .gpu_buffers
            .create_sprite_transforms_bind_group(&ctx.device, &self.sprite_transforms_bg_layout);

        let (color_view, resolve) = if let Some(msaa_view) = &ctx.msaa_view {
            (msaa_view, Some(&view))
        } else {
            (&view, None)
        };
        let depth_stencil_view = ctx
            .depth_stencil_view
            .as_ref()
            .expect("Missing depth/stencil view");

        // Main pass: clear color + stencil once, then replay the encoded draw order
        // so simple and complex fills keep the original layer semantics.
        {
            let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("ordered render pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: resolve,
                    ops: Operations {
                        load: LoadOp::Clear(self.clear_color),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: depth_stencil_view,
                    depth_ops: None,
                    stencil_ops: Some(Operations {
                        load: LoadOp::Clear(0),
                        store: StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rp.set_bind_group(0, &uniform_bg, &[]);
            if let Some(paint_bg) = &paint_bg {
                rp.set_bind_group(1, paint_bg, &[]);
            }
            if let Some(sprite_bg) = &sprite_bg {
                rp.set_bind_group(2, sprite_bg, &[]);
            }
            if let Some(vb) = &self.gpu_buffers.vertex_buffer {
                rp.set_vertex_buffer(0, vb.slice(..));
            }
            if let Some(ib) = &self.gpu_buffers.index_buffer {
                rp.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            }

            for draw in &self.gpu_buffers.ordered_draws {
                match *draw {
                    EncodedDraw::Simple {
                        index_start,
                        index_count,
                        ..
                    } => {
                        if index_count == 0 {
                            continue;
                        }
                        rp.set_pipeline(&self.path_pipeline);
                        rp.draw_indexed(
                            index_start..index_start + index_count,
                            0,
                            0..self.gpu_buffers.instance_count,
                        );
                    }
                    EncodedDraw::Complex(draw) => {
                        let stencil_pipeline = match draw.fill_rule {
                            crate::geometry::tessellation::ComplexFillRule::NonZero => {
                                &self.complex_stencil_nz_pipeline
                            }
                            crate::geometry::tessellation::ComplexFillRule::EvenOdd => {
                                &self.complex_stencil_eo_pipeline
                            }
                        };
                        rp.set_pipeline(stencil_pipeline);
                        rp.draw_indexed(
                            draw.stencil_index_start
                                ..draw.stencil_index_start + draw.stencil_index_count,
                            0,
                            0..self.gpu_buffers.instance_count,
                        );

                        rp.set_pipeline(&self.complex_cover_pipeline);
                        rp.draw_indexed(
                            draw.cover_index_start..draw.cover_index_start + draw.cover_index_count,
                            0,
                            0..self.gpu_buffers.instance_count,
                        );
                    }
                }
            }
        }

        ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

