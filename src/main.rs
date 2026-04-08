#![allow(dead_code)]

mod geometry;
mod lottie;
mod scene;
mod renderer;
mod engine;

use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId, WindowAttributes};

use geometry::math::Mat2D;
use engine::UltraCanvas;
use renderer::context::RenderContext;
use renderer::canvas::RenderCanvas;

struct App {
    window: Option<Arc<Window>>,
    render_ctx: Option<RenderContext>,
    render_canvas: Option<RenderCanvas>,
    ultra_canvas: Option<UltraCanvas>,
    animation_path: String,
    last_frame: Instant,
    frame_count: u64,
    fps_timer: Instant,
}

impl App {
    fn new(animation_path: String) -> Self {
        Self {
            window: None,
            render_ctx: None,
            render_canvas: None,
            ultra_canvas: None,
            animation_path,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("UltraRender - Lottie WebGPU")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        // Initialize GPU
        let render_ctx = pollster::block_on(RenderContext::new(window.clone()));
        let render_canvas = RenderCanvas::new(&render_ctx);

        let width = render_ctx.width as f32;
        let height = render_ctx.height as f32;

        // Create canvas and load animation
        let mut ultra_canvas = UltraCanvas::new(width, height);

        match ultra_canvas.add_animation(&self.animation_path) {
            Ok(sprite_id) => {
                // Center and scale the sprite to fit the window
                if let Some(sprite) = ultra_canvas.get_sprite_mut(sprite_id) {
                    log::info!("Artboard: {}x{}, {} nodes",
                        sprite.artboard.width, sprite.artboard.height,
                        sprite.artboard.nodes.len());
                    let comp_w = sprite.composition_width;
                    let comp_h = sprite.composition_height;
                    let scale = (width / comp_w).min(height / comp_h) * 0.8;
                    sprite.scale = geometry::math::Vec2D::new(scale, scale);
                    sprite.position = geometry::math::Vec2D::new(
                        (width - comp_w * scale) * 0.5,
                        (height - comp_h * scale) * 0.5,
                    );
                }
                log::info!("Loaded animation: {}", self.animation_path);
            }
            Err(e) => {
                log::error!("Failed to load animation: {}", e);
            }
        }

        self.render_ctx = Some(render_ctx);
        self.render_canvas = Some(render_canvas);
        self.ultra_canvas = Some(ultra_canvas);
        self.last_frame = Instant::now();
        self.fps_timer = Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(ctx) = &mut self.render_ctx {
                    ctx.resize(size.width, size.height);
                }
                if let Some(canvas) = &mut self.ultra_canvas {
                    canvas.resize(size.width as f32, size.height as f32);
                }
                // Re-center sprites on resize
                if let (Some(canvas), Some(ctx)) = (&mut self.ultra_canvas, &self.render_ctx) {
                    let width = ctx.width as f32;
                    let height = ctx.height as f32;
                    for sprite in &mut canvas.sprites {
                        let comp_w = sprite.composition_width;
                        let comp_h = sprite.composition_height;
                        let scale = (width / comp_w).min(height / comp_h) * 0.8;
                        sprite.scale = geometry::math::Vec2D::new(scale, scale);
                        sprite.position = geometry::math::Vec2D::new(
                            (width - comp_w * scale) * 0.5,
                            (height - comp_h * scale) * 0.5,
                        );
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Update animations
                if let Some(canvas) = &mut self.ultra_canvas {
                    canvas.update(dt);
                }

                // Render
                // Collect draws first
                let all_draws = if let Some(ultra_canvas) = &mut self.ultra_canvas {
                    ultra_canvas.collect_all_draws()
                } else {
                    Vec::new()
                };

                if let (Some(ctx), Some(render_canvas)) = (
                    &self.render_ctx,
                    &mut self.render_canvas,
                ) {
                    let time = now.duration_since(self.fps_timer).as_secs_f32();

                    if all_draws.is_empty() {
                        let identity = Mat2D::identity();
                        let _ = render_canvas.render(ctx, &[], &identity, time);
                    } else {
                        for (transform, draws) in &all_draws {
                            match render_canvas.render(ctx, draws, transform, time) {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::OutOfMemory) => {
                                    log::error!("Out of GPU memory");
                                    event_loop.exit();
                                    return;
                                }
                                Err(e) => {
                                    log::warn!("Surface error: {:?}", e);
                                }
                            }
                        }
                    }
                }

                // FPS counter
                self.frame_count += 1;
                let fps_elapsed = now.duration_since(self.fps_timer).as_secs_f32();
                if fps_elapsed >= 2.0 {
                    let fps = self.frame_count as f32 / fps_elapsed;
                    if let Some(window) = &self.window {
                        window.set_title(&format!("UltraRender - {:.0} FPS", fps));
                    }
                    self.frame_count = 0;
                    self.fps_timer = now;
                }

                // Request next frame
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let animation_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "animations/json/gfunny.json".to_string()
    };

    log::info!("UltraRender starting with: {}", animation_path);

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::new(animation_path);
    event_loop.run_app(&mut app).unwrap();
}
