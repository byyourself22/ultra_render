//! Shared application struct for both native and web targets.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes, WindowId};

use crate::engine::UltraCanvas;
use crate::geometry::math::Vec2D;
use crate::renderer::canvas::RenderCanvas;
use crate::renderer::context::RenderContext;

// ─── Global render stats ─────────────────────────────────────

pub static STAT_FPS: AtomicU32 = AtomicU32::new(0);
pub static STAT_DRAWS: AtomicU32 = AtomicU32::new(0);
pub static STAT_SPRITES: AtomicU32 = AtomicU32::new(0);
pub static STAT_ANIM_FPS: AtomicU32 = AtomicU32::new(0); // animation frame rate × 10
pub static STAT_ANIM_FRAME: AtomicU32 = AtomicU32::new(0); // current frame × 10
pub static STAT_ANIM_FRAMES: AtomicU32 = AtomicU32::new(0); // total frames × 10
pub static STAT_SUBFRAMES: AtomicU32 = AtomicU32::new(0); // render frames per anim frame × 10
pub static STAT_TESS_UNIQUE: AtomicU32 = AtomicU32::new(0); // unique tessellations this frame
pub static TARGET_SPRITES: AtomicU32 = AtomicU32::new(1);
/// When true, all spawned sprites share animation time (offset = 0 → potential batching).
pub static BATCH_SYNCED: AtomicBool = AtomicBool::new(false);

pub struct App {
    window: Option<Arc<Window>>,
    render_ctx: Option<RenderContext>,
    render_canvas: Option<RenderCanvas>,
    ultra_canvas: Option<UltraCanvas>,
    animation_json: Option<String>,
    animation_path: Option<String>,
    last_time: f64,
    start_time: f64,
    frame_count: u64,
    fps_timer: f64,
    fps_frame_count: u32,
    smoothed_fps: f32,
    last_batch_synced: bool,
}

impl App {
    pub fn new(animation_path: String) -> Self {
        let now = now_seconds();
        Self {
            window: None,
            render_ctx: None,
            render_canvas: None,
            ultra_canvas: None,
            animation_json: None,
            animation_path: Some(animation_path),
            last_time: now,
            start_time: now,
            frame_count: 0,
            fps_timer: now,
            fps_frame_count: 0,
            smoothed_fps: 0.0,
            last_batch_synced: BATCH_SYNCED.load(Ordering::Relaxed),
        }
    }

    #[cfg(feature = "web")]
    pub fn new_web() -> Self {
        let now = now_seconds();
        Self {
            window: None,
            render_ctx: None,
            render_canvas: None,
            ultra_canvas: None,
            animation_json: None,
            animation_path: None,
            last_time: now,
            start_time: now,
            frame_count: 0,
            fps_timer: now,
            fps_frame_count: 0,
            smoothed_fps: 0.0,
            last_batch_synced: BATCH_SYNCED.load(Ordering::Relaxed),
        }
    }

    fn init_with_animation(&mut self, width: f32, height: f32) {
        let result: Result<u64, String> = if let Some(ref json) = self.animation_json {
            let json = json.clone();
            self.ultra_canvas
                .as_mut()
                .unwrap()
                .add_animation_from_json(&json, "web_animation")
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let path = self.animation_path.clone();
                if let Some(path) = path {
                    self.ultra_canvas.as_mut().unwrap().add_animation(&path)
                } else {
                    Err("No animation source".into())
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                Err("No animation source (use JSON on web)".into())
            }
        };

        let canvas = self.ultra_canvas.as_mut().unwrap();
        match result {
            Ok(sprite_id) => {
                if let Some(sprite) = canvas.get_sprite_mut(sprite_id) {
                    log::info!(
                        "Artboard: {}x{}, {} nodes",
                        sprite.artboard.width,
                        sprite.artboard.height,
                        sprite.artboard.nodes.len()
                    );

                    // Publish animation stats
                    let fr = sprite.artboard.player.frame_rate;
                    let ip = sprite.artboard.player.in_point;
                    let op = sprite.artboard.player.out_point;
                    STAT_ANIM_FPS.store((fr * 10.0) as u32, Ordering::Relaxed);
                    STAT_ANIM_FRAMES.store(((op - ip) * 10.0) as u32, Ordering::Relaxed);

                    let comp_w = sprite.composition_width;
                    let comp_h = sprite.composition_height;
                    let scale = (width / comp_w).min(height / comp_h) * 0.8;
                    sprite.scale = Vec2D::new(scale, scale);
                    sprite.position = Vec2D::new(
                        (width - comp_w * scale) * 0.5,
                        (height - comp_h * scale) * 0.5,
                    );
                }
                log::info!("Loaded animation successfully");
            }
            Err(e) => log::error!("Failed to load animation: {}", e),
        }
    }

    fn sync_batch_mode(&mut self) {
        let synced = BATCH_SYNCED.load(Ordering::Relaxed);
        if synced == self.last_batch_synced {
            return;
        }

        if let Some(canvas) = &mut self.ultra_canvas {
            if let Some((first, rest)) = canvas.sprites.split_first_mut() {
                if synced {
                    for sprite in rest {
                        sprite.artboard.sync_playback_from(&first.artboard);
                    }
                } else {
                    let duration = (first.artboard.player.out_point - first.artboard.player.in_point)
                        .max(1.0);
                    for (idx, sprite) in rest.iter_mut().enumerate() {
                        let offset = ((idx + 1) as f32 * 0.3) % duration;
                        sprite
                            .artboard
                            .seek_frame(first.artboard.player.in_point + offset);
                        sprite.artboard.play();
                    }
                }
            }
        }

        self.last_batch_synced = synced;
    }

    fn sync_sprite_count(&mut self) {
        let canvas = match self.ultra_canvas.as_mut() {
            Some(c) => c,
            None => return,
        };
        let target = TARGET_SPRITES.load(Ordering::Relaxed).max(1) as usize;
        let current = canvas.sprite_count();

        if current == target || current == 0 {
            return;
        }

        let w = canvas.viewport_width;
        let h = canvas.viewport_height;
        let synced = BATCH_SYNCED.load(Ordering::Relaxed);

        if target > current {
            if let Some(first) = canvas.sprites.first().cloned() {
                let to_add = target - current;
                for k in 0..to_add {
                    let idx = current + k;
                    let cols = (target as f32).sqrt().ceil() as usize;
                    let row = idx / cols;
                    let col = idx % cols;
                    let cell_w = w / cols as f32;
                    let cell_h = h / ((target + cols - 1) / cols) as f32;
                    let scale = (cell_w / first.composition_width)
                        .min(cell_h / first.composition_height)
                        * 0.8;
                    let x = col as f32 * cell_w + (cell_w - first.composition_width * scale) * 0.5;
                    let y = row as f32 * cell_h + (cell_h - first.composition_height * scale) * 0.5;

                    let mut new_sprite = first.artboard.clone();
                    if synced {
                        // Keep animation time identical → same geometry → batchable
                        new_sprite.player.seek(first.artboard.player.frame);
                    } else {
                        // Offset each sprite so they animate at different phases
                        let dur = new_sprite.player.out_point - new_sprite.player.in_point;
                        let offset = (idx as f32 * 0.3) % dur.max(1.0);
                        new_sprite.player.seek(new_sprite.player.in_point + offset);
                    }
                    new_sprite.player.play();

                    let mut sprite = crate::scene::sprite::Sprite::new(0, new_sprite);
                    sprite.position = Vec2D::new(x, y);
                    sprite.scale = Vec2D::new(scale, scale);
                    sprite.play();
                    canvas.add_sprite(sprite);
                }
                self.layout_sprites_grid(target);
            }
        } else {
            canvas.sprites.truncate(target);
            self.layout_sprites_grid(target);
        }
    }

    fn layout_sprites_grid(&mut self, count: usize) {
        let canvas = match self.ultra_canvas.as_mut() {
            Some(c) => c,
            None => return,
        };
        if count == 0 {
            return;
        }
        let w = canvas.viewport_width;
        let h = canvas.viewport_height;
        let cols = (count as f32).sqrt().ceil() as usize;
        let rows = (count + cols - 1) / cols;
        let cell_w = w / cols as f32;
        let cell_h = h / rows as f32;

        for (i, sprite) in canvas.sprites.iter_mut().enumerate() {
            let row = i / cols;
            let col = i % cols;
            let scale =
                (cell_w / sprite.composition_width).min(cell_h / sprite.composition_height) * 0.8;
            sprite.position = Vec2D::new(
                col as f32 * cell_w + (cell_w - sprite.composition_width * scale) * 0.5,
                row as f32 * cell_h + (cell_h - sprite.composition_height * scale) * 0.5,
            );
            sprite.scale = Vec2D::new(scale, scale);
        }
    }

    fn resize_viewport(&mut self, width: u32, height: u32) {
        if let Some(ctx) = &mut self.render_ctx {
            ctx.resize(width, height);
        }
        if let Some(canvas) = &mut self.ultra_canvas {
            canvas.resize(width as f32, height as f32);
        }

        let sprite_count = self
            .ultra_canvas
            .as_ref()
            .map(|canvas| canvas.sprite_count())
            .unwrap_or(0);
        if sprite_count > 0 {
            self.layout_sprites_grid(sprite_count);
        }
    }

    fn prime_first_frame(&mut self) {
        if let Some(canvas) = &mut self.ultra_canvas {
            canvas.update(0.0, BATCH_SYNCED.load(Ordering::Relaxed));
        }
    }

    fn recover_surface(&mut self) {
        #[cfg(feature = "web")]
        if self.sync_web_canvas_size() {
            return;
        }

        if let Some(window) = &self.window {
            let size = window.inner_size();
            self.resize_viewport(size.width.max(1), size.height.max(1));
        }
    }

    #[cfg(feature = "web")]
    fn sync_web_canvas_size(&mut self) -> bool {
        use winit::platform::web::WindowExtWebSys;

        let Some(window) = &self.window else {
            return false;
        };
        let Some(canvas) = window.canvas() else {
            return false;
        };

        let dpr = web_sys::window()
            .map(|win| win.device_pixel_ratio())
            .unwrap_or(1.0);
        let css_width = canvas.client_width().max(0) as f64;
        let css_height = canvas.client_height().max(0) as f64;
        let width = ((if css_width > 0.0 {
            css_width * dpr
        } else {
            canvas.width() as f64
        })
        .round() as u32)
            .max(1);
        let height = ((if css_height > 0.0 {
            css_height * dpr
        } else {
            canvas.height() as f64
        })
        .round() as u32)
            .max(1);

        if canvas.width() != width {
            canvas.set_width(width);
        }
        if canvas.height() != height {
            canvas.set_height(height);
        }

        let needs_resize = self
            .render_ctx
            .as_ref()
            .map(|ctx| ctx.width != width || ctx.height != height)
            .unwrap_or(false);
        if needs_resize {
            log::info!("Syncing web canvas to {}x{}", width, height);
            self.resize_viewport(width, height);
        }

        needs_resize
    }
}

// ─── Platform time ──────────────────────────────────────────

#[cfg(not(feature = "web"))]
fn now_seconds() -> f64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(feature = "web")]
fn now_seconds() -> f64 {
    js_sys::Date::now() / 1000.0
}

// ─── Event handler ──────────────────────────────────────────

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let mut attrs = WindowAttributes::default().with_title("UltraRender - Lottie WebGPU");

        #[cfg(not(feature = "web"))]
        {
            attrs = attrs.with_inner_size(winit::dpi::LogicalSize::new(800, 600));
        }

        #[cfg(feature = "web")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            // Set canvas pixel dimensions from the browser window BEFORE creating the
            // wgpu surface, so inner_size() returns the correct physical pixels.
            if let Some(win) = web_sys::window() {
                let w = win
                    .inner_width()
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(800.0) as u32;
                let h = win
                    .inner_height()
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(600.0) as u32;
                if let Some(canvas_el) = win
                    .document()
                    .and_then(|d| d.get_element_by_id("ultra-canvas"))
                    .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                {
                    canvas_el.set_width(w);
                    canvas_el.set_height(h);
                }
            }

            let canvas = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("ultra-canvas"))
                .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok());
            if let Some(c) = canvas {
                attrs = attrs.with_canvas(Some(c));
            }
        }

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        #[cfg(feature = "web")]
        {
            use winit::platform::web::WindowExtWebSys;
            if let Some(canvas) = window.canvas() {
                let _ = canvas.style().set_property("width", "100%");
                let _ = canvas.style().set_property("height", "100vh");
            }
        }

        // GPU init
        #[cfg(not(feature = "web"))]
        {
            let render_ctx = pollster::block_on(RenderContext::new(window.clone()));
            let render_canvas = RenderCanvas::new(&render_ctx);
            let w = render_ctx.width as f32;
            let h = render_ctx.height as f32;
            self.ultra_canvas = Some(UltraCanvas::new(w, h));
            self.render_ctx = Some(render_ctx);
            self.render_canvas = Some(render_canvas);
            self.init_with_animation(w, h);
            self.prime_first_frame();
        }

        #[cfg(feature = "web")]
        {
            let app_ptr = self as *mut App;
            let win = window.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let render_ctx = RenderContext::new(win).await;
                let render_canvas = RenderCanvas::new(&render_ctx);
                let initial_w = render_ctx.width as f32;
                let initial_h = render_ctx.height as f32;

                let app = unsafe { &mut *app_ptr };
                app.ultra_canvas = Some(UltraCanvas::new(initial_w, initial_h));
                app.render_ctx = Some(render_ctx);
                app.render_canvas = Some(render_canvas);
                app.sync_web_canvas_size();

                match fetch_animation("/animations/json/gfunny.json").await {
                    Ok(json) => {
                        app.animation_json = Some(json);
                        let (w, h) = app
                            .render_ctx
                            .as_ref()
                            .map(|ctx| (ctx.width as f32, ctx.height as f32))
                            .unwrap_or((initial_w, initial_h));
                        app.init_with_animation(w, h);
                        app.prime_first_frame();
                        // Force a redraw now that animation is loaded
                        if let Some(w) = &app.window {
                            w.request_redraw();
                        }
                    }
                    Err(e) => log::error!("Failed to fetch animation: {:?}", e),
                }
            });
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.resize_viewport(size.width, size.height);
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                #[cfg(feature = "web")]
                self.sync_web_canvas_size();

                let now = now_seconds();
                let raw_dt = ((now - self.last_time) as f32).max(0.0);
                self.last_time = now;

                let sample_dt = raw_dt.clamp(1.0 / 1000.0, 0.1);
                let instant_fps = 1.0 / sample_dt;
                if self.smoothed_fps <= 0.0 {
                    self.smoothed_fps = instant_fps;
                } else {
                    let alpha = (sample_dt * 12.0).clamp(0.05, 0.25);
                    self.smoothed_fps += (instant_fps - self.smoothed_fps) * alpha;
                }
                STAT_FPS.store((self.smoothed_fps * 10.0) as u32, Ordering::Relaxed);

                // Cap dt to avoid animation jumps after async init or tab unfocus.
                let dt = raw_dt.min(0.1);

                self.sync_sprite_count();
                self.sync_batch_mode();

                if let Some(canvas) = &mut self.ultra_canvas {
                    canvas.update(dt, BATCH_SYNCED.load(Ordering::Relaxed));
                }

                // Publish current animation frame + subframe stats.
                if let Some(canvas) = &self.ultra_canvas {
                    if let Some(first) = canvas.sprites.first() {
                        let frame = first.artboard.player.frame;
                        let ip = first.artboard.player.in_point;
                        STAT_ANIM_FRAME.store(((frame - ip) * 10.0) as u32, Ordering::Relaxed);
                    }
                }
                {
                    let anim_fps = STAT_ANIM_FPS.load(Ordering::Relaxed) as f32 / 10.0;
                    if anim_fps > 0.1 {
                        let subframes = self.smoothed_fps / anim_fps;
                        STAT_SUBFRAMES.store((subframes * 10.0) as u32, Ordering::Relaxed);
                    } else {
                        STAT_SUBFRAMES.store(0, Ordering::Relaxed);
                    }
                }

                // Collect LOCAL-space draws grouped by sprite + per-sprite transforms.
                let (sprite_groups, transforms, batch_synced, geometry_stamp) =
                    if let Some(uc) = &mut self.ultra_canvas {
                        uc.collect_draw_groups()
                    } else {
                        (Vec::new(), Vec::new(), false, 0)
                    };

                if let Some(uc) = &self.ultra_canvas {
                    STAT_SPRITES.store(uc.sprite_count() as u32, Ordering::Relaxed);
                }
                let unique_tess = if batch_synced && !transforms.is_empty() {
                    1u32
                } else {
                    sprite_groups.len() as u32
                };
                STAT_TESS_UNIQUE.store(unique_tess, Ordering::Relaxed);

                let mut gpu_draw_calls = 0u32;
                let mut recover_surface = false;
                if let (Some(ctx), Some(rc)) = (&self.render_ctx, &mut self.render_canvas) {
                    let time = (now - self.start_time) as f32;
                    match rc.render(
                        ctx,
                        &sprite_groups,
                        &transforms,
                        batch_synced,
                        geometry_stamp,
                        transforms.len() as u32,
                        time,
                    ) {
                        Ok(_) => {
                            gpu_draw_calls = rc.gpu_draw_call_count();
                            STAT_DRAWS.store(gpu_draw_calls, Ordering::Relaxed);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("Out of GPU memory");
                            event_loop.exit();
                            return;
                        }
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            recover_surface = true;
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            log::warn!("Surface timeout");
                        }
                        Err(wgpu::SurfaceError::Other) => {
                            log::warn!("Surface error: Other");
                        }
                    }
                }
                if recover_surface {
                    log::warn!("Surface lost/outdated; reconfiguring");
                    self.recover_surface();
                }

                self.fps_frame_count += 1;
                self.frame_count += 1;
                let elapsed = (now - self.fps_timer) as f32;
                if elapsed >= 1.0 {
                    let fps = self.fps_frame_count as f32 / elapsed;
                    if let Some(w) = &self.window {
                        w.set_title(&format!(
                            "UltraRender - {:.0} FPS | {} draws | {} sprites",
                            self.smoothed_fps.max(fps),
                            gpu_draw_calls,
                            self.ultra_canvas
                                .as_ref()
                                .map(|c| c.sprite_count())
                                .unwrap_or(0)
                        ));
                    }
                    self.fps_frame_count = 0;
                    self.fps_timer = now;
                }

            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

#[cfg(feature = "web")]
async fn fetch_animation(url: &str) -> Result<String, wasm_bindgen::JsValue> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_str(url)).await?;
    let resp: web_sys::Response = resp_value.dyn_into()?;
    let text = JsFuture::from(resp.text()?).await?;
    Ok(text.as_string().unwrap_or_default())
}


