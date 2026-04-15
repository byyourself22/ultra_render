#![allow(dead_code)]

pub mod app;
pub mod engine;
pub mod geometry;
pub mod lottie;
pub mod renderer;
pub mod scene;

// ─── Web entry point ────────────────────────────────────────

#[cfg(feature = "web")]
pub mod web {
    use std::sync::atomic::Ordering;
    use wasm_bindgen::prelude::*;
    use winit::event_loop::EventLoop;
    use winit::platform::web::EventLoopExtWebSys;

    #[wasm_bindgen(start)]
    pub fn start() {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).ok();
        log::info!("UltraRender WebGPU starting...");

        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

        let app = crate::app::App::new_web();
        event_loop.spawn_app(app);
    }

    // ─── Stats ──────────────────────────────────────────────

    #[wasm_bindgen]
    pub fn get_fps() -> f32 {
        crate::app::STAT_FPS.load(Ordering::Relaxed) as f32 / 10.0
    }

    #[wasm_bindgen]
    pub fn get_draw_calls() -> u32 {
        crate::app::STAT_DRAWS.load(Ordering::Relaxed)
    }

    #[wasm_bindgen]
    pub fn get_sprite_count() -> u32 {
        crate::app::STAT_SPRITES.load(Ordering::Relaxed)
    }

    #[wasm_bindgen]
    pub fn get_anim_fps() -> f32 {
        crate::app::STAT_ANIM_FPS.load(Ordering::Relaxed) as f32 / 10.0
    }

    #[wasm_bindgen]
    pub fn get_anim_frame() -> f32 {
        crate::app::STAT_ANIM_FRAME.load(Ordering::Relaxed) as f32 / 10.0
    }

    #[wasm_bindgen]
    pub fn get_anim_total_frames() -> f32 {
        crate::app::STAT_ANIM_FRAMES.load(Ordering::Relaxed) as f32 / 10.0
    }

    #[wasm_bindgen]
    pub fn get_subframes() -> f32 {
        crate::app::STAT_SUBFRAMES.load(Ordering::Relaxed) as f32 / 10.0
    }

    #[wasm_bindgen]
    pub fn get_tess_unique() -> u32 {
        crate::app::STAT_TESS_UNIQUE.load(Ordering::Relaxed)
    }

    #[wasm_bindgen]
    pub fn get_stats_json() -> String {
        let fps = crate::app::STAT_FPS.load(Ordering::Relaxed) as f32 / 10.0;
        let draws = crate::app::STAT_DRAWS.load(Ordering::Relaxed);
        let sprites = crate::app::STAT_SPRITES.load(Ordering::Relaxed);
        let anim_fps = crate::app::STAT_ANIM_FPS.load(Ordering::Relaxed) as f32 / 10.0;
        let anim_frame = crate::app::STAT_ANIM_FRAME.load(Ordering::Relaxed) as f32 / 10.0;
        let anim_total = crate::app::STAT_ANIM_FRAMES.load(Ordering::Relaxed) as f32 / 10.0;
        let subframes = crate::app::STAT_SUBFRAMES.load(Ordering::Relaxed) as f32 / 10.0;
        let unique_tess = crate::app::STAT_TESS_UNIQUE.load(Ordering::Relaxed);
        format!(
            r#"{{"fps":{:.1},"draws":{},"sprites":{},"animFps":{:.0},"animFrame":{:.1},"animTotal":{:.0},"subframes":{:.1},"uniqueTess":{}}}"#,
            fps, draws, sprites, anim_fps, anim_frame, anim_total, subframes, unique_tess
        )
    }

    // ─── Sprites ────────────────────────────────────────────

    /// Set target sprite count.
    #[wasm_bindgen]
    pub fn request_sprite_count(n: u32) {
        crate::app::TARGET_SPRITES.store(n.clamp(1, 10000), Ordering::Relaxed);
    }

    /// Add N sprites (synced — same frame, single tessellation).
    #[wasm_bindgen]
    pub fn add_sprites(n: u32) {
        let current = crate::app::TARGET_SPRITES.load(Ordering::Relaxed);
        crate::app::TARGET_SPRITES.store((current + n).clamp(1, 10000), Ordering::Relaxed);
    }

    // ─── Playback ───────────────────────────────────────────

    /// Play (resume) the animation.
    #[wasm_bindgen]
    pub fn play() {
        crate::app::PLAYBACK_PAUSED.store(0, Ordering::Relaxed);
    }

    /// Pause the animation.
    #[wasm_bindgen]
    pub fn pause() {
        crate::app::PLAYBACK_PAUSED.store(1, Ordering::Relaxed);
    }

    /// Check if currently paused.
    #[wasm_bindgen]
    pub fn is_paused() -> bool {
        crate::app::PLAYBACK_PAUSED.load(Ordering::Relaxed) != 0
    }

    /// Set playback speed (1.0 = normal, 0.5 = half, 2.0 = double).
    #[wasm_bindgen]
    pub fn set_speed(speed: f32) {
        let s = speed.clamp(0.0, 10.0);
        crate::app::PLAYBACK_SPEED.store(s.to_bits(), Ordering::Relaxed);
    }

    /// Get current playback speed.
    #[wasm_bindgen]
    pub fn get_speed() -> f32 {
        f32::from_bits(crate::app::PLAYBACK_SPEED.load(Ordering::Relaxed))
    }

    // ─── View (zoom / pan / fit) ────────────────────────────

    /// Set zoom level (1.0 = default, >1 = closer).
    /// Zoom is applied as a scale transform on sprites (Rive-style).
    #[wasm_bindgen]
    pub fn set_zoom(zoom: f32) {
        crate::app::ZOOM_LEVEL.store(zoom.clamp(0.1, 20.0).to_bits(), Ordering::Relaxed);
    }

    /// Set pan offset in world pixels.
    #[wasm_bindgen]
    pub fn set_pan(x: f32, y: f32) {
        crate::app::PAN_X.store(x.to_bits(), Ordering::Relaxed);
        crate::app::PAN_Y.store(y.to_bits(), Ordering::Relaxed);
    }

    /// Get current zoom level.
    #[wasm_bindgen]
    pub fn get_zoom() -> f32 {
        f32::from_bits(crate::app::ZOOM_LEVEL.load(Ordering::Relaxed))
    }

    /// Set fit mode: 0=Cover, 1=Contain, 2=Fill, 3=ScaleDown, 4=None.
    #[wasm_bindgen]
    pub fn set_fit(fit: u32) {
        crate::app::FIT_MODE.store(fit.min(4), Ordering::Relaxed);
    }

    /// Set DPI scale factor (typically window.devicePixelRatio).
    #[wasm_bindgen]
    pub fn set_scale_factor(dpr: f32) {
        crate::app::SCALE_FACTOR.store(dpr.max(0.5).to_bits(), Ordering::Relaxed);
    }
}
