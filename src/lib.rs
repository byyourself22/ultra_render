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

    /// Animation frame rate (fps of the Lottie file).
    #[wasm_bindgen]
    pub fn get_anim_fps() -> f32 {
        crate::app::STAT_ANIM_FPS.load(Ordering::Relaxed) as f32 / 10.0
    }

    /// Current playback frame (relative to in_point).
    #[wasm_bindgen]
    pub fn get_anim_frame() -> f32 {
        crate::app::STAT_ANIM_FRAME.load(Ordering::Relaxed) as f32 / 10.0
    }

    /// Total animation duration in frames.
    #[wasm_bindgen]
    pub fn get_anim_total_frames() -> f32 {
        crate::app::STAT_ANIM_FRAMES.load(Ordering::Relaxed) as f32 / 10.0
    }

    /// Render frames per animation frame (how smooth relative to animation Hz).
    #[wasm_bindgen]
    pub fn get_subframes() -> f32 {
        crate::app::STAT_SUBFRAMES.load(Ordering::Relaxed) as f32 / 10.0
    }

    /// Number of unique tessellations performed this frame (1 for synced sprites).
    #[wasm_bindgen]
    pub fn get_tess_unique() -> u32 {
        crate::app::STAT_TESS_UNIQUE.load(Ordering::Relaxed)
    }

    /// Full stats JSON string including animation info.
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

    // ─── Sprite controls ────────────────────────────────────

    /// Set target sprite count.
    #[wasm_bindgen]
    pub fn request_sprite_count(n: u32) {
        let clamped = n.clamp(1, 10000);
        crate::app::TARGET_SPRITES.store(clamped, Ordering::Relaxed);
    }

    /// Add N sprites (always synced — same frame, single draw call).
    #[wasm_bindgen]
    pub fn add_sprites(n: u32) {
        let current = crate::app::TARGET_SPRITES.load(Ordering::Relaxed);
        let new_count = (current + n).clamp(1, 10000);
        crate::app::TARGET_SPRITES.store(new_count, Ordering::Relaxed);
    }

    // ─── Zoom / Pan ─────────────────────────────────────────

    /// Set camera zoom level (1.0 = default, >1 = closer).
    #[wasm_bindgen]
    pub fn set_zoom(zoom: f32) {
        let z = zoom.clamp(0.1, 20.0);
        crate::app::ZOOM_LEVEL.store(z.to_bits(), Ordering::Relaxed);
    }

    /// Set camera pan offset in world pixels.
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
}
