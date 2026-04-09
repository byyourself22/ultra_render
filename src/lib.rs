#![allow(dead_code)]

pub mod geometry;
pub mod lottie;
pub mod scene;
pub mod renderer;
pub mod engine;
pub mod app;

// ─── Web entry point ────────────────────────────────────────

#[cfg(feature = "web")]
pub mod web {
    use wasm_bindgen::prelude::*;
    use winit::event_loop::EventLoop;
    use winit::platform::web::EventLoopExtWebSys;
    use std::sync::atomic::Ordering;

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

    /// Full stats JSON string including animation info.
    #[wasm_bindgen]
    pub fn get_stats_json() -> String {
        let fps = crate::app::STAT_FPS.load(Ordering::Relaxed) as f32 / 10.0;
        let draws = crate::app::STAT_DRAWS.load(Ordering::Relaxed);
        let sprites = crate::app::STAT_SPRITES.load(Ordering::Relaxed);
        let anim_fps = crate::app::STAT_ANIM_FPS.load(Ordering::Relaxed) as f32 / 10.0;
        let anim_frame = crate::app::STAT_ANIM_FRAME.load(Ordering::Relaxed) as f32 / 10.0;
        let anim_total = crate::app::STAT_ANIM_FRAMES.load(Ordering::Relaxed) as f32 / 10.0;
        let synced = crate::app::BATCH_SYNCED.load(Ordering::Relaxed);
        format!(
            r#"{{"fps":{:.1},"draws":{},"sprites":{},"animFps":{:.0},"animFrame":{:.1},"animTotal":{:.0},"synced":{}}}"#,
            fps, draws, sprites, anim_fps, anim_frame, anim_total, synced
        )
    }

    // ─── Sprite controls ────────────────────────────────────

    /// Set target sprite count (clamped to 1–256).
    #[wasm_bindgen]
    pub fn request_sprite_count(n: u32) {
        let clamped = n.clamp(1, 256);
        crate::app::TARGET_SPRITES.store(clamped, Ordering::Relaxed);
    }

    /// Set synced-batch mode.
    /// When true, new sprites share the same animation frame → identical geometry → batchable.
    /// When false, sprites are offset in time for visual variety.
    #[wasm_bindgen]
    pub fn set_batch_synced(synced: bool) {
        crate::app::BATCH_SYNCED.store(synced, Ordering::Relaxed);
    }

    /// Whether sprites are currently in synced-batch mode.
    #[wasm_bindgen]
    pub fn get_batch_synced() -> bool {
        crate::app::BATCH_SYNCED.load(Ordering::Relaxed)
    }

    /// Add N sprites in synced mode (same animation frame).
    #[wasm_bindgen]
    pub fn add_sprites_batched(n: u32) {
        crate::app::BATCH_SYNCED.store(true, Ordering::Relaxed);
        let current = crate::app::TARGET_SPRITES.load(Ordering::Relaxed);
        let new_count = (current + n).clamp(1, 256);
        crate::app::TARGET_SPRITES.store(new_count, Ordering::Relaxed);
    }

    /// Add N sprites in unsynced mode (offset animation time).
    #[wasm_bindgen]
    pub fn add_sprites_unbatched(n: u32) {
        crate::app::BATCH_SYNCED.store(false, Ordering::Relaxed);
        let current = crate::app::TARGET_SPRITES.load(Ordering::Relaxed);
        let new_count = (current + n).clamp(1, 256);
        crate::app::TARGET_SPRITES.store(new_count, Ordering::Relaxed);
    }
}
