#[cfg(not(feature = "web"))]
fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let animation_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "animations/json/gfunny.json".to_string()
    };

    log::info!("UltraRender starting with: {}", animation_path);

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = ultra_render::app::App::new(animation_path);
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(feature = "web")]
fn main() {
    // Web entry is handled by lib.rs #[wasm_bindgen(start)]
}
