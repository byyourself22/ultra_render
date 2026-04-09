pub mod model;
pub mod parser;
pub mod interpolator;
pub mod property;
pub mod modifiers;

#[cfg(not(target_arch = "wasm32"))]
pub use parser::load_animation;
pub use parser::parse_lottie;
