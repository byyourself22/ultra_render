pub mod interpolator;
pub mod model;
pub mod modifiers;
pub mod parser;
pub mod property;

#[cfg(not(target_arch = "wasm32"))]
pub use parser::load_animation;
pub use parser::parse_lottie;
