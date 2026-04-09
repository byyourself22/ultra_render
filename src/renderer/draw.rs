//! Rive-style draw types and sort-key builder.
//!
//! The old CPU tessellation `DrawBatcher` has been removed. Path tessellation
//! now happens on the GPU via compute shader (see `geometry::tessellation`
//! for the encoder and `shaders/tessellate_compute.wgsl` for the shader).

use crate::lottie::model::BlendMode;

// ─── Draw type (matches Rive DrawType) ──────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawType {
    MidpointFanFill,
    StrokeStrip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawContents {
    Opaque,
    Translucent,
}

// ─── Sort key builder (Rive-style 63-bit sort key) ──────────

pub fn build_sort_key(
    blend_mode: BlendMode,
    draw_contents: DrawContents,
    z_index: u32,
    draw_type: DrawType,
) -> u64 {
    let bm = (blend_mode as u64) & 0xF;
    let dc = (draw_contents as u64) & 0x3;
    let zi = (z_index as u64) & 0xFFFF;
    let dt = (draw_type as u64) & 0x7;

    (bm << 21) | (dc << 19) | (zi << 3) | dt
}
