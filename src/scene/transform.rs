use crate::geometry::math::{Mat2D, Vec2D};
use crate::lottie::model::LottieTransform;
use crate::lottie::property::{eval_f32, eval_vec2d, evaluate_vec2d_spatial};

/// Computed transform at a specific frame
#[derive(Clone, Copy, Debug)]
pub struct ComputedTransform {
    pub local: Mat2D,
    pub world: Mat2D,
    pub opacity: f32,
}

impl Default for ComputedTransform {
    fn default() -> Self {
        Self {
            local: Mat2D::identity(),
            world: Mat2D::identity(),
            opacity: 1.0,
        }
    }
}

/// Evaluate a LottieTransform at a given frame into a Mat2D
pub fn evaluate_transform(transform: &LottieTransform, frame: f32) -> (Mat2D, f32) {
    let anchor = eval_vec2d(&transform.anchor, frame);

    let position = if let (Some(px), Some(py)) = (&transform.position_x, &transform.position_y) {
        Vec2D::new(eval_f32(px, frame), eval_f32(py, frame))
    } else {
        evaluate_vec2d_spatial(&transform.position, frame)
    };

    let scale = eval_vec2d(&transform.scale, frame);
    let rotation = eval_f32(&transform.rotation, frame);
    let opacity = eval_f32(&transform.opacity, frame) / 100.0;
    let skew = eval_f32(&transform.skew, frame);
    let skew_axis = eval_f32(&transform.skew_axis, frame);

    let rot_rad = rotation * std::f32::consts::PI / 180.0;
    let sx = scale.x / 100.0;
    let sy = scale.y / 100.0;

    // Build transform: translate(position) * rotate * skew * scale * translate(-anchor)
    let mut mat = Mat2D::from_translate(position.x, position.y);

    if rot_rad.abs() > 1e-6 {
        mat = Mat2D::multiply(&mat, &Mat2D::from_rotation(rot_rad));
    }

    if skew.abs() > 1e-6 {
        let skew_rad = skew * std::f32::consts::PI / 180.0;
        let axis_rad = skew_axis * std::f32::consts::PI / 180.0;
        let skew_mat = build_skew_matrix(skew_rad, axis_rad);
        mat = Mat2D::multiply(&mat, &skew_mat);
    }

    mat = Mat2D::multiply(&mat, &Mat2D::from_scale(sx, sy));
    mat = Mat2D::multiply(&mat, &Mat2D::from_translate(-anchor.x, -anchor.y));

    (mat, opacity)
}

fn build_skew_matrix(skew: f32, axis: f32) -> Mat2D {
    let cos_a = axis.cos();
    let sin_a = axis.sin();
    let tan_s = skew.tan();

    // Rotate to axis, apply skew, rotate back
    let rotate_to = Mat2D {
        values: [cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0],
    };
    let skew_m = Mat2D {
        values: [1.0, 0.0, tan_s, 1.0, 0.0, 0.0],
    };
    let rotate_back = Mat2D {
        values: [cos_a, -sin_a, sin_a, cos_a, 0.0, 0.0],
    };

    let temp = Mat2D::multiply(&rotate_to, &skew_m);
    Mat2D::multiply(&temp, &rotate_back)
}
