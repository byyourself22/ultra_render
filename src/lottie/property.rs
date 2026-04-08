use crate::geometry::math::{Vec2D, Color};
use super::model::*;
use super::interpolator::{ease_mode_from_handles, ease};

/// Trait for types that can be interpolated between keyframes
pub trait Interpolatable: Clone + std::fmt::Debug {
    fn interp(&self, other: &Self, t: f32) -> Self;
}

impl Interpolatable for f32 {
    fn interp(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

impl Interpolatable for Vec2D {
    fn interp(&self, other: &Self, t: f32) -> Self {
        Vec2D::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )
    }
}

impl Interpolatable for Color {
    fn interp(&self, other: &Self, t: f32) -> Self {
        Color::new(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
            self.a + (other.a - self.a) * t,
        )
    }
}

impl Interpolatable for GradientColors {
    fn interp(&self, other: &Self, t: f32) -> Self {
        let len = self.colors.len().min(other.colors.len());
        let mut colors = Vec::with_capacity(len);
        for i in 0..len {
            colors.push(self.colors[i] + (other.colors[i] - self.colors[i]) * t);
        }
        GradientColors {
            color_count: self.color_count,
            colors,
        }
    }
}

impl Interpolatable for BezierPath {
    fn interp(&self, other: &Self, t: f32) -> Self {
        let vert_len = self.vertices.len().min(other.vertices.len());
        let mut vertices = Vec::with_capacity(vert_len);
        let mut in_tangents = Vec::with_capacity(vert_len);
        let mut out_tangents = Vec::with_capacity(vert_len);

        for i in 0..vert_len {
            vertices.push(Vec2D::new(
                self.vertices[i].x + (other.vertices[i].x - self.vertices[i].x) * t,
                self.vertices[i].y + (other.vertices[i].y - self.vertices[i].y) * t,
            ));
        }

        let in_len = self.in_tangents.len().min(other.in_tangents.len());
        for i in 0..in_len {
            in_tangents.push(Vec2D::new(
                self.in_tangents[i].x + (other.in_tangents[i].x - self.in_tangents[i].x) * t,
                self.in_tangents[i].y + (other.in_tangents[i].y - self.in_tangents[i].y) * t,
            ));
        }

        let out_len = self.out_tangents.len().min(other.out_tangents.len());
        for i in 0..out_len {
            out_tangents.push(Vec2D::new(
                self.out_tangents[i].x + (other.out_tangents[i].x - self.out_tangents[i].x) * t,
                self.out_tangents[i].y + (other.out_tangents[i].y - self.out_tangents[i].y) * t,
            ));
        }

        BezierPath {
            vertices,
            in_tangents,
            out_tangents,
            closed: self.closed,
        }
    }
}

/// Evaluate an AnimatedValue at a given frame
pub fn evaluate<T: Interpolatable + Default>(value: &AnimatedValue<T>, frame: f32) -> T {
    match value {
        AnimatedValue::Static(v) => v.clone(),
        AnimatedValue::Animated(keyframes) => evaluate_keyframes(keyframes, frame),
    }
}

/// Evaluate keyframes at a given frame
fn evaluate_keyframes<T: Interpolatable + Default>(keyframes: &[Keyframe<T>], frame: f32) -> T {
    if keyframes.is_empty() {
        return T::default();
    }

    // Before first keyframe
    if frame <= keyframes[0].time {
        return keyframes[0].value.clone();
    }

    // After last keyframe
    let last = &keyframes[keyframes.len() - 1];
    if frame >= last.time {
        return last.value.clone();
    }

    // Find the keyframe segment
    for i in 0..keyframes.len() - 1 {
        let kf = &keyframes[i];
        let next_kf = &keyframes[i + 1];

        if frame >= kf.time && frame < next_kf.time {
            // Hold keyframe - no interpolation
            if kf.hold {
                return kf.value.clone();
            }

            // Calculate linear progress
            let duration = next_kf.time - kf.time;
            if duration <= 0.0 {
                return kf.value.clone();
            }
            let linear_t = (frame - kf.time) / duration;

            // Apply easing
            let ease_mode = ease_mode_from_handles(&kf.easing_out, &kf.easing_in, false);
            let eased_t = ease(ease_mode, linear_t);

            // Get the end value (either explicit or next keyframe's start)
            let end_value = kf.end_value.as_ref().unwrap_or(&next_kf.value);

            return kf.value.interp(end_value, eased_t);
        }
    }

    // Fallback
    keyframes.last().map(|kf| kf.value.clone()).unwrap_or_default()
}

/// Evaluate a Vec2D animated value with spatial tangent support (motion paths)
pub fn evaluate_vec2d_spatial(value: &AnimatedValue<Vec2D>, frame: f32) -> Vec2D {
    match value {
        AnimatedValue::Static(v) => *v,
        AnimatedValue::Animated(keyframes) => evaluate_vec2d_keyframes_spatial(keyframes, frame),
    }
}

fn evaluate_vec2d_keyframes_spatial(keyframes: &[Keyframe<Vec2D>], frame: f32) -> Vec2D {
    if keyframes.is_empty() {
        return Vec2D::ZERO;
    }

    if frame <= keyframes[0].time {
        return keyframes[0].value;
    }

    let last = &keyframes[keyframes.len() - 1];
    if frame >= last.time {
        return last.value;
    }

    for i in 0..keyframes.len() - 1 {
        let kf = &keyframes[i];
        let next_kf = &keyframes[i + 1];

        if frame >= kf.time && frame < next_kf.time {
            if kf.hold {
                return kf.value;
            }

            let duration = next_kf.time - kf.time;
            if duration <= 0.0 {
                return kf.value;
            }
            let linear_t = (frame - kf.time) / duration;

            let ease_mode = ease_mode_from_handles(&kf.easing_out, &kf.easing_in, false);
            let eased_t = ease(ease_mode, linear_t);

            let end_value = kf.end_value.as_ref().unwrap_or(&next_kf.value);

            // If spatial tangents exist, use cubic bezier interpolation for the motion path
            if let (Some(tan_out), Some(tan_in)) = (&kf.tan_out, &next_kf.tan_in) {
                let p0 = kf.value;
                let p1 = Vec2D::new(p0.x + tan_out.x, p0.y + tan_out.y);
                let p2 = Vec2D::new(end_value.x + tan_in.x, end_value.y + tan_in.y);
                let p3 = *end_value;
                return crate::geometry::math::cubic_eval(p0, p1, p2, p3, eased_t);
            }

            return kf.value.interp(end_value, eased_t);
        }
    }

    keyframes.last().map(|kf| kf.value).unwrap_or(Vec2D::ZERO)
}

/// Convenience: evaluate f32 property
pub fn eval_f32(value: &AnimatedValue<f32>, frame: f32) -> f32 {
    evaluate(value, frame)
}

/// Convenience: evaluate Vec2D property
pub fn eval_vec2d(value: &AnimatedValue<Vec2D>, frame: f32) -> Vec2D {
    evaluate(value, frame)
}

/// Convenience: evaluate Color property
pub fn eval_color(value: &AnimatedValue<Color>, frame: f32) -> Color {
    evaluate(value, frame)
}

/// Convenience: evaluate BezierPath property
pub fn eval_bezier_path(value: &AnimatedValue<BezierPath>, frame: f32) -> BezierPath {
    evaluate(value, frame)
}
