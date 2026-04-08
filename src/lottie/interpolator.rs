/// Cubic bezier easing interpolation (ThorVG tvgLottieInterpolator style)
///
/// Given control points (x1,y1) and (x2,y2) in [0,1], computes the eased
/// value for a linear progress t in [0,1].

/// Evaluate a cubic bezier easing curve.
/// Control points: (0,0) -> (x1,y1) -> (x2,y2) -> (1,1)
/// Input `t` is the time ratio [0,1], returns eased ratio [0,1].
pub fn cubic_bezier_ease(x1: f32, y1: f32, x2: f32, y2: f32, t: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }

    // Linear case
    if (x1 - y1).abs() < 1e-6 && (x2 - y2).abs() < 1e-6 {
        return t;
    }

    // Newton's method to find the parameter s such that bezier_x(s) = t
    let s = find_t_for_x(x1, x2, t);

    // Evaluate y at s
    bezier_component(y1, y2, s)
}

/// Evaluate one component of a cubic bezier: (0) -> (c1) -> (c2) -> (1)
fn bezier_component(c1: f32, c2: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    // B(t) = 3*mt^2*t*c1 + 3*mt*t^2*c2 + t^3
    3.0 * mt2 * t * c1 + 3.0 * mt * t2 * c2 + t3
}

/// Derivative of bezier_component
fn bezier_component_deriv(c1: f32, c2: f32, t: f32) -> f32 {
    let mt = 1.0 - t;
    // B'(t) = 3*mt^2*c1 + 6*mt*t*(c2-c1) + 3*t^2*(1-c2)
    3.0 * mt * mt * c1 + 6.0 * mt * t * (c2 - c1) + 3.0 * t * t * (1.0 - c2)
}

/// Newton-Raphson + bisection fallback to find s where bezier_x(s) = target_x
fn find_t_for_x(x1: f32, x2: f32, target_x: f32) -> f32 {
    // Initial guess
    let mut s = target_x;

    // Newton's method (8 iterations)
    for _ in 0..8 {
        let x = bezier_component(x1, x2, s) - target_x;
        let dx = bezier_component_deriv(x1, x2, s);
        if dx.abs() < 1e-10 {
            break;
        }
        let new_s = s - x / dx;
        if (new_s - s).abs() < 1e-7 {
            return new_s.clamp(0.0, 1.0);
        }
        s = new_s;
    }

    // Bisection fallback if Newton didn't converge well
    let s_clamped = s.clamp(0.0, 1.0);
    let residual = (bezier_component(x1, x2, s_clamped) - target_x).abs();
    if residual < 1e-5 {
        return s_clamped;
    }

    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    s = target_x;
    for _ in 0..20 {
        let x = bezier_component(x1, x2, s);
        if (x - target_x).abs() < 1e-7 {
            return s;
        }
        if x < target_x {
            lo = s;
        } else {
            hi = s;
        }
        s = (lo + hi) * 0.5;
    }
    s
}

/// Interpolation mode for a keyframe segment
#[derive(Clone, Copy, Debug)]
pub enum EaseMode {
    /// Linear interpolation
    Linear,
    /// Hold (step) - no interpolation
    Hold,
    /// Cubic bezier with control points
    CubicBezier { x1: f32, y1: f32, x2: f32, y2: f32 },
}

/// Compute the eased value given an EaseMode and linear progress t
pub fn ease(mode: EaseMode, t: f32) -> f32 {
    match mode {
        EaseMode::Linear => t,
        EaseMode::Hold => 0.0,
        EaseMode::CubicBezier { x1, y1, x2, y2 } => cubic_bezier_ease(x1, y1, x2, y2, t),
    }
}

/// Determine the EaseMode from keyframe easing handles.
/// Uses the first component of in/out handles (multi-dimensional keyframes
/// may have per-component easing, but we use the first for simplicity).
pub fn ease_mode_from_handles(
    easing_out: &super::model::EasingHandle,
    easing_in: &super::model::EasingHandle,
    hold: bool,
) -> EaseMode {
    if hold {
        return EaseMode::Hold;
    }

    let x1 = easing_out.x.first().copied().unwrap_or(0.0);
    let y1 = easing_out.y.first().copied().unwrap_or(0.0);
    let x2 = easing_in.x.first().copied().unwrap_or(1.0);
    let y2 = easing_in.y.first().copied().unwrap_or(1.0);

    // Check if linear
    if (x1 - y1).abs() < 1e-4 && (x2 - y2).abs() < 1e-4 {
        EaseMode::Linear
    } else {
        EaseMode::CubicBezier { x1, y1, x2, y2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_ease() {
        assert!((cubic_bezier_ease(0.0, 0.0, 1.0, 1.0, 0.5) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_ease_in_out() {
        let mid = cubic_bezier_ease(0.42, 0.0, 0.58, 1.0, 0.5);
        assert!((mid - 0.5).abs() < 0.05); // ease-in-out is ~0.5 at t=0.5
    }

    #[test]
    fn test_endpoints() {
        assert!((cubic_bezier_ease(0.25, 0.1, 0.25, 1.0, 0.0)).abs() < 1e-6);
        assert!((cubic_bezier_ease(0.25, 0.1, 0.25, 1.0, 1.0) - 1.0).abs() < 1e-6);
    }
}
