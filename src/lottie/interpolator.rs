/// Cubic bezier easing interpolation (matches ThorVG's LottieInterpolator)
///
/// Given control points (x1,y1) and (x2,y2) in [0,1], computes the eased
/// value for a linear progress t in [0,1].
///
/// Uses a pre-computed sampling table for initial guess, then refines with
/// Newton-Raphson (when slope is steep enough) or binary subdivision (when
/// slope is too shallow for Newton to converge reliably).

const SPLINE_TABLE_SIZE: usize = 11;
const SAMPLE_STEP_SIZE: f32 = 1.0 / (SPLINE_TABLE_SIZE as f32 - 1.0);
const NEWTON_MIN_SLOPE: f32 = 0.02;
const NEWTON_ITERATIONS: u32 = 4;
const SUBDIVISION_PRECISION: f32 = 0.0000001;
const SUBDIVISION_MAX_ITERATIONS: u32 = 10;

#[inline]
fn const_a(a1: f32, a2: f32) -> f32 {
    1.0 - 3.0 * a2 + 3.0 * a1
}

#[inline]
fn const_b(a1: f32, a2: f32) -> f32 {
    3.0 * a2 - 6.0 * a1
}

#[inline]
fn const_c(a1: f32) -> f32 {
    3.0 * a1
}

/// Evaluate cubic bezier component using Horner form (ThorVG _calcBezier)
#[inline]
fn calc_bezier(t: f32, a1: f32, a2: f32) -> f32 {
    ((const_a(a1, a2) * t + const_b(a1, a2)) * t + const_c(a1)) * t
}

/// Derivative of cubic bezier component (ThorVG _getSlope)
#[inline]
fn get_slope(t: f32, a1: f32, a2: f32) -> f32 {
    3.0 * const_a(a1, a2) * t * t + 2.0 * const_b(a1, a2) * t + const_c(a1)
}

/// Build the sample table for x-component (11 evenly spaced samples)
fn build_samples(x1: f32, x2: f32) -> [f32; SPLINE_TABLE_SIZE] {
    let mut samples = [0.0f32; SPLINE_TABLE_SIZE];
    for i in 0..SPLINE_TABLE_SIZE {
        samples[i] = calc_bezier(i as f32 * SAMPLE_STEP_SIZE, x1, x2);
    }
    samples
}

/// Newton-Raphson refinement (ThorVG NewtonRaphsonIterate)
fn newton_raphson(target_x: f32, guess: f32, x1: f32, x2: f32) -> f32 {
    let mut t = guess;
    for _ in 0..NEWTON_ITERATIONS {
        let current_x = calc_bezier(t, x1, x2) - target_x;
        let slope = get_slope(t, x1, x2);
        if slope == 0.0 {
            return t;
        }
        t -= current_x / slope;
    }
    t
}

/// Binary subdivision fallback (ThorVG binarySubdivide)
fn binary_subdivide(target_x: f32, mut lo: f32, mut hi: f32, x1: f32, x2: f32) -> f32 {
    let mut t;
    let mut i = 0u32;
    loop {
        t = lo + (hi - lo) / 2.0;
        let x = calc_bezier(t, x1, x2) - target_x;
        if x > 0.0 {
            hi = t;
        } else {
            lo = t;
        }
        i += 1;
        if x.abs() <= SUBDIVISION_PRECISION || i >= SUBDIVISION_MAX_ITERATIONS {
            break;
        }
    }
    t
}

/// Find t for a given x using sample table + Newton/bisection (ThorVG getTForX)
fn get_t_for_x(target_x: f32, samples: &[f32; SPLINE_TABLE_SIZE], x1: f32, x2: f32) -> f32 {
    // Find interval where t lies
    let mut interval_start = 0.0f32;
    let mut current = 1;
    let last = SPLINE_TABLE_SIZE - 1;

    while current < last && samples[current] <= target_x {
        interval_start += SAMPLE_STEP_SIZE;
        current += 1;
    }
    current -= 1;

    // Interpolate to provide an initial guess for t
    let dist = (target_x - samples[current]) / (samples[current + 1] - samples[current]);
    let guess = interval_start + dist * SAMPLE_STEP_SIZE;

    // Check slope to decide strategy
    let initial_slope = get_slope(guess, x1, x2);
    if initial_slope >= NEWTON_MIN_SLOPE {
        newton_raphson(target_x, guess, x1, x2)
    } else if initial_slope == 0.0 {
        guess
    } else {
        binary_subdivide(target_x, interval_start, interval_start + SAMPLE_STEP_SIZE, x1, x2)
    }
}

/// Evaluate a cubic bezier easing curve (matches ThorVG LottieInterpolator::progress).
/// Control points: (0,0) -> (x1,y1) -> (x2,y2) -> (1,1)
/// Input `t` is the time ratio [0,1], returns eased ratio [0,1].
pub fn cubic_bezier_ease(x1: f32, y1: f32, x2: f32, y2: f32, t: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }

    // Linear case (matches ThorVG: outTangent.x == outTangent.y && inTangent.x == inTangent.y)
    if x1 == y1 && x2 == y2 {
        return t;
    }

    let samples = build_samples(x1, x2);
    let s = get_t_for_x(t, &samples, x1, x2);
    calc_bezier(s, y1, y2)
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
