/// ThorVG-style path modifier chain.
///
/// Modifiers process paths in order: geometry shapes → modifiers → paints.
/// Each modifier transforms a RawPath into a new RawPath.
/// ThorVG implements this as a linked-list chain with `next` pointer;
/// here we use a Vec<Modifier> applied sequentially.
use crate::geometry::math::Vec2D;
use crate::geometry::path::{PathVerb, RawPath};
use crate::lottie::model::*;
use crate::lottie::property::eval_f32;

/// A path modifier that transforms geometry before painting
#[derive(Clone, Debug)]
pub enum Modifier {
    TrimPath {
        start: AnimatedValue<f32>,
        end: AnimatedValue<f32>,
        offset: AnimatedValue<f32>,
        trim_type: TrimPathType,
    },
    RoundedCorners {
        radius: AnimatedValue<f32>,
    },
    // Future: OffsetPath, PuckerBloat, ZigZag
}

/// Evaluate and apply a chain of modifiers to a set of paths at a given frame.
/// ThorVG processes modifiers bottom-up within each group — modifiers apply
/// to all geometry shapes above them in the same group.
pub fn apply_modifiers(paths: &[RawPath], modifiers: &[Modifier], frame: f32) -> Vec<RawPath> {
    let mut result = paths.to_vec();

    for modifier in modifiers {
        result = match modifier {
            Modifier::TrimPath {
                start,
                end,
                offset,
                trim_type,
            } => {
                let s = eval_f32(start, frame) / 100.0;
                let e = eval_f32(end, frame) / 100.0;
                let o = eval_f32(offset, frame) / 360.0;
                match trim_type {
                    TrimPathType::Simultaneously => {
                        // All paths trimmed as one combined path
                        let combined = combine_paths_for_trim(&result);
                        vec![trim_path(&combined, s, e, o)]
                    }
                    TrimPathType::Individually => {
                        // Each path trimmed independently
                        result.iter().map(|p| trim_path(p, s, e, o)).collect()
                    }
                }
            }
            Modifier::RoundedCorners { radius } => {
                let r = eval_f32(radius, frame);
                if r > 0.0 {
                    result.iter().map(|p| round_corners(p, r)).collect()
                } else {
                    result
                }
            }
        };
    }

    result
}

/// Extract modifiers from a shape group's items (ThorVG-style bottom-up scan)
pub fn collect_modifiers(shapes: &[ShapeItem]) -> Vec<Modifier> {
    let mut modifiers = Vec::new();
    for shape in shapes {
        match shape {
            ShapeItem::TrimPath(tp) if !tp.hidden => {
                modifiers.push(Modifier::TrimPath {
                    start: tp.start.clone(),
                    end: tp.end.clone(),
                    offset: tp.offset.clone(),
                    trim_type: tp.trim_type,
                });
            }
            ShapeItem::RoundedCorners(rc) if !rc.hidden => {
                modifiers.push(Modifier::RoundedCorners {
                    radius: rc.radius.clone(),
                });
            }
            _ => {}
        }
    }
    modifiers
}

// ─── Trim Path Implementation ───────────────────────────────

/// Measure total path length by walking segments
fn path_length(path: &RawPath) -> f32 {
    let mut total = 0.0;
    let mut pi = 0usize;
    let mut last = Vec2D::ZERO;

    for verb in &path.verbs {
        match verb {
            PathVerb::Move => {
                last = path.points[pi];
                pi += 1;
            }
            PathVerb::Line => {
                let to = path.points[pi];
                total += (to - last).length();
                last = to;
                pi += 1;
            }
            PathVerb::Quad => {
                let cp = path.points[pi];
                let to = path.points[pi + 1];
                total += approx_quad_length(last, cp, to);
                last = to;
                pi += 2;
            }
            PathVerb::Cubic => {
                let c1 = path.points[pi];
                let c2 = path.points[pi + 1];
                let to = path.points[pi + 2];
                total += approx_cubic_length(last, c1, c2, to);
                last = to;
                pi += 3;
            }
            PathVerb::Close => {
                // Close doesn't advance point index
            }
        }
    }
    total
}

/// Approximate quadratic bezier length using chord+control polygon average
fn approx_quad_length(p0: Vec2D, p1: Vec2D, p2: Vec2D) -> f32 {
    let chord = (p2 - p0).length();
    let control = (p1 - p0).length() + (p2 - p1).length();
    (chord + control) * 0.5
}

/// Approximate cubic bezier length using chord+control polygon average
fn approx_cubic_length(p0: Vec2D, p1: Vec2D, p2: Vec2D, p3: Vec2D) -> f32 {
    let chord = (p3 - p0).length();
    let control = (p1 - p0).length() + (p2 - p1).length() + (p3 - p2).length();
    (chord + control) * 0.5
}

/// Evaluate a point on a cubic bezier at parameter t
fn cubic_at(p0: Vec2D, p1: Vec2D, p2: Vec2D, p3: Vec2D, t: f32) -> Vec2D {
    let u = 1.0 - t;
    let tt = t * t;
    let uu = u * u;
    Vec2D::new(
        uu * u * p0.x + 3.0 * uu * t * p1.x + 3.0 * u * tt * p2.x + tt * t * p3.x,
        uu * u * p0.y + 3.0 * uu * t * p1.y + 3.0 * u * tt * p2.y + tt * t * p3.y,
    )
}

/// Evaluate a point on a quad bezier at parameter t
fn quad_at(p0: Vec2D, p1: Vec2D, p2: Vec2D, t: f32) -> Vec2D {
    let u = 1.0 - t;
    Vec2D::new(
        u * u * p0.x + 2.0 * u * t * p1.x + t * t * p2.x,
        u * u * p0.y + 2.0 * u * t * p1.y + t * t * p2.y,
    )
}

/// Split a cubic bezier at parameter t using De Casteljau
fn split_cubic(p0: Vec2D, p1: Vec2D, p2: Vec2D, p3: Vec2D, t: f32) -> ([Vec2D; 4], [Vec2D; 4]) {
    let a = Vec2D::lerp(p0, p1, t);
    let b = Vec2D::lerp(p1, p2, t);
    let c = Vec2D::lerp(p2, p3, t);
    let d = Vec2D::lerp(a, b, t);
    let e = Vec2D::lerp(b, c, t);
    let f = Vec2D::lerp(d, e, t);
    ([p0, a, d, f], [f, e, c, p3])
}

/// Trim a path to [start_frac..end_frac] with offset, all in 0..1 range
fn trim_path(path: &RawPath, start: f32, end: f32, offset: f32) -> RawPath {
    if path.is_empty() {
        return RawPath::new();
    }

    let total_len = path_length(path);
    if total_len <= 0.0 {
        return path.clone();
    }

    // Apply offset and normalize
    let s = (start + offset).rem_euclid(1.0);
    let e = (end + offset).rem_euclid(1.0);

    if (s - e).abs() < 1e-6 {
        return if (start - end).abs() < 1e-6 {
            RawPath::new() // zero-length trim
        } else {
            path.clone() // full path
        };
    }

    // Handle wrap-around case by splitting into two trims
    if s > e {
        let mut p1 = trim_path_segment(path, total_len, s, 1.0);
        let p2 = trim_path_segment(path, total_len, 0.0, e);
        // Append p2 to p1
        for verb in &p2.verbs {
            p1.verbs.push(*verb);
        }
        for pt in &p2.points {
            p1.points.push(*pt);
        }
        return p1;
    }

    trim_path_segment(path, total_len, s, e)
}

/// Extract the segment of path between start_frac and end_frac (0..1)
fn trim_path_segment(path: &RawPath, total_len: f32, start_frac: f32, end_frac: f32) -> RawPath {
    let start_len = start_frac * total_len;
    let end_len = end_frac * total_len;
    let mut result = RawPath::new();
    let mut cumulative = 0.0f32;
    let mut pi = 0usize;
    let mut last = Vec2D::ZERO;
    let mut started = false;

    for verb in &path.verbs {
        match verb {
            PathVerb::Move => {
                last = path.points[pi];
                pi += 1;
            }
            PathVerb::Line => {
                let to = path.points[pi];
                let seg_len = (to - last).length();
                let seg_start = cumulative;
                let seg_end = cumulative + seg_len;

                if seg_end > start_len && seg_start < end_len {
                    let t0 = ((start_len - seg_start) / seg_len).clamp(0.0, 1.0);
                    let t1 = ((end_len - seg_start) / seg_len).clamp(0.0, 1.0);
                    let p0 = Vec2D::lerp(last, to, t0);
                    let p1 = Vec2D::lerp(last, to, t1);
                    if !started {
                        result.move_to(p0.x, p0.y);
                        started = true;
                    }
                    result.line_to(p1.x, p1.y);
                }

                cumulative = seg_end;
                last = to;
                pi += 1;
            }
            PathVerb::Quad => {
                let cp = path.points[pi];
                let to = path.points[pi + 1];
                let seg_len = approx_quad_length(last, cp, to);
                let seg_start = cumulative;
                let seg_end = cumulative + seg_len;

                if seg_end > start_len && seg_start < end_len {
                    // Approximate: linearize the portion we need
                    let t0 = ((start_len - seg_start) / seg_len).clamp(0.0, 1.0);
                    let t1 = ((end_len - seg_start) / seg_len).clamp(0.0, 1.0);
                    let p0 = quad_at(last, cp, to, t0);
                    let p1 = quad_at(last, cp, to, t1);
                    if !started {
                        result.move_to(p0.x, p0.y);
                        started = true;
                    }
                    result.line_to(p1.x, p1.y);
                }

                cumulative = seg_end;
                last = to;
                pi += 2;
            }
            PathVerb::Cubic => {
                let c1 = path.points[pi];
                let c2 = path.points[pi + 1];
                let to = path.points[pi + 2];
                let seg_len = approx_cubic_length(last, c1, c2, to);
                let seg_start = cumulative;
                let seg_end = cumulative + seg_len;

                if seg_end > start_len && seg_start < end_len {
                    let t0 = ((start_len - seg_start) / seg_len).clamp(0.0, 1.0);
                    let t1 = ((end_len - seg_start) / seg_len).clamp(0.0, 1.0);

                    // Split cubic at t0 and t1 for accurate trimming
                    let (_, right) = split_cubic(last, c1, c2, to, t0);
                    let adjusted_t1 = if (1.0 - t0).abs() > 1e-6 {
                        (t1 - t0) / (1.0 - t0)
                    } else {
                        1.0
                    };
                    let (seg, _) = split_cubic(
                        right[0],
                        right[1],
                        right[2],
                        right[3],
                        adjusted_t1.clamp(0.0, 1.0),
                    );

                    if !started {
                        result.move_to(seg[0].x, seg[0].y);
                        started = true;
                    }
                    result.cubic_to(seg[1].x, seg[1].y, seg[2].x, seg[2].y, seg[3].x, seg[3].y);
                }

                cumulative = seg_end;
                last = to;
                pi += 3;
            }
            PathVerb::Close => {}
        }

        if cumulative >= end_len {
            break;
        }
    }

    result
}

/// Combine multiple paths into one for simultaneous trimming
fn combine_paths_for_trim(paths: &[RawPath]) -> RawPath {
    let mut combined = RawPath::new();
    for p in paths {
        for verb in &p.verbs {
            combined.verbs.push(*verb);
        }
        for pt in &p.points {
            combined.points.push(*pt);
        }
    }
    combined
}

// ─── Rounded Corners Implementation ────────────────────────

/// Apply rounded corners to a path (ThorVG LottieRoundnessModifier)
/// Replaces sharp corners with circular arcs of given radius.
fn round_corners(path: &RawPath, radius: f32) -> RawPath {
    if radius <= 0.0 || path.is_empty() {
        return path.clone();
    }

    let mut result = RawPath::new();
    let kappa = 0.5522847498_f32;

    // Process each contour separately
    let mut pi = 0usize;
    let mut contour_points: Vec<Vec2D> = Vec::new();
    let mut is_closed = false;

    for verb in &path.verbs {
        match verb {
            PathVerb::Move => {
                // Flush previous contour
                if !contour_points.is_empty() {
                    round_contour(&contour_points, is_closed, radius, kappa, &mut result);
                    contour_points.clear();
                }
                contour_points.push(path.points[pi]);
                is_closed = false;
                pi += 1;
            }
            PathVerb::Line => {
                contour_points.push(path.points[pi]);
                pi += 1;
            }
            PathVerb::Quad => {
                // For quads, just pass through (rounding only affects line corners)
                contour_points.push(path.points[pi]);
                contour_points.push(path.points[pi + 1]);
                pi += 2;
            }
            PathVerb::Cubic => {
                // For cubics, just pass through
                contour_points.push(path.points[pi]);
                contour_points.push(path.points[pi + 1]);
                contour_points.push(path.points[pi + 2]);
                pi += 3;
            }
            PathVerb::Close => {
                is_closed = true;
            }
        }
    }

    // Flush last contour
    if !contour_points.is_empty() {
        round_contour(&contour_points, is_closed, radius, kappa, &mut result);
    }

    result
}

/// Round corners of a single contour made of line segments
fn round_contour(points: &[Vec2D], closed: bool, radius: f32, kappa: f32, out: &mut RawPath) {
    let n = points.len();
    if n < 2 {
        if n == 1 {
            out.move_to(points[0].x, points[0].y);
        }
        return;
    }

    // For each corner, compute the rounding
    let count = if closed { n } else { n };
    let mut first = true;

    for i in 0..count {
        let curr = points[i];

        if !closed && (i == 0 || i == n - 1) {
            // Start/end points of open path: no rounding
            if first {
                out.move_to(curr.x, curr.y);
                first = false;
            } else {
                out.line_to(curr.x, curr.y);
            }
            continue;
        }

        let prev = if i == 0 { points[n - 1] } else { points[i - 1] };
        let next = if i == n - 1 { points[0] } else { points[i + 1] };

        let d_prev = prev - curr;
        let d_next = next - curr;
        let len_prev = d_prev.length();
        let len_next = d_next.length();

        if len_prev < 1e-4 || len_next < 1e-4 {
            if first {
                out.move_to(curr.x, curr.y);
                first = false;
            } else {
                out.line_to(curr.x, curr.y);
            }
            continue;
        }

        // Clamp radius to half of shortest adjacent edge
        let r = radius.min(len_prev * 0.5).min(len_next * 0.5);

        let dir_prev = Vec2D::new(d_prev.x / len_prev, d_prev.y / len_prev);
        let dir_next = Vec2D::new(d_next.x / len_next, d_next.y / len_next);

        // Points where the arc starts and ends
        let arc_start = Vec2D::new(curr.x + dir_prev.x * r, curr.y + dir_prev.y * r);
        let arc_end = Vec2D::new(curr.x + dir_next.x * r, curr.y + dir_next.y * r);

        // Control points for cubic approximation of circular arc
        let cp1 = Vec2D::new(
            arc_start.x - dir_prev.x * r * kappa,
            arc_start.y - dir_prev.y * r * kappa,
        );
        let cp2 = Vec2D::new(
            arc_end.x - dir_next.x * r * kappa,
            arc_end.y - dir_next.y * r * kappa,
        );

        if first {
            out.move_to(arc_start.x, arc_start.y);
            first = false;
        } else {
            out.line_to(arc_start.x, arc_start.y);
        }
        out.cubic_to(cp1.x, cp1.y, cp2.x, cp2.y, arc_end.x, arc_end.y);
    }

    if closed {
        out.close();
    }
}
