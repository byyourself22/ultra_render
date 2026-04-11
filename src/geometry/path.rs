use super::math::{Mat2D, Vec2D, AABB};

/// Path commands (matching Rive's PathVerb)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathVerb {
    Move,
    Line,
    Quad,
    Cubic,
    Close,
}

/// A raw path of commands and points (Rive RawPath equivalent)
#[derive(Clone, Debug, Default)]
pub struct RawPath {
    pub verbs: Vec<PathVerb>,
    pub points: Vec<Vec2D>,
}

impl RawPath {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn move_to(&mut self, x: f32, y: f32) {
        self.verbs.push(PathVerb::Move);
        self.points.push(Vec2D::new(x, y));
    }

    pub fn line_to(&mut self, x: f32, y: f32) {
        self.verbs.push(PathVerb::Line);
        self.points.push(Vec2D::new(x, y));
    }

    pub fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.verbs.push(PathVerb::Quad);
        self.points.push(Vec2D::new(cx, cy));
        self.points.push(Vec2D::new(x, y));
    }

    pub fn cubic_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) {
        self.verbs.push(PathVerb::Cubic);
        self.points.push(Vec2D::new(c1x, c1y));
        self.points.push(Vec2D::new(c2x, c2y));
        self.points.push(Vec2D::new(x, y));
    }

    pub fn close(&mut self) {
        self.verbs.push(PathVerb::Close);
    }

    pub fn is_empty(&self) -> bool {
        self.verbs.is_empty()
    }

    pub fn reset(&mut self) {
        self.verbs.clear();
        self.points.clear();
    }

    pub fn bounds(&self) -> AABB {
        let mut aabb = AABB::empty();
        for p in &self.points {
            aabb.expand_to_include(*p);
        }
        aabb
    }

    pub fn transform(&self, mat: &Mat2D) -> Self {
        let mut result = Self::new();
        result.verbs = self.verbs.clone();
        result.points = self
            .points
            .iter()
            .map(|p| mat.transform_point(*p))
            .collect();
        result
    }

    pub fn add_path(&mut self, other: &RawPath, transform: &Mat2D) {
        for verb in &other.verbs {
            self.verbs.push(*verb);
        }
        for p in &other.points {
            self.points.push(transform.transform_point(*p));
        }
    }

    /// Iterate over contours (subpaths)
    pub fn contours(&self) -> ContourIter<'_> {
        ContourIter {
            path: self,
            verb_idx: 0,
            point_idx: 0,
        }
    }

    /// Add a rectangle
    pub fn add_rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.move_to(x, y);
        self.line_to(x + w, y);
        self.line_to(x + w, y + h);
        self.line_to(x, y + h);
        self.close();
    }

    /// Add a rounded rectangle
    pub fn add_rounded_rect(&mut self, x: f32, y: f32, w: f32, h: f32, r: f32) {
        let r = r.min(w * 0.5).min(h * 0.5);
        if r <= 0.0 {
            self.add_rect(x, y, w, h);
            return;
        }
        let k = 0.5522847498; // kappa for circular arcs
        let kr = k * r;

        self.move_to(x + r, y);
        self.line_to(x + w - r, y);
        self.cubic_to(x + w - r + kr, y, x + w, y + r - kr, x + w, y + r);
        self.line_to(x + w, y + h - r);
        self.cubic_to(
            x + w,
            y + h - r + kr,
            x + w - r + kr,
            y + h,
            x + w - r,
            y + h,
        );
        self.line_to(x + r, y + h);
        self.cubic_to(x + r - kr, y + h, x, y + h - r + kr, x, y + h - r);
        self.line_to(x, y + r);
        self.cubic_to(x, y + r - kr, x + r - kr, y, x + r, y);
        self.close();
    }

    /// Add an ellipse
    pub fn add_ellipse(&mut self, cx: f32, cy: f32, rx: f32, ry: f32) {
        let k = 0.5522847498;
        let kx = k * rx;
        let ky = k * ry;

        self.move_to(cx + rx, cy);
        self.cubic_to(cx + rx, cy + ky, cx + kx, cy + ry, cx, cy + ry);
        self.cubic_to(cx - kx, cy + ry, cx - rx, cy + ky, cx - rx, cy);
        self.cubic_to(cx - rx, cy - ky, cx - kx, cy - ry, cx, cy - ry);
        self.cubic_to(cx + kx, cy - ry, cx + rx, cy - ky, cx + rx, cy);
        self.close();
    }

    /// Add a polystar (star or polygon)
    pub fn add_polystar(
        &mut self,
        cx: f32,
        cy: f32,
        points: u32,
        outer_radius: f32,
        inner_radius: f32,
        outer_roundness: f32,
        inner_roundness: f32,
        rotation: f32,
        is_star: bool,
    ) {
        let num_points = if is_star { points * 2 } else { points };
        let angle_per_point = std::f32::consts::TAU / num_points as f32;
        let half_angle = angle_per_point * 0.5;
        let start_angle = rotation - std::f32::consts::FRAC_PI_2;

        for i in 0..num_points {
            let angle = start_angle + angle_per_point * i as f32;
            let (radius, roundness) = if is_star && i % 2 == 1 {
                (inner_radius, inner_roundness)
            } else {
                (outer_radius, outer_roundness)
            };

            let x = cx + angle.cos() * radius;
            let y = cy + angle.sin() * radius;

            if i == 0 {
                self.move_to(x, y);
            } else if roundness > 0.0 {
                let prev_angle = start_angle + angle_per_point * (i as f32 - 1.0);
                let (prev_radius, _) = if is_star && (i - 1) % 2 == 1 {
                    (inner_radius, inner_roundness)
                } else {
                    (outer_radius, outer_roundness)
                };

                let cp_len = prev_radius * roundness * 0.5 * half_angle.tan();
                let c1x = cx + prev_angle.cos() * prev_radius - prev_angle.sin() * cp_len;
                let c1y = cy + prev_angle.sin() * prev_radius + prev_angle.cos() * cp_len;

                let cp_len2 = radius * roundness * 0.5 * half_angle.tan();
                let c2x = x + angle.sin() * cp_len2;
                let c2y = y - angle.cos() * cp_len2;

                self.cubic_to(c1x, c1y, c2x, c2y, x, y);
            } else {
                self.line_to(x, y);
            }
        }
        self.close();
    }
}

pub struct ContourIter<'a> {
    path: &'a RawPath,
    verb_idx: usize,
    point_idx: usize,
}

pub struct Contour {
    pub points: Vec<Vec2D>,
    pub verbs: Vec<PathVerb>,
    pub closed: bool,
}

impl<'a> Iterator for ContourIter<'a> {
    type Item = Contour;

    fn next(&mut self) -> Option<Self::Item> {
        if self.verb_idx >= self.path.verbs.len() {
            return None;
        }

        let mut contour = Contour {
            points: Vec::new(),
            verbs: Vec::new(),
            closed: false,
        };

        while self.verb_idx < self.path.verbs.len() {
            let verb = self.path.verbs[self.verb_idx];
            match verb {
                PathVerb::Move => {
                    if !contour.points.is_empty() {
                        break; // start of next contour
                    }
                    contour.verbs.push(verb);
                    contour.points.push(self.path.points[self.point_idx]);
                    self.point_idx += 1;
                }
                PathVerb::Line => {
                    contour.verbs.push(verb);
                    contour.points.push(self.path.points[self.point_idx]);
                    self.point_idx += 1;
                }
                PathVerb::Quad => {
                    contour.verbs.push(verb);
                    contour.points.push(self.path.points[self.point_idx]);
                    contour.points.push(self.path.points[self.point_idx + 1]);
                    self.point_idx += 2;
                }
                PathVerb::Cubic => {
                    contour.verbs.push(verb);
                    contour.points.push(self.path.points[self.point_idx]);
                    contour.points.push(self.path.points[self.point_idx + 1]);
                    contour.points.push(self.path.points[self.point_idx + 2]);
                    self.point_idx += 3;
                }
                PathVerb::Close => {
                    contour.verbs.push(verb);
                    contour.closed = true;
                    self.verb_idx += 1;
                    break;
                }
            }
            self.verb_idx += 1;
        }

        if contour.points.is_empty() {
            None
        } else {
            Some(contour)
        }
    }
}

/// Fill rule (Rive-style)
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum FillRule {
    #[default]
    NonZero,
    EvenOdd,
}

/// Stroke cap style
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum StrokeCap {
    #[default]
    Butt,
    Round,
    Square,
}

/// Stroke join style
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum StrokeJoin {
    #[default]
    Miter,
    Round,
    Bevel,
}
