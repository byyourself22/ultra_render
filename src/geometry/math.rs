use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec2, Vec4};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vec2D {
    pub x: f32,
    pub y: f32,
}

impl Vec2D {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            return Self::ZERO;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
        }
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    pub fn to_glam(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    pub fn from_glam(v: Vec2) -> Self {
        Self { x: v.x, y: v.y }
    }

    pub fn distance(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl std::ops::Add for Vec2D {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub for Vec2D {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Mul<f32> for Vec2D {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl std::ops::Neg for Vec2D {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

/// 2D affine transform matrix [a, b, c, d, tx, ty]
/// | a  c  tx |
/// | b  d  ty |
/// | 0  0  1  |
#[derive(Clone, Copy, Debug)]
pub struct Mat2D {
    pub values: [f32; 6],
}

impl Default for Mat2D {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mat2D {
    pub fn identity() -> Self {
        Self {
            values: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    pub fn from_translate(tx: f32, ty: f32) -> Self {
        Self {
            values: [1.0, 0.0, 0.0, 1.0, tx, ty],
        }
    }

    pub fn from_scale(sx: f32, sy: f32) -> Self {
        Self {
            values: [sx, 0.0, 0.0, sy, 0.0, 0.0],
        }
    }

    pub fn from_rotation(radians: f32) -> Self {
        let c = radians.cos();
        let s = radians.sin();
        Self {
            values: [c, s, -s, c, 0.0, 0.0],
        }
    }

    pub fn from_components(tx: f32, ty: f32, sx: f32, sy: f32, rotation: f32) -> Self {
        let mut result = Self::from_rotation(rotation);
        result.values[0] *= sx;
        result.values[1] *= sx;
        result.values[2] *= sy;
        result.values[3] *= sy;
        result.values[4] = tx;
        result.values[5] = ty;
        result
    }

    pub fn multiply(a: &Self, b: &Self) -> Self {
        Self {
            values: [
                a.values[0] * b.values[0] + a.values[2] * b.values[1],
                a.values[1] * b.values[0] + a.values[3] * b.values[1],
                a.values[0] * b.values[2] + a.values[2] * b.values[3],
                a.values[1] * b.values[2] + a.values[3] * b.values[3],
                a.values[0] * b.values[4] + a.values[2] * b.values[5] + a.values[4],
                a.values[1] * b.values[4] + a.values[3] * b.values[5] + a.values[5],
            ],
        }
    }

    pub fn transform_point(&self, p: Vec2D) -> Vec2D {
        Vec2D {
            x: self.values[0] * p.x + self.values[2] * p.y + self.values[4],
            y: self.values[1] * p.x + self.values[3] * p.y + self.values[5],
        }
    }

    pub fn transform_direction(&self, p: Vec2D) -> Vec2D {
        Vec2D {
            x: self.values[0] * p.x + self.values[2] * p.y,
            y: self.values[1] * p.x + self.values[3] * p.y,
        }
    }

    pub fn invert(&self) -> Option<Self> {
        let det = self.values[0] * self.values[3] - self.values[1] * self.values[2];
        if det.abs() < 1e-10 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self {
            values: [
                self.values[3] * inv_det,
                -self.values[1] * inv_det,
                -self.values[2] * inv_det,
                self.values[0] * inv_det,
                (self.values[2] * self.values[5] - self.values[3] * self.values[4]) * inv_det,
                (self.values[1] * self.values[4] - self.values[0] * self.values[5]) * inv_det,
            ],
        })
    }

    pub fn max_scale(&self) -> f32 {
        let sx = (self.values[0] * self.values[0] + self.values[1] * self.values[1]).sqrt();
        let sy = (self.values[2] * self.values[2] + self.values[3] * self.values[3]).sqrt();
        sx.max(sy)
    }

    pub fn to_mat3(&self) -> Mat3 {
        Mat3::from_cols(
            glam::Vec3::new(self.values[0], self.values[1], 0.0),
            glam::Vec3::new(self.values[2], self.values[3], 0.0),
            glam::Vec3::new(self.values[4], self.values[5], 1.0),
        )
    }

    pub fn to_mat4_bytes(&self) -> [f32; 16] {
        [
            self.values[0],
            self.values[1],
            0.0,
            0.0,
            self.values[2],
            self.values[3],
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            self.values[4],
            self.values[5],
            0.0,
            1.0,
        ]
    }
}

/// Axis-aligned bounding box
#[derive(Clone, Copy, Debug)]
pub struct AABB {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

impl Default for AABB {
    fn default() -> Self {
        Self::empty()
    }
}

impl AABB {
    pub fn empty() -> Self {
        Self {
            min_x: f32::MAX,
            min_y: f32::MAX,
            max_x: f32::MIN,
            max_y: f32::MIN,
        }
    }

    pub fn new(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    pub fn width(&self) -> f32 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f32 {
        self.max_y - self.min_y
    }

    pub fn center(&self) -> Vec2D {
        Vec2D::new(
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
        )
    }

    pub fn expand_to_include(&mut self, point: Vec2D) {
        self.min_x = self.min_x.min(point.x);
        self.min_y = self.min_y.min(point.y);
        self.max_x = self.max_x.max(point.x);
        self.max_y = self.max_y.max(point.y);
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    pub fn contains_point(&self, p: Vec2D) -> bool {
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }

    pub fn is_empty(&self) -> bool {
        self.min_x >= self.max_x || self.min_y >= self.max_y
    }

    pub fn transform(&self, mat: &Mat2D) -> Self {
        let corners = [
            Vec2D::new(self.min_x, self.min_y),
            Vec2D::new(self.max_x, self.min_y),
            Vec2D::new(self.max_x, self.max_y),
            Vec2D::new(self.min_x, self.max_y),
        ];
        let mut result = Self::empty();
        for c in &corners {
            result.expand_to_include(mat.transform_point(*c));
        }
        result
    }
}

/// Cubic bezier utilities (Rive-style)
pub fn cubic_tangents(p0: Vec2D, p1: Vec2D, p2: Vec2D, p3: Vec2D) -> (Vec2D, Vec2D) {
    let mut tan0 = p1 - p0;
    if tan0.length_sq() < 1e-10 {
        tan0 = p2 - p0;
        if tan0.length_sq() < 1e-10 {
            tan0 = p3 - p0;
        }
    }
    let mut tan1 = p3 - p2;
    if tan1.length_sq() < 1e-10 {
        tan1 = p3 - p1;
        if tan1.length_sq() < 1e-10 {
            tan1 = p3 - p0;
        }
    }
    (tan0, tan1)
}

/// Wang's formula for cubic bezier segment count estimation
/// Based on Rive's tessellation approach
pub fn wang_cubic_segment_count(p0: Vec2D, p1: Vec2D, p2: Vec2D, p3: Vec2D, precision: f32) -> u32 {
    let d0 = p0 - p1 * 2.0 + p2;
    let d1 = p1 - p2 * 2.0 + p3;
    let m = d0.length_sq().max(d1.length_sq());
    let n = (0.75 * 4.0 * m.sqrt()).sqrt().ceil().max(1.0);
    (n * precision).min(1024.0) as u32
}

/// Evaluate cubic bezier at parameter t using De Casteljau's algorithm (Rive-style)
pub fn cubic_eval(p0: Vec2D, p1: Vec2D, p2: Vec2D, p3: Vec2D, t: f32) -> Vec2D {
    let ab = p0.lerp(p1, t);
    let bc = p1.lerp(p2, t);
    let cd = p2.lerp(p3, t);
    let abc = ab.lerp(bc, t);
    let bcd = bc.lerp(cd, t);
    abc.lerp(bcd, t)
}

/// Evaluate cubic bezier tangent at parameter t
pub fn cubic_tangent_at(p0: Vec2D, p1: Vec2D, p2: Vec2D, p3: Vec2D, t: f32) -> Vec2D {
    let ab = p0.lerp(p1, t);
    let bc = p1.lerp(p2, t);
    let cd = p2.lerp(p3, t);
    let abc = ab.lerp(bc, t);
    let bcd = bc.lerp(cd, t);
    bcd - abc
}

/// Color as RGBA u8
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn from_rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    pub fn premultiply(&self) -> Self {
        Self {
            r: self.r * self.a,
            g: self.g * self.a,
            b: self.b * self.a,
            a: self.a,
        }
    }

    pub fn with_opacity(&self, opacity: f32) -> Self {
        Self {
            r: self.r,
            g: self.g,
            b: self.b,
            a: self.a * opacity,
        }
    }

    pub fn to_vec4(&self) -> Vec4 {
        Vec4::new(self.r, self.g, self.b, self.a)
    }
}
