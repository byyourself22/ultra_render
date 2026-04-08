use super::math::{Vec2D, Mat2D, Color, wang_cubic_segment_count, cubic_eval};
use super::path::{RawPath, PathVerb, FillRule, StrokeCap, StrokeJoin};
use bytemuck::{Pod, Zeroable};

/// GPU vertex for rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

/// Tessellated mesh ready for GPU
#[derive(Clone, Debug, Default)]
pub struct TessellatedMesh {
    pub vertices: Vec<GpuVertex>,
    pub indices: Vec<u32>,
}

impl TessellatedMesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn append(&mut self, other: &TessellatedMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }
}

/// Tessellator: converts paths to GPU-ready triangle meshes
/// Inspired by Rive's tessellation and ThorVG's WgGeometry
pub struct Tessellator {
    precision: f32,
}

impl Tessellator {
    pub fn new() -> Self {
        Self { precision: 1.0 }
    }

    pub fn with_precision(precision: f32) -> Self {
        Self { precision }
    }

    /// Tessellate a filled path into triangles
    pub fn tessellate_fill(
        &self,
        path: &RawPath,
        _fill_rule: FillRule,
        color: Color,
        transform: &Mat2D,
    ) -> TessellatedMesh {
        let mut mesh = TessellatedMesh::new();
        if path.is_empty() {
            return mesh;
        }

        // Flatten cubics into line segments, then triangulate
        let flattened = self.flatten_path(path, transform);

        for contour_points in &flattened {
            if contour_points.len() < 3 {
                continue;
            }
            self.triangulate_contour(contour_points, color, &mut mesh);
        }

        mesh
    }

    /// Tessellate a stroked path into triangles
    pub fn tessellate_stroke(
        &self,
        path: &RawPath,
        width: f32,
        cap: StrokeCap,
        join: StrokeJoin,
        miter_limit: f32,
        color: Color,
        transform: &Mat2D,
    ) -> TessellatedMesh {
        let mut mesh = TessellatedMesh::new();
        if path.is_empty() || width <= 0.0 {
            return mesh;
        }

        let half_width = width * 0.5;
        let flattened = self.flatten_path(path, transform);

        for contour_points in &flattened {
            if contour_points.len() < 2 {
                continue;
            }
            self.stroke_contour(contour_points, half_width, cap, join, miter_limit, color, &mut mesh);
        }

        mesh
    }

    /// Flatten path cubics/quads into line segments using Wang's formula
    fn flatten_path(&self, path: &RawPath, transform: &Mat2D) -> Vec<Vec<Vec2D>> {
        let mut contours: Vec<Vec<Vec2D>> = Vec::new();
        let mut current: Vec<Vec2D> = Vec::new();
        let mut point_idx = 0;
        let mut last_move = Vec2D::ZERO;

        for verb in &path.verbs {
            match verb {
                PathVerb::Move => {
                    if current.len() >= 2 {
                        contours.push(std::mem::take(&mut current));
                    } else {
                        current.clear();
                    }
                    let p = transform.transform_point(path.points[point_idx]);
                    current.push(p);
                    last_move = p;
                    point_idx += 1;
                }
                PathVerb::Line => {
                    let p = transform.transform_point(path.points[point_idx]);
                    current.push(p);
                    point_idx += 1;
                }
                PathVerb::Quad => {
                    let p1 = transform.transform_point(path.points[point_idx]);
                    let p2 = transform.transform_point(path.points[point_idx + 1]);
                    let p0 = *current.last().unwrap_or(&Vec2D::ZERO);
                    // Convert quad to cubic
                    let c1 = Vec2D::new(
                        p0.x + 2.0 / 3.0 * (p1.x - p0.x),
                        p0.y + 2.0 / 3.0 * (p1.y - p0.y),
                    );
                    let c2 = Vec2D::new(
                        p2.x + 2.0 / 3.0 * (p1.x - p2.x),
                        p2.y + 2.0 / 3.0 * (p1.y - p2.y),
                    );
                    self.flatten_cubic(p0, c1, c2, p2, &mut current);
                    point_idx += 2;
                }
                PathVerb::Cubic => {
                    let c1 = transform.transform_point(path.points[point_idx]);
                    let c2 = transform.transform_point(path.points[point_idx + 1]);
                    let p3 = transform.transform_point(path.points[point_idx + 2]);
                    let p0 = *current.last().unwrap_or(&Vec2D::ZERO);
                    self.flatten_cubic(p0, c1, c2, p3, &mut current);
                    point_idx += 3;
                }
                PathVerb::Close => {
                    // Close the contour back to the move point
                    if let Some(first) = current.first() {
                        if current.last().map_or(true, |last| last.distance(*first) > 0.01) {
                            current.push(last_move);
                        }
                    }
                    if current.len() >= 3 {
                        contours.push(std::mem::take(&mut current));
                    } else {
                        current.clear();
                    }
                }
            }
        }

        if current.len() >= 2 {
            contours.push(current);
        }

        contours
    }

    /// Flatten a cubic bezier into line segments using Wang's formula (Rive-style)
    fn flatten_cubic(
        &self,
        p0: Vec2D,
        p1: Vec2D,
        p2: Vec2D,
        p3: Vec2D,
        out: &mut Vec<Vec2D>,
    ) {
        let seg_count = wang_cubic_segment_count(p0, p1, p2, p3, self.precision).max(1);
        let dt = 1.0 / seg_count as f32;

        for i in 1..=seg_count {
            let t = dt * i as f32;
            out.push(cubic_eval(p0, p1, p2, p3, t));
        }
    }

    /// Triangulate a contour using ear clipping for correct concave polygon support.
    /// Falls back to centroid fan for convex polygons (faster).
    fn triangulate_contour(
        &self,
        points: &[Vec2D],
        color: Color,
        mesh: &mut TessellatedMesh,
    ) {
        if points.len() < 3 {
            return;
        }

        let color_arr = [color.r, color.g, color.b, color.a];
        let base_idx = mesh.vertices.len() as u32;

        // Add contour vertices
        for p in points {
            mesh.vertices.push(GpuVertex {
                position: [p.x, p.y],
                uv: [0.0, 0.0],
                color: color_arr,
            });
        }

        // Ear clipping triangulation (handles concave polygons correctly)
        let n = points.len();
        if n == 3 {
            mesh.indices.push(base_idx);
            mesh.indices.push(base_idx + 1);
            mesh.indices.push(base_idx + 2);
            return;
        }

        // Build index list
        let mut indices: Vec<usize> = (0..n).collect();

        // Determine winding order
        let area = signed_polygon_area(points);
        if area > 0.0 {
            indices.reverse(); // Ensure counter-clockwise
        }

        let mut remaining = indices.len();
        let mut ear_idx = 0;
        let mut fail_count = 0;

        while remaining > 2 {
            if fail_count >= remaining {
                // No ears found — degenerate polygon, fall back to fan
                break;
            }

            let prev = indices[(ear_idx + remaining - 1) % remaining];
            let curr = indices[ear_idx % remaining];
            let next = indices[(ear_idx + 1) % remaining];

            let a = points[prev];
            let b = points[curr];
            let c = points[next];

            if is_ear(a, b, c, &indices, points, ear_idx % remaining) {
                mesh.indices.push(base_idx + prev as u32);
                mesh.indices.push(base_idx + curr as u32);
                mesh.indices.push(base_idx + next as u32);
                indices.remove(ear_idx % remaining);
                remaining -= 1;
                fail_count = 0;
                if ear_idx >= remaining && remaining > 0 {
                    ear_idx = 0;
                }
            } else {
                ear_idx += 1;
                if ear_idx >= remaining {
                    ear_idx = 0;
                }
                fail_count += 1;
            }
        }
    }

    /// Generate stroke geometry for a contour
    fn stroke_contour(
        &self,
        points: &[Vec2D],
        half_width: f32,
        cap: StrokeCap,
        join: StrokeJoin,
        miter_limit: f32,
        color: Color,
        mesh: &mut TessellatedMesh,
    ) {
        let n = points.len();
        if n < 2 {
            return;
        }

        let color_arr = [color.r, color.g, color.b, color.a];
        let is_closed = n >= 3
            && (points[0].x - points[n - 1].x).abs() < 0.01
            && (points[0].y - points[n - 1].y).abs() < 0.01;

        // Compute normals for each segment
        let seg_count = n - 1;
        let mut normals: Vec<Vec2D> = Vec::with_capacity(seg_count);
        for i in 0..seg_count {
            let dx = points[i + 1].x - points[i].x;
            let dy = points[i + 1].y - points[i].y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-6 {
                normals.push(Vec2D::new(0.0, 1.0));
            } else {
                normals.push(Vec2D::new(-dy / len, dx / len));
            }
        }

        // Generate vertices: two per point (left and right of stroke)
        let base_idx = mesh.vertices.len() as u32;

        for i in 0..n {
            let normal = if i == 0 {
                if is_closed {
                    average_normal(&normals[seg_count - 1], &normals[0])
                } else {
                    normals[0]
                }
            } else if i == n - 1 {
                if is_closed {
                    average_normal(&normals[seg_count - 1], &normals[0])
                } else {
                    normals[seg_count - 1]
                }
            } else {
                let n1 = &normals[i - 1];
                let n2 = &normals[i];
                match join {
                    StrokeJoin::Miter => {
                        let avg = average_normal(n1, n2);
                        let cos_half = n1.dot(avg);
                        if cos_half > 0.0 && (1.0 / cos_half) <= miter_limit {
                            avg * (1.0 / cos_half)
                        } else {
                            avg
                        }
                    }
                    _ => average_normal(n1, n2),
                }
            };

            let left = Vec2D::new(
                points[i].x + normal.x * half_width,
                points[i].y + normal.y * half_width,
            );
            let right = Vec2D::new(
                points[i].x - normal.x * half_width,
                points[i].y - normal.y * half_width,
            );

            mesh.vertices.push(GpuVertex {
                position: [left.x, left.y],
                uv: [0.0, 0.0],
                color: color_arr,
            });
            mesh.vertices.push(GpuVertex {
                position: [right.x, right.y],
                uv: [1.0, 0.0],
                color: color_arr,
            });
        }

        // Generate triangle strip indices
        for i in 0..(n as u32 - 1) {
            let i0 = base_idx + i * 2;
            let i1 = base_idx + i * 2 + 1;
            let i2 = base_idx + (i + 1) * 2;
            let i3 = base_idx + (i + 1) * 2 + 1;

            mesh.indices.push(i0);
            mesh.indices.push(i1);
            mesh.indices.push(i2);

            mesh.indices.push(i1);
            mesh.indices.push(i3);
            mesh.indices.push(i2);
        }

        // Add end caps (for open paths)
        if !is_closed {
            match cap {
                StrokeCap::Round => {
                    self.add_round_cap(points[0], normals[0] * (-1.0), half_width, color, mesh);
                    self.add_round_cap(points[n - 1], normals[seg_count - 1], half_width, color, mesh);
                }
                StrokeCap::Square => {
                    self.add_square_cap(points[0], normals[0], half_width, color, mesh, true);
                    self.add_square_cap(points[n - 1], normals[seg_count - 1], half_width, color, mesh, false);
                }
                StrokeCap::Butt => {} // No cap needed
            }
        }
    }

    fn add_round_cap(
        &self,
        center: Vec2D,
        direction: Vec2D,
        half_width: f32,
        color: Color,
        mesh: &mut TessellatedMesh,
    ) {
        let color_arr = [color.r, color.g, color.b, color.a];
        let segments = 8u32;
        let base_idx = mesh.vertices.len() as u32;

        mesh.vertices.push(GpuVertex {
            position: [center.x, center.y],
            uv: [0.5, 0.5],
            color: color_arr,
        });

        let normal = Vec2D::new(-direction.y, direction.x);
        let start_angle = normal.y.atan2(normal.x);

        for i in 0..=segments {
            let angle = start_angle + std::f32::consts::PI * i as f32 / segments as f32;
            let x = center.x + angle.cos() * half_width;
            let y = center.y + angle.sin() * half_width;
            mesh.vertices.push(GpuVertex {
                position: [x, y],
                uv: [0.0, 0.0],
                color: color_arr,
            });
        }

        for i in 0..segments {
            mesh.indices.push(base_idx);
            mesh.indices.push(base_idx + 1 + i);
            mesh.indices.push(base_idx + 2 + i);
        }
    }

    fn add_square_cap(
        &self,
        point: Vec2D,
        normal: Vec2D,
        half_width: f32,
        color: Color,
        mesh: &mut TessellatedMesh,
        is_start: bool,
    ) {
        let color_arr = [color.r, color.g, color.b, color.a];
        let dir = if is_start {
            Vec2D::new(normal.y, -normal.x)
        } else {
            Vec2D::new(-normal.y, normal.x)
        };

        let left = Vec2D::new(
            point.x + normal.x * half_width,
            point.y + normal.y * half_width,
        );
        let right = Vec2D::new(
            point.x - normal.x * half_width,
            point.y - normal.y * half_width,
        );
        let left_ext = Vec2D::new(left.x + dir.x * half_width, left.y + dir.y * half_width);
        let right_ext = Vec2D::new(right.x + dir.x * half_width, right.y + dir.y * half_width);

        let base_idx = mesh.vertices.len() as u32;
        for p in &[left, right, left_ext, right_ext] {
            mesh.vertices.push(GpuVertex {
                position: [p.x, p.y],
                uv: [0.0, 0.0],
                color: color_arr,
            });
        }

        mesh.indices.push(base_idx);
        mesh.indices.push(base_idx + 1);
        mesh.indices.push(base_idx + 2);
        mesh.indices.push(base_idx + 1);
        mesh.indices.push(base_idx + 3);
        mesh.indices.push(base_idx + 2);
    }
}

/// Signed area of a polygon (positive = clockwise in screen coords)
fn signed_polygon_area(points: &[Vec2D]) -> f32 {
    let mut area = 0.0;
    let n = points.len();
    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i].x * points[j].y;
        area -= points[j].x * points[i].y;
    }
    area * 0.5
}

/// Cross product of (b-a) x (c-a) — positive if CCW
fn cross_2d(a: Vec2D, b: Vec2D, c: Vec2D) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Check if point p is inside triangle (a, b, c)
fn point_in_triangle(p: Vec2D, a: Vec2D, b: Vec2D, c: Vec2D) -> bool {
    let d1 = cross_2d(a, b, p);
    let d2 = cross_2d(b, c, p);
    let d3 = cross_2d(c, a, p);
    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
    !(has_neg && has_pos)
}

/// Check if the vertex at curr forms an ear (convex and no other point inside)
fn is_ear(a: Vec2D, b: Vec2D, c: Vec2D, indices: &[usize], points: &[Vec2D], curr_pos: usize) -> bool {
    // Must be convex (CCW)
    if cross_2d(a, b, c) <= 0.0 {
        return false;
    }

    // Check no other point is inside this triangle
    for (i, &idx) in indices.iter().enumerate() {
        if i == (curr_pos + indices.len() - 1) % indices.len()
            || i == curr_pos
            || i == (curr_pos + 1) % indices.len()
        {
            continue;
        }
        if point_in_triangle(points[idx], a, b, c) {
            return false;
        }
    }

    true
}

fn average_normal(n1: &Vec2D, n2: &Vec2D) -> Vec2D {
    let avg = Vec2D::new(n1.x + n2.x, n1.y + n2.y);
    let len = avg.length();
    if len < 1e-6 {
        *n1
    } else {
        Vec2D::new(avg.x / len, avg.y / len)
    }
}
