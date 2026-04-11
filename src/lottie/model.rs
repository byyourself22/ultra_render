use crate::geometry::math::{Color, Vec2D};

/// Root Lottie composition
#[derive(Clone, Debug)]
pub struct LottieComposition {
    pub version: String,
    pub width: f32,
    pub height: f32,
    pub frame_rate: f32,
    pub in_point: f32,
    pub out_point: f32,
    pub name: String,
    pub is_3d: bool,
    pub assets: Vec<LottieAsset>,
    pub layers: Vec<LottieLayer>,
    pub markers: Vec<LottieMarker>,
}

impl LottieComposition {
    pub fn duration_frames(&self) -> f32 {
        self.out_point - self.in_point
    }

    pub fn duration_seconds(&self) -> f32 {
        self.duration_frames() / self.frame_rate
    }
}

/// Asset (precomp, image, etc.)
#[derive(Clone, Debug)]
pub struct LottieAsset {
    pub id: String,
    pub name: String,
    pub layers: Vec<LottieLayer>,
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub path: Option<String>,
    pub filename: Option<String>,
    pub is_image: bool,
}

/// Timeline marker
#[derive(Clone, Debug)]
pub struct LottieMarker {
    pub name: String,
    pub time: f32,
    pub duration: f32,
}

/// Layer types (from ThorVG LottieLayer)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LayerType {
    Precomp = 0,
    Solid = 1,
    Image = 2,
    Null = 3,
    Shape = 4,
    Text = 5,
}

/// Blend modes
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum BlendMode {
    #[default]
    Normal = 0,
    Multiply = 1,
    Screen = 2,
    Overlay = 3,
    Darken = 4,
    Lighten = 5,
    ColorDodge = 6,
    ColorBurn = 7,
    HardLight = 8,
    SoftLight = 9,
    Difference = 10,
    Exclusion = 11,
    Hue = 12,
    Saturation = 13,
    ColorMode = 14,
    Luminosity = 15,
    Add = 16,
}

impl BlendMode {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Normal,
            1 => Self::Multiply,
            2 => Self::Screen,
            3 => Self::Overlay,
            4 => Self::Darken,
            5 => Self::Lighten,
            6 => Self::ColorDodge,
            7 => Self::ColorBurn,
            8 => Self::HardLight,
            9 => Self::SoftLight,
            10 => Self::Difference,
            11 => Self::Exclusion,
            12 => Self::Hue,
            13 => Self::Saturation,
            14 => Self::ColorMode,
            15 => Self::Luminosity,
            16 => Self::Add,
            _ => Self::Normal,
        }
    }
}

/// Matte type
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum MatteType {
    #[default]
    None = 0,
    Alpha = 1,
    InvertedAlpha = 2,
    Luma = 3,
    InvertedLuma = 4,
}

/// Layer in the composition tree
#[derive(Clone, Debug)]
pub struct LottieLayer {
    pub name: String,
    pub layer_type: LayerType,
    pub index: Option<i32>,
    pub parent_index: Option<i32>,
    pub in_point: f32,
    pub out_point: f32,
    pub start_time: f32,
    pub stretch: f32,
    pub blend_mode: BlendMode,
    pub matte_type: MatteType,
    pub is_3d: bool,
    pub auto_orient: bool,
    pub hidden: bool,
    pub transform: LottieTransform,
    pub shapes: Vec<ShapeItem>,
    pub masks: Vec<LottieMask>,
    pub effects: Vec<LottieEffect>,
    // Precomp reference
    pub ref_id: Option<String>,
    // Solid layer
    pub solid_color: Option<String>,
    pub solid_width: Option<f32>,
    pub solid_height: Option<f32>,
}

impl Default for LottieLayer {
    fn default() -> Self {
        Self {
            name: String::new(),
            layer_type: LayerType::Null,
            index: None,
            parent_index: None,
            in_point: 0.0,
            out_point: 0.0,
            start_time: 0.0,
            stretch: 1.0,
            blend_mode: BlendMode::Normal,
            matte_type: MatteType::None,
            is_3d: false,
            auto_orient: false,
            hidden: false,
            transform: LottieTransform::default(),
            shapes: Vec::new(),
            masks: Vec::new(),
            effects: Vec::new(),
            ref_id: None,
            solid_color: None,
            solid_width: None,
            solid_height: None,
        }
    }
}

/// Animated value with keyframes
#[derive(Clone, Debug)]
pub enum AnimatedValue<T: Clone + std::fmt::Debug> {
    Static(T),
    Animated(Vec<Keyframe<T>>),
}

impl<T: Clone + std::fmt::Debug + Default> Default for AnimatedValue<T> {
    fn default() -> Self {
        AnimatedValue::Static(T::default())
    }
}

impl<T: Clone + std::fmt::Debug> AnimatedValue<T> {
    pub fn is_animated(&self) -> bool {
        matches!(self, AnimatedValue::Animated(_))
    }

    pub fn static_value(&self) -> Option<&T> {
        match self {
            AnimatedValue::Static(v) => Some(v),
            _ => None,
        }
    }
}

/// A single keyframe (from ThorVG LottieScalarFrame/LottieVectorFrame)
#[derive(Clone, Debug)]
pub struct Keyframe<T: Clone + std::fmt::Debug> {
    pub time: f32,
    pub value: T,
    pub end_value: Option<T>,
    pub easing_in: EasingHandle,
    pub easing_out: EasingHandle,
    pub hold: bool,
    // Spatial tangents (for position)
    pub tan_in: Option<Vec2D>,
    pub tan_out: Option<Vec2D>,
}

/// Bezier easing handle
#[derive(Clone, Debug, Default)]
pub struct EasingHandle {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}

/// Transform properties (from ThorVG LottieTransform)
#[derive(Clone, Debug)]
pub struct LottieTransform {
    pub anchor: AnimatedValue<Vec2D>,
    pub position: AnimatedValue<Vec2D>,
    pub scale: AnimatedValue<Vec2D>,
    pub rotation: AnimatedValue<f32>,
    pub opacity: AnimatedValue<f32>,
    pub skew: AnimatedValue<f32>,
    pub skew_axis: AnimatedValue<f32>,
    // Separate position dimensions
    pub position_x: Option<AnimatedValue<f32>>,
    pub position_y: Option<AnimatedValue<f32>>,
}

impl Default for LottieTransform {
    fn default() -> Self {
        Self {
            anchor: AnimatedValue::Static(Vec2D::ZERO),
            position: AnimatedValue::Static(Vec2D::ZERO),
            scale: AnimatedValue::Static(Vec2D::new(100.0, 100.0)),
            rotation: AnimatedValue::Static(0.0),
            opacity: AnimatedValue::Static(100.0),
            skew: AnimatedValue::Static(0.0),
            skew_axis: AnimatedValue::Static(0.0),
            position_x: None,
            position_y: None,
        }
    }
}

/// Shape items (ThorVG LottieObject types)
#[derive(Clone, Debug)]
pub enum ShapeItem {
    Group(ShapeGroup),
    Rect(ShapeRect),
    Ellipse(ShapeEllipse),
    Path(ShapePath),
    Polystar(ShapePolystar),
    Fill(ShapeFill),
    Stroke(ShapeStroke),
    GradientFill(ShapeGradientFill),
    GradientStroke(ShapeGradientStroke),
    Transform(LottieTransform),
    TrimPath(ShapeTrimPath),
    RoundedCorners(ShapeRoundedCorners),
    Repeater(ShapeRepeater),
    MergePaths(ShapeMergePaths),
}

/// Group of shapes
#[derive(Clone, Debug)]
pub struct ShapeGroup {
    pub name: String,
    pub items: Vec<ShapeItem>,
    pub blend_mode: BlendMode,
    pub hidden: bool,
}

/// Rectangle shape
#[derive(Clone, Debug)]
pub struct ShapeRect {
    pub name: String,
    pub position: AnimatedValue<Vec2D>,
    pub size: AnimatedValue<Vec2D>,
    pub roundness: AnimatedValue<f32>,
    pub hidden: bool,
}

/// Ellipse shape
#[derive(Clone, Debug)]
pub struct ShapeEllipse {
    pub name: String,
    pub position: AnimatedValue<Vec2D>,
    pub size: AnimatedValue<Vec2D>,
    pub hidden: bool,
}

/// Bezier path shape
#[derive(Clone, Debug)]
pub struct ShapePath {
    pub name: String,
    pub shape: AnimatedValue<BezierPath>,
    pub hidden: bool,
}

/// Bezier path data
#[derive(Clone, Debug, Default)]
pub struct BezierPath {
    pub vertices: Vec<Vec2D>,
    pub in_tangents: Vec<Vec2D>,
    pub out_tangents: Vec<Vec2D>,
    pub closed: bool,
}

/// Polystar (star/polygon)
#[derive(Clone, Debug)]
pub struct ShapePolystar {
    pub name: String,
    pub star_type: PolystarType,
    pub position: AnimatedValue<Vec2D>,
    pub points: AnimatedValue<f32>,
    pub rotation: AnimatedValue<f32>,
    pub outer_radius: AnimatedValue<f32>,
    pub outer_roundness: AnimatedValue<f32>,
    pub inner_radius: AnimatedValue<f32>,
    pub inner_roundness: AnimatedValue<f32>,
    pub hidden: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PolystarType {
    Star = 1,
    Polygon = 2,
}

/// Solid fill
#[derive(Clone, Debug)]
pub struct ShapeFill {
    pub name: String,
    pub color: AnimatedValue<Color>,
    pub opacity: AnimatedValue<f32>,
    pub fill_rule: crate::geometry::path::FillRule,
    pub hidden: bool,
}

/// Solid stroke
#[derive(Clone, Debug)]
pub struct ShapeStroke {
    pub name: String,
    pub color: AnimatedValue<Color>,
    pub opacity: AnimatedValue<f32>,
    pub width: AnimatedValue<f32>,
    pub line_cap: crate::geometry::path::StrokeCap,
    pub line_join: crate::geometry::path::StrokeJoin,
    pub miter_limit: f32,
    pub dashes: Vec<StrokeDash>,
    pub hidden: bool,
}

#[derive(Clone, Debug)]
pub struct StrokeDash {
    pub name: String,
    pub value: AnimatedValue<f32>,
    pub dash_type: StrokeDashType,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StrokeDashType {
    Dash,
    Gap,
    Offset,
}

/// Gradient fill
#[derive(Clone, Debug)]
pub struct ShapeGradientFill {
    pub name: String,
    pub gradient_type: GradientType,
    pub start_point: AnimatedValue<Vec2D>,
    pub end_point: AnimatedValue<Vec2D>,
    pub colors: AnimatedValue<GradientColors>,
    pub opacity: AnimatedValue<f32>,
    pub fill_rule: crate::geometry::path::FillRule,
    pub hidden: bool,
}

/// Gradient stroke
#[derive(Clone, Debug)]
pub struct ShapeGradientStroke {
    pub name: String,
    pub gradient_type: GradientType,
    pub start_point: AnimatedValue<Vec2D>,
    pub end_point: AnimatedValue<Vec2D>,
    pub colors: AnimatedValue<GradientColors>,
    pub opacity: AnimatedValue<f32>,
    pub width: AnimatedValue<f32>,
    pub line_cap: crate::geometry::path::StrokeCap,
    pub line_join: crate::geometry::path::StrokeJoin,
    pub miter_limit: f32,
    pub hidden: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradientType {
    Linear = 1,
    Radial = 2,
}

#[derive(Clone, Debug, Default)]
pub struct GradientColors {
    pub color_count: usize,
    pub colors: Vec<f32>, // [offset, r, g, b, ...] repeated
}

/// Trim path modifier
#[derive(Clone, Debug)]
pub struct ShapeTrimPath {
    pub name: String,
    pub start: AnimatedValue<f32>,
    pub end: AnimatedValue<f32>,
    pub offset: AnimatedValue<f32>,
    pub trim_type: TrimPathType,
    pub hidden: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrimPathType {
    Simultaneously = 1,
    Individually = 2,
}

/// Rounded corners modifier
#[derive(Clone, Debug)]
pub struct ShapeRoundedCorners {
    pub name: String,
    pub radius: AnimatedValue<f32>,
    pub hidden: bool,
}

/// Repeater modifier
#[derive(Clone, Debug)]
pub struct ShapeRepeater {
    pub name: String,
    pub copies: AnimatedValue<f32>,
    pub offset: AnimatedValue<f32>,
    pub transform: LottieTransform,
    pub hidden: bool,
}

/// Merge paths
#[derive(Clone, Debug)]
pub struct ShapeMergePaths {
    pub name: String,
    pub mode: u32,
    pub hidden: bool,
}

/// Mask
#[derive(Clone, Debug)]
pub struct LottieMask {
    pub mode: MaskMode,
    pub shape: AnimatedValue<BezierPath>,
    pub opacity: AnimatedValue<f32>,
    pub expand: AnimatedValue<f32>,
    pub inverted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaskMode {
    None,
    Add,
    Subtract,
    Intersect,
    Lighten,
    Darken,
    Difference,
}

/// Effect types (from ThorVG)
#[derive(Clone, Debug)]
pub enum LottieEffect {
    GaussianBlur {
        blurriness: AnimatedValue<f32>,
        direction: AnimatedValue<f32>,
        wrap: AnimatedValue<f32>,
    },
    DropShadow {
        color: AnimatedValue<Color>,
        opacity: AnimatedValue<f32>,
        angle: AnimatedValue<f32>,
        distance: AnimatedValue<f32>,
        blur: AnimatedValue<f32>,
    },
    Tint {
        black: AnimatedValue<Color>,
        white: AnimatedValue<Color>,
        intensity: AnimatedValue<f32>,
    },
    Tritone {
        bright: AnimatedValue<Color>,
        midtone: AnimatedValue<Color>,
        dark: AnimatedValue<Color>,
    },
    Fill {
        color: AnimatedValue<Color>,
        opacity: AnimatedValue<f32>,
    },
}
