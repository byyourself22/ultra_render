use serde_json::Value;
use crate::geometry::math::{Vec2D, Color};
use crate::geometry::path::{FillRule, StrokeCap, StrokeJoin};
use super::model::*;

/// Parse a Lottie JSON string into a LottieComposition
pub fn parse_lottie(json_str: &str) -> Result<LottieComposition, String> {
    let root: Value = serde_json::from_str(json_str)
        .map_err(|e| format!("JSON parse error: {}", e))?;
    parse_composition(&root)
}

/// Parse a Lottie JSON value into a LottieComposition
pub fn parse_composition(root: &Value) -> Result<LottieComposition, String> {
    let version = root["v"].as_str().unwrap_or("0.0.0").to_string();
    let width = root["w"].as_f64().unwrap_or(0.0) as f32;
    let height = root["h"].as_f64().unwrap_or(0.0) as f32;
    let frame_rate = root["fr"].as_f64().unwrap_or(30.0) as f32;
    let in_point = root["ip"].as_f64().unwrap_or(0.0) as f32;
    let out_point = root["op"].as_f64().unwrap_or(0.0) as f32;
    let name = root["nm"].as_str().unwrap_or("").to_string();
    let is_3d = root["ddd"].as_i64().unwrap_or(0) != 0;

    let assets = parse_assets(&root["assets"]);
    let layers = parse_layers(&root["layers"]);
    let markers = parse_markers(&root["markers"]);

    Ok(LottieComposition {
        version,
        width,
        height,
        frame_rate,
        in_point,
        out_point,
        name,
        is_3d,
        assets,
        layers,
        markers,
    })
}

fn parse_assets(val: &Value) -> Vec<LottieAsset> {
    let Some(arr) = val.as_array() else { return Vec::new() };
    arr.iter().map(|a| {
        let id = a["id"].as_str().unwrap_or("").to_string();
        let name = a["nm"].as_str().unwrap_or("").to_string();
        let width = a["w"].as_f64().map(|v| v as f32);
        let height = a["h"].as_f64().map(|v| v as f32);
        let path = a["u"].as_str().map(|s| s.to_string());
        let filename = a["p"].as_str().map(|s| s.to_string());
        let is_image = filename.is_some() && !a.get("layers").is_some();
        let layers = parse_layers(&a["layers"]);

        LottieAsset {
            id,
            name,
            layers,
            width,
            height,
            path,
            filename,
            is_image,
        }
    }).collect()
}

fn parse_markers(val: &Value) -> Vec<LottieMarker> {
    let Some(arr) = val.as_array() else { return Vec::new() };
    arr.iter().map(|m| {
        LottieMarker {
            name: m["cm"].as_str().unwrap_or("").to_string(),
            time: m["tm"].as_f64().unwrap_or(0.0) as f32,
            duration: m["dr"].as_f64().unwrap_or(0.0) as f32,
        }
    }).collect()
}

fn parse_layers(val: &Value) -> Vec<LottieLayer> {
    let Some(arr) = val.as_array() else { return Vec::new() };
    arr.iter().map(|l| parse_layer(l)).collect()
}

fn parse_layer(val: &Value) -> LottieLayer {
    let layer_type = match val["ty"].as_i64().unwrap_or(3) {
        0 => LayerType::Precomp,
        1 => LayerType::Solid,
        2 => LayerType::Image,
        3 => LayerType::Null,
        4 => LayerType::Shape,
        5 => LayerType::Text,
        _ => LayerType::Null,
    };

    let blend_mode = BlendMode::from_u32(val["bm"].as_u64().unwrap_or(0) as u32);
    let matte_type = match val["tt"].as_i64().unwrap_or(0) {
        1 => MatteType::Alpha,
        2 => MatteType::InvertedAlpha,
        3 => MatteType::Luma,
        4 => MatteType::InvertedLuma,
        _ => MatteType::None,
    };

    let transform = parse_transform(&val["ks"]);
    let shapes = parse_shapes(&val["shapes"]);
    let masks = parse_masks(&val["masksProperties"]);
    let effects = parse_effects(&val["ef"]);

    LottieLayer {
        name: val["nm"].as_str().unwrap_or("").to_string(),
        layer_type,
        index: val["ind"].as_i64().map(|v| v as i32),
        parent_index: val["parent"].as_i64().map(|v| v as i32),
        in_point: val["ip"].as_f64().unwrap_or(0.0) as f32,
        out_point: val["op"].as_f64().unwrap_or(0.0) as f32,
        start_time: val["st"].as_f64().unwrap_or(0.0) as f32,
        stretch: val["sr"].as_f64().unwrap_or(1.0) as f32,
        blend_mode,
        matte_type,
        is_3d: val["ddd"].as_i64().unwrap_or(0) != 0,
        auto_orient: val["ao"].as_i64().unwrap_or(0) != 0,
        hidden: val["hd"].as_bool().unwrap_or(false),
        transform,
        shapes,
        masks,
        effects,
        ref_id: val["refId"].as_str().map(|s| s.to_string()),
        solid_color: val["sc"].as_str().map(|s| s.to_string()),
        solid_width: val["sw"].as_f64().map(|v| v as f32),
        solid_height: val["sh"].as_f64().map(|v| v as f32),
    }
}

// ─── Transform ───────────────────────────────────────────────

fn parse_transform(val: &Value) -> LottieTransform {
    if val.is_null() {
        return LottieTransform::default();
    }

    // Check for separate position dimensions
    let has_separate_pos = val["p"]["s"].as_bool().unwrap_or(false)
        || (val["p"]["x"].is_object() || val["p"]["y"].is_object());

    let (position, position_x, position_y) = if has_separate_pos {
        (
            AnimatedValue::Static(Vec2D::ZERO),
            Some(parse_animated_f32(&val["p"]["x"])),
            Some(parse_animated_f32(&val["p"]["y"])),
        )
    } else {
        (parse_animated_vec2d(&val["p"]), None, None)
    };

    LottieTransform {
        anchor: parse_animated_vec2d(&val["a"]),
        position,
        scale: parse_animated_vec2d_with_default(&val["s"], Vec2D::new(100.0, 100.0)),
        rotation: parse_animated_f32(&val["r"]),
        opacity: parse_animated_f32_with_default(&val["o"], 100.0),
        skew: parse_animated_f32(&val["sk"]),
        skew_axis: parse_animated_f32(&val["sa"]),
        position_x,
        position_y,
    }
}

// ─── Animated properties ─────────────────────────────────────

fn parse_animated_f32(val: &Value) -> AnimatedValue<f32> {
    parse_animated_f32_with_default(val, 0.0)
}

fn parse_animated_f32_with_default(val: &Value, default: f32) -> AnimatedValue<f32> {
    if val.is_null() {
        return AnimatedValue::Static(default);
    }

    let animated = val["a"].as_i64().unwrap_or(0) != 0;
    let k = &val["k"];

    if !animated {
        let v = extract_f32(k, default);
        return AnimatedValue::Static(v);
    }

    let Some(arr) = k.as_array() else {
        return AnimatedValue::Static(default);
    };

    let keyframes = arr.iter().filter_map(|kf| {
        let time = kf["t"].as_f64()? as f32;
        let value = extract_f32_from_keyframe_value(&kf["s"], default);
        let end_value = if kf.get("e").is_some() {
            Some(extract_f32_from_keyframe_value(&kf["e"], default))
        } else {
            None
        };
        let hold = kf["h"].as_i64().unwrap_or(0) != 0;
        let easing_in = parse_easing_handle(&kf["i"]);
        let easing_out = parse_easing_handle(&kf["o"]);

        Some(Keyframe {
            time,
            value,
            end_value,
            easing_in,
            easing_out,
            hold,
            tan_in: None,
            tan_out: None,
        })
    }).collect::<Vec<_>>();

    if keyframes.is_empty() {
        AnimatedValue::Static(default)
    } else {
        AnimatedValue::Animated(keyframes)
    }
}

fn parse_animated_vec2d(val: &Value) -> AnimatedValue<Vec2D> {
    parse_animated_vec2d_with_default(val, Vec2D::ZERO)
}

fn parse_animated_vec2d_with_default(val: &Value, default: Vec2D) -> AnimatedValue<Vec2D> {
    if val.is_null() {
        return AnimatedValue::Static(default);
    }

    let animated = val["a"].as_i64().unwrap_or(0) != 0;
    let k = &val["k"];

    if !animated {
        let v = extract_vec2d(k, default);
        return AnimatedValue::Static(v);
    }

    let Some(arr) = k.as_array() else {
        return AnimatedValue::Static(default);
    };

    let keyframes = arr.iter().filter_map(|kf| {
        let time = kf["t"].as_f64()? as f32;
        let value = extract_vec2d_from_arr(&kf["s"], default);
        let end_value = if kf.get("e").is_some() {
            Some(extract_vec2d_from_arr(&kf["e"], default))
        } else {
            None
        };
        let hold = kf["h"].as_i64().unwrap_or(0) != 0;
        let easing_in = parse_easing_handle(&kf["i"]);
        let easing_out = parse_easing_handle(&kf["o"]);

        let tan_in = parse_spatial_tangent(&kf["ti"]);
        let tan_out = parse_spatial_tangent(&kf["to"]);

        Some(Keyframe {
            time,
            value,
            end_value,
            easing_in,
            easing_out,
            hold,
            tan_in,
            tan_out,
        })
    }).collect::<Vec<_>>();

    if keyframes.is_empty() {
        AnimatedValue::Static(default)
    } else {
        AnimatedValue::Animated(keyframes)
    }
}

fn parse_animated_color(val: &Value) -> AnimatedValue<Color> {
    if val.is_null() {
        return AnimatedValue::Static(Color::WHITE);
    }

    let animated = val["a"].as_i64().unwrap_or(0) != 0;
    let k = &val["k"];

    if !animated {
        let c = extract_color(k);
        return AnimatedValue::Static(c);
    }

    let Some(arr) = k.as_array() else {
        return AnimatedValue::Static(Color::WHITE);
    };

    let keyframes = arr.iter().filter_map(|kf| {
        let time = kf["t"].as_f64()? as f32;
        let value = extract_color_from_arr(&kf["s"]);
        let end_value = if kf.get("e").is_some() {
            Some(extract_color_from_arr(&kf["e"]))
        } else {
            None
        };
        let hold = kf["h"].as_i64().unwrap_or(0) != 0;
        let easing_in = parse_easing_handle(&kf["i"]);
        let easing_out = parse_easing_handle(&kf["o"]);

        Some(Keyframe {
            time,
            value,
            end_value,
            easing_in,
            easing_out,
            hold,
            tan_in: None,
            tan_out: None,
        })
    }).collect::<Vec<_>>();

    if keyframes.is_empty() {
        AnimatedValue::Static(Color::WHITE)
    } else {
        AnimatedValue::Animated(keyframes)
    }
}

fn parse_animated_bezier_path(val: &Value) -> AnimatedValue<BezierPath> {
    if val.is_null() {
        return AnimatedValue::Static(BezierPath::default());
    }

    let animated = val["a"].as_i64().unwrap_or(0) != 0;
    let k = &val["k"];

    if !animated {
        let path = parse_bezier_shape(k);
        return AnimatedValue::Static(path);
    }

    let Some(arr) = k.as_array() else {
        return AnimatedValue::Static(BezierPath::default());
    };

    let keyframes = arr.iter().filter_map(|kf| {
        let time = kf["t"].as_f64()? as f32;
        let value = if let Some(s_arr) = kf["s"].as_array() {
            if s_arr.len() == 1 {
                parse_bezier_shape(&s_arr[0])
            } else {
                parse_bezier_shape(&kf["s"])
            }
        } else {
            parse_bezier_shape(&kf["s"])
        };
        let end_value = if kf.get("e").is_some() {
            let ev = if let Some(e_arr) = kf["e"].as_array() {
                if e_arr.len() == 1 && e_arr[0].is_object() {
                    parse_bezier_shape(&e_arr[0])
                } else {
                    parse_bezier_shape(&kf["e"])
                }
            } else {
                parse_bezier_shape(&kf["e"])
            };
            Some(ev)
        } else {
            None
        };
        let hold = kf["h"].as_i64().unwrap_or(0) != 0;
        let easing_in = parse_easing_handle(&kf["i"]);
        let easing_out = parse_easing_handle(&kf["o"]);

        Some(Keyframe {
            time,
            value,
            end_value,
            easing_in,
            easing_out,
            hold,
            tan_in: None,
            tan_out: None,
        })
    }).collect::<Vec<_>>();

    if keyframes.is_empty() {
        AnimatedValue::Static(BezierPath::default())
    } else {
        AnimatedValue::Animated(keyframes)
    }
}

fn parse_animated_gradient_colors(val: &Value) -> AnimatedValue<GradientColors> {
    if val.is_null() {
        return AnimatedValue::Static(GradientColors::default());
    }

    let color_count = val["p"].as_u64().unwrap_or(2) as usize;
    let k_val = &val["k"];

    let animated = k_val["a"].as_i64().unwrap_or(0) != 0;
    let k = &k_val["k"];

    if !animated {
        let colors = extract_f32_array(k);
        return AnimatedValue::Static(GradientColors { color_count, colors });
    }

    let Some(arr) = k.as_array() else {
        return AnimatedValue::Static(GradientColors::default());
    };

    let keyframes = arr.iter().filter_map(|kf| {
        let time = kf["t"].as_f64()? as f32;
        let colors = extract_f32_array(&kf["s"]);
        let end_colors = if kf.get("e").is_some() {
            Some(GradientColors {
                color_count,
                colors: extract_f32_array(&kf["e"]),
            })
        } else {
            None
        };
        let hold = kf["h"].as_i64().unwrap_or(0) != 0;
        let easing_in = parse_easing_handle(&kf["i"]);
        let easing_out = parse_easing_handle(&kf["o"]);

        Some(Keyframe {
            time,
            value: GradientColors { color_count, colors },
            end_value: end_colors,
            easing_in,
            easing_out,
            hold,
            tan_in: None,
            tan_out: None,
        })
    }).collect::<Vec<_>>();

    if keyframes.is_empty() {
        AnimatedValue::Static(GradientColors::default())
    } else {
        AnimatedValue::Animated(keyframes)
    }
}

// ─── Shapes ──────────────────────────────────────────────────

fn parse_shapes(val: &Value) -> Vec<ShapeItem> {
    let Some(arr) = val.as_array() else { return Vec::new() };
    arr.iter().filter_map(|s| parse_shape_item(s)).collect()
}

fn parse_shape_item(val: &Value) -> Option<ShapeItem> {
    let ty = val["ty"].as_str().unwrap_or("");
    let hidden = val["hd"].as_bool().unwrap_or(false);
    let name = val["nm"].as_str().unwrap_or("").to_string();

    match ty {
        "gr" => {
            let items = parse_shapes(&val["it"]);
            let blend_mode = BlendMode::from_u32(val["bm"].as_u64().unwrap_or(0) as u32);
            Some(ShapeItem::Group(ShapeGroup {
                name,
                items,
                blend_mode,
                hidden,
            }))
        }
        "rc" => {
            Some(ShapeItem::Rect(ShapeRect {
                name,
                position: parse_animated_vec2d(&val["p"]),
                size: parse_animated_vec2d(&val["s"]),
                roundness: parse_animated_f32(&val["r"]),
                hidden,
            }))
        }
        "el" => {
            Some(ShapeItem::Ellipse(ShapeEllipse {
                name,
                position: parse_animated_vec2d(&val["p"]),
                size: parse_animated_vec2d(&val["s"]),
                hidden,
            }))
        }
        "sh" => {
            Some(ShapeItem::Path(ShapePath {
                name,
                shape: parse_animated_bezier_path(&val["ks"]),
                hidden,
            }))
        }
        "sr" => {
            let star_type = match val["sy"].as_i64().unwrap_or(1) {
                2 => PolystarType::Polygon,
                _ => PolystarType::Star,
            };
            Some(ShapeItem::Polystar(ShapePolystar {
                name,
                star_type,
                position: parse_animated_vec2d(&val["p"]),
                points: parse_animated_f32(&val["pt"]),
                rotation: parse_animated_f32(&val["r"]),
                outer_radius: parse_animated_f32(&val["or"]),
                outer_roundness: parse_animated_f32(&val["os"]),
                inner_radius: parse_animated_f32(&val["ir"]),
                inner_roundness: parse_animated_f32(&val["is"]),
                hidden,
            }))
        }
        "fl" => {
            let fill_rule = match val["r"].as_i64().unwrap_or(1) {
                2 => FillRule::EvenOdd,
                _ => FillRule::NonZero,
            };
            Some(ShapeItem::Fill(ShapeFill {
                name,
                color: parse_animated_color(&val["c"]),
                opacity: parse_animated_f32_with_default(&val["o"], 100.0),
                fill_rule,
                hidden,
            }))
        }
        "st" => {
            Some(ShapeItem::Stroke(ShapeStroke {
                name,
                color: parse_animated_color(&val["c"]),
                opacity: parse_animated_f32_with_default(&val["o"], 100.0),
                width: parse_animated_f32(&val["w"]),
                line_cap: parse_stroke_cap(val["lc"].as_i64().unwrap_or(1) as u32),
                line_join: parse_stroke_join(val["lj"].as_i64().unwrap_or(1) as u32),
                miter_limit: val["ml"].as_f64().unwrap_or(4.0) as f32,
                dashes: parse_stroke_dashes(&val["d"]),
                hidden,
            }))
        }
        "gf" => {
            let gradient_type = match val["t"].as_i64().unwrap_or(1) {
                2 => GradientType::Radial,
                _ => GradientType::Linear,
            };
            let fill_rule = match val["r"].as_i64().unwrap_or(1) {
                2 => FillRule::EvenOdd,
                _ => FillRule::NonZero,
            };
            Some(ShapeItem::GradientFill(ShapeGradientFill {
                name,
                gradient_type,
                start_point: parse_animated_vec2d(&val["s"]),
                end_point: parse_animated_vec2d(&val["e"]),
                colors: parse_animated_gradient_colors(&val["g"]),
                opacity: parse_animated_f32_with_default(&val["o"], 100.0),
                fill_rule,
                hidden,
            }))
        }
        "gs" => {
            let gradient_type = match val["t"].as_i64().unwrap_or(1) {
                2 => GradientType::Radial,
                _ => GradientType::Linear,
            };
            Some(ShapeItem::GradientStroke(ShapeGradientStroke {
                name,
                gradient_type,
                start_point: parse_animated_vec2d(&val["s"]),
                end_point: parse_animated_vec2d(&val["e"]),
                colors: parse_animated_gradient_colors(&val["g"]),
                opacity: parse_animated_f32_with_default(&val["o"], 100.0),
                width: parse_animated_f32(&val["w"]),
                line_cap: parse_stroke_cap(val["lc"].as_i64().unwrap_or(1) as u32),
                line_join: parse_stroke_join(val["lj"].as_i64().unwrap_or(1) as u32),
                miter_limit: val["ml"].as_f64().unwrap_or(4.0) as f32,
                hidden,
            }))
        }
        "tr" => {
            Some(ShapeItem::Transform(parse_transform(val)))
        }
        "tm" => {
            let trim_type = match val["m"].as_i64().unwrap_or(1) {
                2 => TrimPathType::Individually,
                _ => TrimPathType::Simultaneously,
            };
            Some(ShapeItem::TrimPath(ShapeTrimPath {
                name,
                start: parse_animated_f32(&val["s"]),
                end: parse_animated_f32_with_default(&val["e"], 100.0),
                offset: parse_animated_f32(&val["o"]),
                trim_type,
                hidden,
            }))
        }
        "rd" => {
            Some(ShapeItem::RoundedCorners(ShapeRoundedCorners {
                name,
                radius: parse_animated_f32(&val["r"]),
                hidden,
            }))
        }
        "rp" => {
            Some(ShapeItem::Repeater(ShapeRepeater {
                name,
                copies: parse_animated_f32(&val["c"]),
                offset: parse_animated_f32(&val["o"]),
                transform: parse_transform(&val["tr"]),
                hidden,
            }))
        }
        "mm" => {
            Some(ShapeItem::MergePaths(ShapeMergePaths {
                name,
                mode: val["mm"].as_u64().unwrap_or(1) as u32,
                hidden,
            }))
        }
        _ => None,
    }
}

// ─── Masks ───────────────────────────────────────────────────

fn parse_masks(val: &Value) -> Vec<LottieMask> {
    let Some(arr) = val.as_array() else { return Vec::new() };
    arr.iter().map(|m| {
        let mode = match m["mode"].as_str().unwrap_or("a") {
            "n" => MaskMode::None,
            "a" => MaskMode::Add,
            "s" => MaskMode::Subtract,
            "i" => MaskMode::Intersect,
            "l" => MaskMode::Lighten,
            "d" => MaskMode::Darken,
            "f" => MaskMode::Difference,
            _ => MaskMode::Add,
        };
        LottieMask {
            mode,
            shape: parse_animated_bezier_path(&m["pt"]),
            opacity: parse_animated_f32_with_default(&m["o"], 100.0),
            expand: parse_animated_f32(&m["x"]),
            inverted: m["inv"].as_bool().unwrap_or(false),
        }
    }).collect()
}

// ─── Effects ─────────────────────────────────────────────────

fn parse_effects(val: &Value) -> Vec<LottieEffect> {
    let Some(arr) = val.as_array() else { return Vec::new() };
    arr.iter().filter_map(|e| parse_effect(e)).collect()
}

fn parse_effect(val: &Value) -> Option<LottieEffect> {
    let ty = val["ty"].as_i64().unwrap_or(0);
    let params = &val["ef"];

    match ty {
        29 => {
            // Gaussian Blur
            Some(LottieEffect::GaussianBlur {
                blurriness: get_effect_param_f32(params, 0),
                direction: get_effect_param_f32(params, 1),
                wrap: get_effect_param_f32(params, 2),
            })
        }
        25 => {
            // Drop Shadow
            Some(LottieEffect::DropShadow {
                color: get_effect_param_color(params, 0),
                opacity: get_effect_param_f32(params, 1),
                angle: get_effect_param_f32(params, 2),
                distance: get_effect_param_f32(params, 3),
                blur: get_effect_param_f32(params, 4),
            })
        }
        20 => {
            // Tint
            Some(LottieEffect::Tint {
                black: get_effect_param_color(params, 0),
                white: get_effect_param_color(params, 1),
                intensity: get_effect_param_f32(params, 2),
            })
        }
        23 => {
            // Tritone
            Some(LottieEffect::Tritone {
                bright: get_effect_param_color(params, 0),
                midtone: get_effect_param_color(params, 1),
                dark: get_effect_param_color(params, 2),
            })
        }
        21 => {
            // Fill
            Some(LottieEffect::Fill {
                color: get_effect_param_color(params, 2),
                opacity: get_effect_param_f32(params, 6),
            })
        }
        _ => None,
    }
}

fn get_effect_param_f32(params: &Value, index: usize) -> AnimatedValue<f32> {
    let Some(arr) = params.as_array() else { return AnimatedValue::Static(0.0) };
    if index < arr.len() {
        parse_animated_f32(&arr[index]["v"])
    } else {
        AnimatedValue::Static(0.0)
    }
}

fn get_effect_param_color(params: &Value, index: usize) -> AnimatedValue<Color> {
    let Some(arr) = params.as_array() else { return AnimatedValue::Static(Color::WHITE) };
    if index < arr.len() {
        parse_animated_color(&arr[index]["v"])
    } else {
        AnimatedValue::Static(Color::WHITE)
    }
}

// ─── Bezier shape parsing ────────────────────────────────────

fn parse_bezier_shape(val: &Value) -> BezierPath {
    if val.is_null() {
        return BezierPath::default();
    }

    let vertices = parse_vec2d_array(&val["v"]);
    let in_tangents = parse_vec2d_array(&val["i"]);
    let out_tangents = parse_vec2d_array(&val["o"]);
    let closed = val["c"].as_bool().unwrap_or(false);

    BezierPath {
        vertices,
        in_tangents,
        out_tangents,
        closed,
    }
}

fn parse_vec2d_array(val: &Value) -> Vec<Vec2D> {
    let Some(arr) = val.as_array() else { return Vec::new() };
    arr.iter().map(|v| {
        let x = v.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let y = v.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        Vec2D::new(x, y)
    }).collect()
}

// ─── Stroke dashes ───────────────────────────────────────────

fn parse_stroke_dashes(val: &Value) -> Vec<StrokeDash> {
    let Some(arr) = val.as_array() else { return Vec::new() };
    arr.iter().map(|d| {
        let dash_type = match d["n"].as_str().unwrap_or("d") {
            "d" => StrokeDashType::Dash,
            "g" => StrokeDashType::Gap,
            "o" => StrokeDashType::Offset,
            _ => StrokeDashType::Dash,
        };
        StrokeDash {
            name: d["nm"].as_str().unwrap_or("").to_string(),
            value: parse_animated_f32(&d["v"]),
            dash_type,
        }
    }).collect()
}

// ─── Easing ──────────────────────────────────────────────────

fn parse_easing_handle(val: &Value) -> EasingHandle {
    if val.is_null() {
        return EasingHandle::default();
    }
    EasingHandle {
        x: parse_easing_component(&val["x"]),
        y: parse_easing_component(&val["y"]),
    }
}

fn parse_easing_component(val: &Value) -> Vec<f32> {
    if let Some(n) = val.as_f64() {
        vec![n as f32]
    } else if let Some(arr) = val.as_array() {
        arr.iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect()
    } else {
        vec![0.0]
    }
}

fn parse_spatial_tangent(val: &Value) -> Option<Vec2D> {
    let arr = val.as_array()?;
    if arr.len() >= 2 {
        Some(Vec2D::new(
            arr[0].as_f64().unwrap_or(0.0) as f32,
            arr[1].as_f64().unwrap_or(0.0) as f32,
        ))
    } else {
        None
    }
}

// ─── Value extraction helpers ────────────────────────────────

fn extract_f32(val: &Value, default: f32) -> f32 {
    if let Some(n) = val.as_f64() {
        n as f32
    } else if let Some(arr) = val.as_array() {
        arr.first().and_then(|v| v.as_f64()).unwrap_or(default as f64) as f32
    } else {
        default
    }
}

fn extract_f32_from_keyframe_value(val: &Value, default: f32) -> f32 {
    if let Some(n) = val.as_f64() {
        n as f32
    } else if let Some(arr) = val.as_array() {
        arr.first().and_then(|v| v.as_f64()).unwrap_or(default as f64) as f32
    } else {
        default
    }
}

fn extract_vec2d(val: &Value, default: Vec2D) -> Vec2D {
    if let Some(arr) = val.as_array() {
        let x = arr.get(0).and_then(|v| v.as_f64()).unwrap_or(default.x as f64) as f32;
        let y = arr.get(1).and_then(|v| v.as_f64()).unwrap_or(default.y as f64) as f32;
        Vec2D::new(x, y)
    } else {
        default
    }
}

fn extract_vec2d_from_arr(val: &Value, default: Vec2D) -> Vec2D {
    extract_vec2d(val, default)
}

fn extract_color(val: &Value) -> Color {
    if let Some(arr) = val.as_array() {
        let r = arr.get(0).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let g = arr.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let b = arr.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let a = arr.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        Color::new(r, g, b, a)
    } else {
        Color::WHITE
    }
}

fn extract_color_from_arr(val: &Value) -> Color {
    extract_color(val)
}

fn extract_f32_array(val: &Value) -> Vec<f32> {
    let Some(arr) = val.as_array() else { return Vec::new() };
    arr.iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect()
}

fn parse_stroke_cap(v: u32) -> StrokeCap {
    match v {
        1 => StrokeCap::Butt,
        2 => StrokeCap::Round,
        3 => StrokeCap::Square,
        _ => StrokeCap::Butt,
    }
}

fn parse_stroke_join(v: u32) -> StrokeJoin {
    match v {
        1 => StrokeJoin::Miter,
        2 => StrokeJoin::Round,
        3 => StrokeJoin::Bevel,
        _ => StrokeJoin::Miter,
    }
}

// ─── File loading helpers ────────────────────────────────────

/// Load a Lottie composition from a .json file
#[cfg(not(target_arch = "wasm32"))]
pub fn load_from_file(path: &str) -> Result<LottieComposition, String> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file {}: {}", path, e))?;
    parse_lottie(&data)
}

/// Load a Lottie composition from a .lottie (ZIP) file
#[cfg(feature = "native")]
pub fn load_from_dotlottie(path: &str) -> Result<LottieComposition, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open file {}: {}", path, e))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| format!("Failed to read ZIP: {}", e))?;

    // Collect file names first to avoid borrow issues
    let file_names: Vec<String> = archive.file_names().map(|n| n.to_string()).collect();
    let has_manifest = file_names.iter().any(|n| n == "manifest.json");

    let json_name = if has_manifest {
        let manifest = archive.by_name("manifest.json")
            .map_err(|e| format!("Failed to read manifest: {}", e))?;
        let manifest_val: Value = serde_json::from_reader(manifest)
            .map_err(|e| format!("Failed to parse manifest: {}", e))?;
        manifest_val["animations"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|a| a["id"].as_str())
            .map(|id| format!("animations/{}.json", id))
            .unwrap_or_else(|| "animations/0.json".to_string())
    } else {
        file_names.iter()
            .find(|n| n.ends_with(".json") && !n.contains("manifest"))
            .cloned()
            .ok_or_else(|| "No animation JSON found in .lottie file".to_string())?
    };

    let json_file = archive.by_name(&json_name)
        .map_err(|e| format!("Failed to read {} from ZIP: {}", json_name, e))?;
    let root: Value = serde_json::from_reader(json_file)
        .map_err(|e| format!("Failed to parse animation JSON: {}", e))?;

    parse_composition(&root)
}

/// Load from either .json or .lottie based on extension
#[cfg(not(target_arch = "wasm32"))]
pub fn load_animation(path: &str) -> Result<LottieComposition, String> {
    #[cfg(feature = "native")]
    if path.ends_with(".lottie") {
        return load_from_dotlottie(path);
    }
    {
        load_from_file(path)
    }
}
