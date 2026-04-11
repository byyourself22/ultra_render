use crate::lottie::interpolator::cubic_bezier_ease;
/// Rive-style StateMachine with proper transition mixing, exit time,
/// state layers, and SMI inputs (Bool/Number/Trigger).
///
/// Architecture mirrors Rive's StateMachineInstance:
/// - Multiple state layers that can run in parallel
/// - Transitions with mix (blend) time and easing
/// - Exit time conditions
/// - SMI input types with fire-once trigger semantics
/// - Animation mixing during transitions
use std::collections::HashMap;

// ─── Inputs (Rive SMIBool/SMINumber/SMITrigger) ────────────

/// State machine input value (matches Rive SMI types)
#[derive(Clone, Debug)]
pub enum SMInput {
    Bool(bool),
    Number(f32),
    Trigger(bool),
}

// ─── Conditions ─────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum ConditionOp {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterEqual,
    LessEqual,
}

/// A single condition on a transition
#[derive(Clone, Debug)]
pub struct TransitionCondition {
    pub input_name: String,
    pub op: ConditionOp,
    pub value: f32, // For bool: 0.0 = false, 1.0 = true
}

impl TransitionCondition {
    pub fn evaluate(&self, inputs: &HashMap<String, SMInput>) -> bool {
        let Some(input) = inputs.get(&self.input_name) else {
            return false;
        };
        match input {
            SMInput::Bool(v) => {
                let input_val = if *v { 1.0 } else { 0.0 };
                compare(input_val, self.value, &self.op)
            }
            SMInput::Number(v) => compare(*v, self.value, &self.op),
            SMInput::Trigger(fired) => *fired,
        }
    }
}

fn compare(a: f32, b: f32, op: &ConditionOp) -> bool {
    match op {
        ConditionOp::Equal => (a - b).abs() < 1e-6,
        ConditionOp::NotEqual => (a - b).abs() >= 1e-6,
        ConditionOp::GreaterThan => a > b,
        ConditionOp::LessThan => a < b,
        ConditionOp::GreaterEqual => a >= b,
        ConditionOp::LessEqual => a <= b,
    }
}

// ─── Easing ─────────────────────────────────────────────────

/// Transition easing curve
#[derive(Clone, Debug)]
pub enum TransitionEasing {
    Linear,
    CubicBezier { x1: f32, y1: f32, x2: f32, y2: f32 },
}

impl TransitionEasing {
    pub fn evaluate(&self, t: f32) -> f32 {
        match self {
            TransitionEasing::Linear => t,
            TransitionEasing::CubicBezier { x1, y1, x2, y2 } => {
                cubic_bezier_ease(*x1, *y1, *x2, *y2, t)
            }
        }
    }
}

// ─── States ─────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StateType {
    Entry,
    Exit,
    Animation,
    AnyState,
    BlendState1D,
}

#[derive(Clone, Debug)]
pub struct SMState {
    pub name: String,
    pub state_type: StateType,
    pub animation_index: Option<usize>,
    pub speed: f32,
    pub loop_animation: bool,
}

// ─── Transitions ────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SMTransition {
    pub from_state: usize,
    pub to_state: usize,
    pub conditions: Vec<TransitionCondition>,
    pub mix_time: f32, // Blend duration in seconds (Rive transition mixing)
    pub exit_time: Option<f32>, // Normalized exit time (0..1) — Rive ExitTimeCondition
    pub easing: TransitionEasing,
    pub pause_on_exit: bool, // Pause source animation when transitioning out
}

// ─── State Layer (Rive StateMachineLayerInstance) ────────────

/// A single state layer in the state machine.
/// Rive supports multiple layers that blend independently.
#[derive(Clone, Debug)]
pub struct SMLayer {
    pub current_state: usize,
    pub next_state: Option<usize>,
    pub mix_progress: f32, // 0..1 blend progress during transition
    pub mix_duration: f32, // Total transition time
    pub mix_easing: TransitionEasing,
    pub state_time: f32,     // Time spent in current state (seconds)
    pub wait_for_exit: bool, // Waiting for exit time condition
}

impl SMLayer {
    pub fn new(initial_state: usize) -> Self {
        Self {
            current_state: initial_state,
            next_state: None,
            mix_progress: 0.0,
            mix_duration: 0.0,
            mix_easing: TransitionEasing::Linear,
            state_time: 0.0,
            wait_for_exit: false,
        }
    }

    /// Get the mix factor (0 = fully current, 1 = fully next)
    pub fn mix_factor(&self) -> f32 {
        if self.next_state.is_none() {
            return 0.0;
        }
        self.mix_easing.evaluate(self.mix_progress.clamp(0.0, 1.0))
    }

    /// Is this layer currently in a transition?
    pub fn is_transitioning(&self) -> bool {
        self.next_state.is_some()
    }
}

// ─── StateMachine ───────────────────────────────────────────

/// Rive-style StateMachine with layers, mixing, and proper input handling.
#[derive(Clone, Debug)]
pub struct StateMachine {
    pub states: Vec<SMState>,
    pub transitions: Vec<SMTransition>,
    pub inputs: HashMap<String, SMInput>,
    pub layers: Vec<SMLayer>,
}

impl StateMachine {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            inputs: HashMap::new(),
            layers: Vec::new(),
        }
    }

    /// Add a layer starting at the given state
    pub fn add_layer(&mut self, initial_state: usize) {
        self.layers.push(SMLayer::new(initial_state));
    }

    // ─── Input setters (Rive SMIBool/SMINumber/SMITrigger) ──

    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.inputs.insert(name.to_string(), SMInput::Bool(value));
    }

    pub fn get_bool(&self, name: &str) -> bool {
        match self.inputs.get(name) {
            Some(SMInput::Bool(v)) => *v,
            _ => false,
        }
    }

    pub fn set_number(&mut self, name: &str, value: f32) {
        self.inputs.insert(name.to_string(), SMInput::Number(value));
    }

    pub fn get_number(&self, name: &str) -> f32 {
        match self.inputs.get(name) {
            Some(SMInput::Number(v)) => *v,
            _ => 0.0,
        }
    }

    pub fn fire_trigger(&mut self, name: &str) {
        self.inputs.insert(name.to_string(), SMInput::Trigger(true));
    }

    // ─── Advance ────────────────────────────────────────────

    /// Advance the state machine by delta_seconds.
    /// Returns a list of (layer_idx, AnimMixInfo) for the caller to blend animations.
    pub fn advance(&mut self, delta_seconds: f32) -> Vec<AnimMixInfo> {
        let mut results = Vec::new();

        for layer_idx in 0..self.layers.len() {
            let info = self.advance_layer(layer_idx, delta_seconds);
            results.push(info);
        }

        // Reset all triggers after evaluation
        for (_, input) in &mut self.inputs {
            if let SMInput::Trigger(_) = input {
                *input = SMInput::Trigger(false);
            }
        }

        results
    }

    fn advance_layer(&mut self, layer_idx: usize, dt: f32) -> AnimMixInfo {
        let layer = &mut self.layers[layer_idx];
        layer.state_time += dt;

        // If we're in a transition, advance the mix
        if let Some(next) = layer.next_state {
            if layer.mix_duration > 0.0 {
                layer.mix_progress += dt / layer.mix_duration;
            } else {
                layer.mix_progress = 1.0;
            }

            if layer.mix_progress >= 1.0 {
                // Transition complete
                let completed_state = next;
                layer.current_state = completed_state;
                layer.next_state = None;
                layer.mix_progress = 0.0;
                layer.state_time = 0.0;

                return AnimMixInfo {
                    current_anim: self
                        .states
                        .get(completed_state)
                        .and_then(|s| s.animation_index),
                    next_anim: None,
                    mix: 0.0,
                    speed: self
                        .states
                        .get(completed_state)
                        .map(|s| s.speed)
                        .unwrap_or(1.0),
                };
            }

            let mix = layer.mix_easing.evaluate(layer.mix_progress);
            return AnimMixInfo {
                current_anim: self
                    .states
                    .get(layer.current_state)
                    .and_then(|s| s.animation_index),
                next_anim: self.states.get(next).and_then(|s| s.animation_index),
                mix,
                speed: self.states.get(next).map(|s| s.speed).unwrap_or(1.0),
            };
        }

        // Not in a transition — check for new transitions
        let current = layer.current_state;
        let state_time = layer.state_time;

        for t in &self.transitions {
            if t.from_state != current {
                continue;
            }

            // Check exit time condition
            if let Some(exit_time) = t.exit_time {
                // exit_time is normalized 0..1 relative to animation duration
                // For now, use state_time directly as a simple approximation
                if state_time < exit_time {
                    continue;
                }
            }

            // Check all conditions
            if t.conditions.iter().all(|c| c.evaluate(&self.inputs)) {
                // Start transition
                let layer = &mut self.layers[layer_idx];
                layer.next_state = Some(t.to_state);
                layer.mix_progress = 0.0;
                layer.mix_duration = t.mix_time;
                layer.mix_easing = t.easing.clone();
                break;
            }
        }

        AnimMixInfo {
            current_anim: self.states.get(current).and_then(|s| s.animation_index),
            next_anim: None,
            mix: 0.0,
            speed: self.states.get(current).map(|s| s.speed).unwrap_or(1.0),
        }
    }

    /// Get current state name for a layer
    pub fn current_state_name(&self, layer_idx: usize) -> &str {
        self.layers
            .get(layer_idx)
            .and_then(|l| self.states.get(l.current_state))
            .map(|s| s.name.as_str())
            .unwrap_or("")
    }
}

// ─── Animation Mix Info ─────────────────────────────────────

/// Information about how to blend animations during a state transition.
/// The caller uses this to crossfade between two animations.
#[derive(Clone, Debug)]
pub struct AnimMixInfo {
    pub current_anim: Option<usize>, // Index of current animation
    pub next_anim: Option<usize>,    // Index of target animation (during transition)
    pub mix: f32,                    // 0.0 = fully current, 1.0 = fully next
    pub speed: f32,                  // Playback speed
}

impl AnimMixInfo {
    /// Is this layer transitioning between two animations?
    pub fn is_mixing(&self) -> bool {
        self.next_anim.is_some() && self.mix > 0.0
    }
}
