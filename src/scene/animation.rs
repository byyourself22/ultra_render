/// Animation playback state
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlayState {
    Playing,
    Paused,
    Stopped,
}

/// Controls animation playback for a single animation
#[derive(Clone, Debug)]
pub struct AnimationPlayer {
    pub frame: f32,
    pub frame_rate: f32,
    pub in_point: f32,
    pub out_point: f32,
    pub speed: f32,
    pub looping: bool,
    pub state: PlayState,
    pub ping_pong: bool,
    forward: bool,
}

impl AnimationPlayer {
    pub fn new(frame_rate: f32, in_point: f32, out_point: f32) -> Self {
        Self {
            frame: in_point,
            frame_rate,
            in_point,
            out_point,
            speed: 1.0,
            looping: true,
            state: PlayState::Stopped,
            ping_pong: false,
            forward: true,
        }
    }

    pub fn play(&mut self) {
        self.state = PlayState::Playing;
    }

    pub fn pause(&mut self) {
        self.state = PlayState::Paused;
    }

    pub fn stop(&mut self) {
        self.state = PlayState::Stopped;
        self.frame = self.in_point;
        self.forward = true;
    }

    pub fn seek(&mut self, frame: f32) {
        self.frame = frame.clamp(self.in_point, self.out_point);
    }

    pub fn seek_normalized(&mut self, t: f32) {
        let t = t.clamp(0.0, 1.0);
        self.frame = self.in_point + t * (self.out_point - self.in_point);
    }

    /// Advance animation by delta_seconds. Returns true if animation is active.
    pub fn advance(&mut self, delta_seconds: f32) -> bool {
        if self.state != PlayState::Playing {
            return false;
        }

        let delta_frames = delta_seconds * self.frame_rate * self.speed;
        let duration = self.out_point - self.in_point;

        if duration <= 0.0 {
            return false;
        }

        if self.ping_pong {
            if self.forward {
                self.frame += delta_frames;
                if self.frame >= self.out_point {
                    self.frame = self.out_point - (self.frame - self.out_point);
                    self.forward = false;
                }
            } else {
                self.frame -= delta_frames;
                if self.frame <= self.in_point {
                    if self.looping {
                        self.frame = self.in_point + (self.in_point - self.frame);
                        self.forward = true;
                    } else {
                        self.frame = self.in_point;
                        self.state = PlayState::Stopped;
                    }
                }
            }
        } else {
            self.frame += delta_frames;
            if self.frame >= self.out_point {
                if self.looping {
                    self.frame = self.in_point + (self.frame - self.out_point) % duration;
                } else {
                    self.frame = self.out_point;
                    self.state = PlayState::Stopped;
                }
            }
        }

        true
    }

    pub fn progress(&self) -> f32 {
        let duration = self.out_point - self.in_point;
        if duration <= 0.0 {
            0.0
        } else {
            (self.frame - self.in_point) / duration
        }
    }

    pub fn is_playing(&self) -> bool {
        self.state == PlayState::Playing
    }

    pub fn duration_seconds(&self) -> f32 {
        (self.out_point - self.in_point) / self.frame_rate
    }
}
