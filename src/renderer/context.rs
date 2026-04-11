use wgpu::*;

/// MSAA sample count (ThorVG uses 4x)
pub const MSAA_SAMPLES: u32 = 4;

/// Core GPU render context — owns device, queue, surface
pub struct RenderContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface: Option<Surface<'static>>,
    pub surface_config: Option<SurfaceConfiguration>,
    pub format: TextureFormat,
    /// True when the surface format is sRGB — shaders must output linear values
    /// and the GPU will convert linear→sRGB on write. False = output sRGB directly.
    pub is_srgb: bool,
    pub width: u32,
    pub height: u32,
    pub msaa_texture: Option<Texture>,
    pub msaa_view: Option<TextureView>,
    pub depth_stencil_texture: Option<Texture>,
    pub depth_stencil_view: Option<TextureView>,
}

impl RenderContext {
    /// Create a render context with a window surface
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("UltraRender Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // Rive-style: prefer Mailbox (uncapped, no tearing) > Immediate > Fifo (vsync fallback)
        let present_mode = if surface_caps.present_modes.contains(&PresentMode::Mailbox) {
            PresentMode::Mailbox
        } else if surface_caps.present_modes.contains(&PresentMode::Immediate) {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo // always supported
        };

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (msaa_texture, msaa_view) = create_msaa_texture(&device, format, width, height);
        let (depth_stencil_texture, depth_stencil_view) =
            create_depth_stencil_texture(&device, width, height);

        let is_srgb = format.is_srgb();

        Self {
            instance,
            adapter,
            device,
            queue,
            surface: Some(surface),
            surface_config: Some(config),
            format,
            is_srgb,
            width,
            height,
            msaa_texture: Some(msaa_texture),
            msaa_view: Some(msaa_view),
            depth_stencil_texture: Some(depth_stencil_texture),
            depth_stencil_view: Some(depth_stencil_view),
        }
    }

    /// Resize the surface
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
        if let Some(config) = &mut self.surface_config {
            config.width = width;
            config.height = height;
            if let Some(surface) = &self.surface {
                surface.configure(&self.device, config);
            }
        }
        let (tex, view) = create_msaa_texture(&self.device, self.format, width, height);
        self.msaa_texture = Some(tex);
        self.msaa_view = Some(view);
        let (depth_tex, depth_view) = create_depth_stencil_texture(&self.device, width, height);
        self.depth_stencil_texture = Some(depth_tex);
        self.depth_stencil_view = Some(depth_view);
    }
}

/// Create an MSAA multisample texture + view for anti-aliased rendering.
/// The render pass draws into this texture, then resolves to the surface texture.
fn create_msaa_texture(
    device: &Device,
    format: TextureFormat,
    width: u32,
    height: u32,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("msaa texture"),
        size: Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

fn create_depth_stencil_texture(
    device: &Device,
    width: u32,
    height: u32,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("depth stencil texture"),
        size: Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth24PlusStencil8,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

