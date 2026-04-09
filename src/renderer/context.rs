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
    pub width: u32,
    pub height: u32,
    pub msaa_texture: Option<Texture>,
    pub msaa_view: Option<TextureView>,
}

impl RenderContext {
    /// Create a render context with a window surface
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
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

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (msaa_texture, msaa_view) = create_msaa_texture(&device, format, size.width.max(1), size.height.max(1));

        Self {
            instance,
            adapter,
            device,
            queue,
            surface: Some(surface),
            surface_config: Some(config),
            format,
            width: size.width,
            height: size.height,
            msaa_texture: Some(msaa_texture),
            msaa_view: Some(msaa_view),
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
    }
}

/// Create an MSAA multisample texture + view for anti-aliased rendering.
/// The render pass draws into this texture, then resolves to the surface texture.
fn create_msaa_texture(device: &Device, format: TextureFormat, width: u32, height: u32) -> (Texture, TextureView) {
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
