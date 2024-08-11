

use std::sync::mpsc;
use std::sync::Arc;

use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;
use wgpu::BindGroupDescriptor;
use wgpu::PipelineLayoutDescriptor;
use winit::dpi::PhysicalSize;
use winit::window::*;
use winit::event::*;
use winit::event_loop::{ActiveEventLoop,EventLoop};

fn pad(x: u32, pad: u32) -> u32 {
    if x > 0 {
        return (((x -1) / pad) + 1) * pad;
    } else {
        return 0;
    }
}

fn round_up_div(dividend: u64, divisor: u64) -> u64 {
    ((dividend -1) / divisor) + 1
}

#[repr(C)]
#[derive(Clone,Copy,bytemuck::Zeroable,bytemuck::Pod)]
struct Point(u32,u32);

#[allow(dead_code)]
struct GPU<'a> {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,

    backing_width: u32,
    backing_height: u32,
    pad_width: u32,
    pad_height: u32,

    render_pipeline: wgpu::RenderPipeline,
    diffusion_pipeline: wgpu::ComputePipeline,
    movement_pipeline: wgpu::ComputePipeline,
    line_pipeline: wgpu::ComputePipeline,

    line_group_layout: wgpu::BindGroupLayout,

    diffuse_storage: wgpu::Buffer,
    secondary_diffuse_storage: wgpu::Buffer,
    diffuse: wgpu::Texture,

    size_bind_group: wgpu::BindGroup,
    diffuse_storage_bind_group: wgpu::BindGroup,
    secondary_diffuse_storage_bind_group: wgpu::BindGroup,
    wall_buffer_bind_group: wgpu::BindGroup,
    fragment_bind_group: wgpu::BindGroup
}

impl<'a> GPU<'a> {
    async fn new(window: Arc<Window>,backing_width: u32, backing_height: u32) -> Self {
        let pad_width = pad(backing_width,16);
        let pad_height = pad(backing_height,8);

        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptionsBase::default()).await.unwrap();
        let (device,queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance
        }, None).await.unwrap();

        let size = window.inner_size();
        let surface = instance.create_surface(window).unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);
    
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let size_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Size Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ 
                        ty: wgpu::BufferBindingType::Uniform, 
                        has_dynamic_offset: false, 
                        min_binding_size: None
                    },
                    count: None
                }
            ]
        });

        let diffuse_storage_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            label: Some("Compute Group Layout"), 
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ 
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                        has_dynamic_offset: false, 
                        min_binding_size: None
                    },
                    count: None
                },
                wgpu::BindGroupLayoutEntry{
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ 
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                        has_dynamic_offset: false, 
                        min_binding_size: None 
                    },
                    count: None
                },
                wgpu::BindGroupLayoutEntry{
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ 
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                        has_dynamic_offset: false, 
                        min_binding_size: None 
                    },
                    count: None
                }
            ]
        });

        let wall_buffer_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            label: Some("Wall Buffer Layout"), 
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                        has_dynamic_offset: false, 
                        min_binding_size: None
                    },
                    count: None
                }
            ]
        });

        let fragment_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            label: Some("Fragment Group Layout"), 
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture{ 
                        sample_type: wgpu::TextureSampleType::Float { filterable: false }, 
                        view_dimension: wgpu::TextureViewDimension::D2, 
                        multisampled: false
                    },
                    count: None
                },
                wgpu::BindGroupLayoutEntry{
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None
                }
            ]
        });

        let line_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Line Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ 
                        ty: wgpu::BufferBindingType::Storage { read_only: true }, 
                        has_dynamic_offset: false, 
                        min_binding_size: None 
                    },
                    count: None
                },
                wgpu::BindGroupLayoutEntry{
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ 
                        ty: wgpu::BufferBindingType::Uniform, 
                        has_dynamic_offset: false, 
                        min_binding_size: None,
                    },
                    count: None
                }
            ]
        });
    
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&size_group_layout,&fragment_group_layout],
            push_constant_ranges: &[],
        });
    
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // 1.
                buffers: &[], // 2.
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState { // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { // 4.
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1, // 2.
                mask: !0, // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
            cache: None, // 6.
        });

        let diffusion_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("Diffusion Pipeline Layout"),
            bind_group_layouts: &[&size_group_layout,&diffuse_storage_group_layout,&diffuse_storage_group_layout,&wall_buffer_layout],
            push_constant_ranges: &[]
        });

        let diffusion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Diffusion Pipeline"),
            layout: Some(&diffusion_pipeline_layout),
            module: &shader,
            entry_point: "diffusion_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let movement_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("Movement Pipeline Layout"),
            bind_group_layouts: &[&size_group_layout,&diffuse_storage_group_layout,&diffuse_storage_group_layout,&wall_buffer_layout],
            push_constant_ranges: &[]
        });

        let movement_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Movement Pipeline"),
            layout: Some(&movement_pipeline_layout),
            module: &shader,
            entry_point: "movement_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let line_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("Line Pipeline Layout"),
            bind_group_layouts: &[&size_group_layout,&diffuse_storage_group_layout,&line_group_layout],
            push_constant_ranges: &[]
        });

        let line_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Line Pipeline"),
            layout: Some(&line_pipeline_layout),
            module: &shader,
            entry_point: "line_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let size_buffer = device.create_buffer_init(&BufferInitDescriptor{
            label: Some("Size Buffer"),
            contents: bytemuck::cast_slice(&[backing_width,backing_height,pad_width]),
            usage: wgpu::BufferUsages::UNIFORM
        });

        let diffuse_storage = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Diffuse Buffer"),
            size: pad_width  as u64 * backing_height as u64 * 4 * 4,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false
        });

        let velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Velocity Buffer"),
            size: backing_width as u64 * backing_height as u64 * 4 * 2,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false
        });

        let heat_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Heat Buffer"),
            size: backing_width as u64 * backing_height as u64 * 4,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false
        });

        let secondary_diffuse_storage = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Secondary Diffuse Buffer"),
            size: pad_width  as u64 * backing_height as u64 * 4 * 4,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false
        });

        let secondary_velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Secondary Velocity Buffer"),
            size: backing_width as u64 * backing_height as u64 * 4 * 2,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false
        });

        let secondary_heat_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Secondary Heat Buffer"),
            size: backing_width as u64 * backing_height as u64 * 4,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false
        });

        let wall_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Wall Buffer"),
            size: round_up_div(backing_width as u64 * backing_height as u64, 32), //stored as a bit array which is truly [u32]
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false
        });

        let diffuse = device.create_texture(&wgpu::TextureDescriptor{
            label: Some("Diffuse"),
            size: wgpu::Extent3d { width: backing_width, height: backing_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[wgpu::TextureFormat::Rgba32Float]
        });

        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor { 
            label: Some("Sampler"), 
            address_mode_u: wgpu::AddressMode::Repeat, 
            address_mode_v: wgpu::AddressMode::Repeat, 
            address_mode_w: wgpu::AddressMode::Repeat, 
            mag_filter: wgpu::FilterMode::Nearest,//Linear, 
            min_filter: wgpu::FilterMode::Nearest,//Linear, 
            mipmap_filter: wgpu::FilterMode::Nearest, 
            ..Default::default()
        });

        let size_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: Some("Size Bind Group"), 
            layout: &size_group_layout, 
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: size_buffer.as_entire_binding()
                }
            ]
        });

        let diffuse_storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: Some("Compute Bind Group"), 
            layout: &diffuse_storage_group_layout, 
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: diffuse_storage.as_entire_binding()
                },
                wgpu::BindGroupEntry{
                    binding: 1,
                    resource: velocity_buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry{
                    binding: 2,
                    resource: heat_buffer.as_entire_binding()
                }
            ] 
        });

        let secondary_diffuse_storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: Some("Secondary Diffuse Bind Group"), 
            layout: &diffuse_storage_group_layout, 
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: secondary_diffuse_storage.as_entire_binding()
                },
                wgpu::BindGroupEntry{
                    binding: 1,
                    resource: secondary_velocity_buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry{
                    binding: 2,
                    resource: secondary_heat_buffer.as_entire_binding()
                }
            ] 
        });

        let wall_buffer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Wall Bind Group"),
            layout: &wall_buffer_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: wall_buffer.as_entire_binding()
                }
            ]
        });

        let texture_view = diffuse.create_view(&wgpu::TextureViewDescriptor{ 
            label: Some("Texture View"), 
            format: Some(wgpu::TextureFormat::Rgba32Float), 
            dimension: Some(wgpu::TextureViewDimension::D2), 
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0, 
            mip_level_count: None, 
            base_array_layer: 0, 
            array_layer_count: None
        });

        let fragment_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("Sampler Bind Group"),
            layout: &fragment_group_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&texture_view)
                },
                wgpu::BindGroupEntry{
                    binding: 11,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler)
                }
            ]
        });

        GPU{
            instance,
            adapter,
            device,
            queue,
            surface,
            surface_config,

            backing_width,
            backing_height,
            pad_width,
            pad_height,

            render_pipeline,
            diffusion_pipeline,
            movement_pipeline,
            line_pipeline,

            line_group_layout,
            
            diffuse_storage,
            secondary_diffuse_storage,
            diffuse,
            
            size_bind_group,
            diffuse_storage_bind_group,
            secondary_diffuse_storage_bind_group,
            wall_buffer_bind_group,
            fragment_bind_group
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.clamp(1, 2048);
        self.surface_config.height = height.clamp(1,2048);
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn render(&mut self) {
        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });


        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBufferBase{
                buffer: &self.diffuse_storage,
                layout: wgpu::ImageDataLayout{ 
                    offset: 0, 
                    bytes_per_row: Some(self.pad_width * 4 * 4), 
                    rows_per_image: None 
                }
            },
            wgpu::ImageCopyTexture{
                texture: &self.diffuse,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All
            },
            wgpu::Extent3d{
                width: self.backing_width,
                height: self.backing_height,
                depth_or_array_layers: 1
            }
        );


        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(
                                wgpu::Color {
                                    r: 0.1,
                                    g: 0.2,
                                    b: 0.3,
                                    a: 1.0,
                                }
                            ),
                            store: wgpu::StoreOp::Store,
                        }
                    })
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None
            });
        
            // NEW!
            render_pass.set_pipeline(&self.render_pipeline); // 2.
            render_pass.set_bind_group(0, &self.size_bind_group, &[]);
            //render_pass.set_bind_group(1, &diffuse_storage_bind_group, &[]);
            render_pass.set_bind_group(1, &self.fragment_bind_group, &[]);
            render_pass.draw(0..6, 0..1); // 3.
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn compute(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None 
            });       
            
            for _ in 0..1 {
                compute_pass.set_pipeline(&self.movement_pipeline);
                compute_pass.set_bind_group(0, &self.size_bind_group, &[]);
                compute_pass.set_bind_group(1, &self.diffuse_storage_bind_group, &[]);
                compute_pass.set_bind_group(2, &self.secondary_diffuse_storage_bind_group, &[]);
                compute_pass.set_bind_group(3, &self.wall_buffer_bind_group, &[]);
                compute_pass.dispatch_workgroups(pad(self.backing_width,8)/8, self.pad_height/8, 1);
                compute_pass.set_pipeline(&self.diffusion_pipeline);
                compute_pass.set_bind_group(0, &self.size_bind_group, &[]);
                compute_pass.set_bind_group(1, &self.secondary_diffuse_storage_bind_group, &[]);
                compute_pass.set_bind_group(2, &self.diffuse_storage_bind_group, &[]);
                compute_pass.set_bind_group(3, &self.wall_buffer_bind_group, &[]);
                compute_pass.dispatch_workgroups(pad(self.backing_width,8)/8, self.pad_height/8, 1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn line(&mut self, line_points: &[Point], color: [f32; 4], velocity: [f32; 2], heat: f32) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Line Encoder"),
        }); 

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None 
            });       

            let line_points_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Line Points Buffer"),
                contents: bytemuck::cast_slice(line_points),
                usage: wgpu::BufferUsages::STORAGE
            });

            let mut line_config = [0.0; 8];
            line_config[0..4].copy_from_slice(&color);
            line_config[4..6].copy_from_slice(&velocity);
            line_config[6] = heat;

            let line_config_buffer = self.device.create_buffer_init(&BufferInitDescriptor{
                label: Some("Line Color Buffer"),
                contents: bytemuck::cast_slice(&line_config),
                usage: wgpu::BufferUsages::UNIFORM
            });

            let line_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Line Bind Group"),
                layout: &self.line_group_layout,
                entries: &[
                    wgpu::BindGroupEntry{
                        binding: 10,
                        resource: line_points_buffer.as_entire_binding()
                    },
                    wgpu::BindGroupEntry{
                        binding: 11,
                        resource: line_config_buffer.as_entire_binding()
                    }
                ]
            });

            compute_pass.set_pipeline(&self.line_pipeline);
            compute_pass.set_bind_group(0, &self.size_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.diffuse_storage_bind_group, &[]);
            compute_pass.set_bind_group(2, &line_bind_group, &[]);
            compute_pass.dispatch_workgroups(pad(line_points.len() as u32 -1,64), 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}

enum GPUCommand {
    Resize{width: u32, height: u32},
    Compute,
    ToggleLoop,
    Line{line_points: Vec<Point>, color: [f32; 4], velocity: [f32; 2], heat: f32}
}

struct GPUSize {
    surface_width: u32,
    surface_height: u32,
    backing_width: u32,
    backing_height: u32
}

struct App {
    idx: usize,
    window_id: Option<WindowId>,
    window: Option<Arc<Window>>,

    mouse_held: bool,
    line_points: Vec<Point>,
    line_color: [f32; 4],
    line_velocity: [f32; 2],
    line_heat: f32,

    gpu: Option<mpsc::Sender<GPUCommand>>,
    gpu_size: Option<GPUSize>
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        } else {
            let window = Arc::new(event_loop.create_window(Window::default_attributes()).unwrap());
            window.set_max_inner_size(Some(PhysicalSize{width: 2048, height: 2048}));
            let size = window.inner_size();
            let mut gpu = pollster::block_on(GPU::new(window.clone(),size.width,size.height));
            self.window = Some(window);
            let (tx,rx) = mpsc::channel();
            std::thread::spawn(move || {
                let mut compute_loop = false;
                loop {
                    match rx.recv_timeout(std::time::Duration::from_millis(20)) {
                        Err(e) => {
                            match e {
                                mpsc::RecvTimeoutError::Disconnected => {break;}
                                mpsc::RecvTimeoutError::Timeout => {
                                    if compute_loop {
                                        gpu.compute();
                                        gpu.render();
                                    }
                                }
                            }
                        }
                        Ok(command) => {
                            match command {
                                GPUCommand::Resize { width, height } => {gpu.resize(width, height); gpu.render()}
                                GPUCommand::Compute => {gpu.compute(); gpu.render()}
                                GPUCommand::ToggleLoop => {compute_loop = !compute_loop}
                                GPUCommand::Line { line_points, color , velocity, heat} => {gpu.line(&line_points,color, velocity, heat); gpu.render()}
                            }
                        }
                    }
                }
            });
            let mut block_points = Vec::new();
            for i in 0..100 {
                block_points.push(Point(300,200+2*i));
                block_points.push(Point(500,200+2*i));
                block_points.push(Point(500,200+2*i+1));
                block_points.push(Point(300,200+2*i+1));
            }
            block_points.push(Point(299,399));
            tx.send(GPUCommand::Line{
                line_points: block_points, 
                color: self.line_color, 
                velocity: self.line_velocity,
                heat: self.line_heat
            }).unwrap();
            self.gpu = Some(tx);
            self.gpu_size = Some(GPUSize{
                surface_width: size.width, 
                surface_height: size.height,
                backing_width: size.width,
                backing_height: size.height
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        //println!("{:?}",event);
        if event == WindowEvent::Destroyed && self.window_id == Some(window_id) {
            println!(
                "--------------------------------------------------------- Window {} Destroyed",
                self.idx
            );
            self.window_id = None;
            event_loop.exit();
            return;
        }

        let _window = match self.window.as_mut() {
            Some(window) => window,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                println!(
                    "--------------------------------------------------------- Window {} \
                     CloseRequested",
                    self.idx
                );
                self.window = None;
                self.gpu = None;
            },
            WindowEvent::RedrawRequested => {
                
            },  
            WindowEvent::Resized(size) => {
                self.gpu.as_mut().map(|inner| inner.send(GPUCommand::Resize{width: size.width, height: size.height}));
                self.gpu_size.as_mut().map(|inner| {inner.surface_width = size.width; inner.surface_height = size.height});
            },
            WindowEvent::KeyboardInput {event, ..} => {
                match event.logical_key {
                    winit::keyboard::Key::Character(key) => {
                        if event.state.is_pressed() {
                            match key.as_str() {
                                "r" => {
                                    if let Some(gpu) = &mut self.gpu {
                                        gpu.send(GPUCommand::Compute).unwrap();
                                    }
                                }
                                "f" => {
                                    if let Some(gpu) = &mut self.gpu {
                                        gpu.send(GPUCommand::ToggleLoop).unwrap();
                                    }
                                }
                                "1" => {
                                    self.line_color[0] = 1.0 - self.line_color[0];
                                }
                                "2" => {
                                    self.line_color[1] = 1.0 - self.line_color[1];
                                }
                                "3" => {
                                    self.line_color[2] = 1.0 - self.line_color[2];
                                }
                                "4" => {
                                    println!("Enter X Velocity: ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    self.line_velocity[0] = input.parse::<f32>().unwrap();
                                }
                                "5" => {
                                    println!("Enter Y Velocity: ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    self.line_velocity[1] = input.parse::<f32>().unwrap();
                                }
                                "6" => {
                                    println!("Enter Heat: ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    self.line_heat = input.parse::<f32>().unwrap();
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved {position, ..} => {
                if let Some(size) = &self.gpu_size {
                    if self.mouse_held {
                        let x = position.x * size.backing_width as f64 / size.surface_width as f64;
                        let y = position.y * size.backing_height as f64 / size.surface_height as f64;
                        self.line_points.push(Point(x as u32, y as u32));
                    }
                }
            }
            WindowEvent::MouseInput {state, button , ..} => {
                match button {
                    MouseButton::Left => {
                        match state {
                            ElementState::Pressed => {self.mouse_held = true},
                            ElementState::Released => {
                                self.mouse_held = false;
                                if let Some(gpu) = &self.gpu {
                                    gpu.send(GPUCommand::Line{
                                        line_points: std::mem::replace(&mut self.line_points, Vec::new()), 
                                        color: self.line_color, 
                                        velocity: self.line_velocity,
                                        heat: self.line_heat
                                    }).unwrap();
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => (),
        }
    }
}

fn main() {

    env_logger::init();


    let event_loop = EventLoop::new().unwrap();
    let mut app = App{
        idx: 0, 
        window_id: None, 
        window: None, 

        mouse_held: false,
        line_points: Vec::new(), 
        line_color: [1.0,1.0,1.0,1.0], 
        line_velocity: [0.0, 0.0],
        line_heat: 0.5,

        gpu: None, 
        gpu_size: None
    };
    event_loop.run_app(&mut app).unwrap();
    println!("test");
}

