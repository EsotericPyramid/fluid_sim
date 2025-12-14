

use std::sync::mpsc;
use std::sync::Arc;
use std::vec;

use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;
use wgpu::BindGroupDescriptor;
use wgpu::PipelineLayoutDescriptor;
use winit::dpi::PhysicalSize;
use winit::window::*;
use winit::event::*;
use winit::event_loop::{ActiveEventLoop,EventLoop};

//no clue why this gives warnings
#[repr(C)]
#[derive(Clone,Copy,bytemuck::Zeroable,bytemuck::Pod)]
struct Point(u32,u32);

#[derive(Clone)]
struct LineConfig{
    color: [f32; 4],
    velocity: [f32; 2],
    heat: f32,
    color_mode: ColorMode,
    static_velocity: bool,
    static_heat: bool,
    is_wall: bool,
}

#[derive(Clone)]
enum ColorMode{
    Default,
    Static,
    Paint
}

impl LineConfig {
    fn as_byte_array(self) -> [u8; 32] {
        bytemuck::cast({
            let mut line_config = [0.0; 8];
            line_config[0..4].copy_from_slice(&self.color);
            line_config[4..6].copy_from_slice(&self.velocity);
            line_config[6] = self.heat;
            if self.is_wall {
                line_config[7] = bytemuck::cast(0b1111); 
            } else {
                let mut mode: u32= match self.color_mode {
                    ColorMode::Default => 0,
                    ColorMode::Static => 1,
                    ColorMode::Paint => 2,
                };
                mode |= (self.static_velocity as u32) << 2;
                mode |= (self.static_heat as u32) << 3;
                line_config[7] = bytemuck::cast(mode);
            }
            line_config
        })
    }
}

struct GPU<'a> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,

    decoder: AsyncVideoDecoder,
    frame_buffer: vec::IntoIter<ffmpeg_next::util::frame::Video>,

    backing_width: u32,
    backing_height: u32,

    render_pipeline: wgpu::RenderPipeline,
    diffusion_pipeline: wgpu::ComputePipeline,
    movement_pipeline: wgpu::ComputePipeline,
    line_pipeline: wgpu::ComputePipeline,
    load_texture_pipeline: wgpu::ComputePipeline,
    // diagnostic util pipelines
    half_sum_pipeline: wgpu::ComputePipeline,
    square_pipeline: wgpu::ComputePipeline,
    diffusion_scale_pipeline: wgpu::ComputePipeline,
    // fun pipeline
    frame_apply_pipeline: wgpu::ComputePipeline,

    line_group_layout: wgpu::BindGroupLayout,
    load_texture_layout: wgpu::BindGroupLayout,
    util_buffer_group_layout: wgpu::BindGroupLayout,
    half_sum_group_layout: wgpu::BindGroupLayout,
    frame_group_layout: wgpu::BindGroupLayout,

    diffuse_buffer: wgpu::Buffer,
    velocity_buffer: wgpu::Buffer,
    heat_buffer: wgpu::Buffer,
    secondary_diffuse_buffer: wgpu::Buffer,
    secondary_velocity_buffer: wgpu::Buffer,
    secondary_heat_buffer: wgpu::Buffer,
    diffuse_view: wgpu::TextureView,

    size_bind_group: wgpu::BindGroup,
    gas_data_bind_group: wgpu::BindGroup,
    secondary_gas_data_bind_group: wgpu::BindGroup,
    mode_bind_group: wgpu::BindGroup,
    fragment_bind_group: wgpu::BindGroup,
}

impl<'a> GPU<'a> {
    async fn new(window: Arc<Window>,backing_width: u32, backing_height: u32) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptionsBase::default()).await.unwrap();
        let (device,queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
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
    
        let sim_shader = device.create_shader_module(wgpu::include_wgsl!("sim_shader.wgsl"));
        let util_shader = device.create_shader_module(wgpu::include_wgsl!("util_shader.wgsl"));

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

        let diffuse_buffer_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
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

        let mode_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
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

        let load_texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Load To Texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 20,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { 
                        access: wgpu::StorageTextureAccess::WriteOnly, 
                        format: wgpu::TextureFormat::Rgba32Float, 
                        view_dimension: wgpu::TextureViewDimension::D2 
                    },
                    count: None
                },
                wgpu::BindGroupLayoutEntry{
                    binding: 21,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Uniform, 
                        has_dynamic_offset: false, 
                        min_binding_size: None 
                    },
                    count: None
                }
            ]
        });

        let util_buffer_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Util Buffer Group Layout"),
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

        let half_sum_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Half Sum Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 10,
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

        let frame_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Frame Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 30,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ 
                        ty: wgpu::BufferBindingType::Storage { read_only: true }, 
                        has_dynamic_offset: false, 
                        min_binding_size: None
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
                module: &sim_shader,
                entry_point: "vs_main", // 1.
                buffers: &[], // 2.
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState { // 3.
                module: &sim_shader,
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
            bind_group_layouts: &[&size_group_layout,&diffuse_buffer_group_layout,&diffuse_buffer_group_layout,&mode_layout],
            push_constant_ranges: &[]
        });

        let diffusion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Diffusion Pipeline"),
            layout: Some(&diffusion_pipeline_layout),
            module: &sim_shader,
            entry_point: "diffusion_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let movement_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("Movement Pipeline Layout"),
            bind_group_layouts: &[&size_group_layout,&diffuse_buffer_group_layout,&diffuse_buffer_group_layout,&mode_layout],
            push_constant_ranges: &[]
        });

        let movement_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Movement Pipeline"),
            layout: Some(&movement_pipeline_layout),
            module: &sim_shader,
            entry_point: "movement_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let line_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("Line Pipeline Layout"),
            bind_group_layouts: &[&size_group_layout,&diffuse_buffer_group_layout,&line_group_layout,&mode_layout],
            push_constant_ranges: &[]
        });

        let line_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Line Pipeline"),
            layout: Some(&line_pipeline_layout),
            module: &sim_shader,
            entry_point: "line_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let load_texture_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Load To Texture Pipeline Layout"),
            bind_group_layouts: &[&size_group_layout,&diffuse_buffer_group_layout,&load_texture_layout,&mode_layout],
            push_constant_ranges: &[]
        });

        let load_texture_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
           label: Some("Load To Texture Pipeline"),
           layout: Some(&load_texture_pipeline_layout),
           module: &sim_shader,
           entry_point: "load_to_texture",
           compilation_options: wgpu::PipelineCompilationOptions::default(),
           cache: None
        });

        let half_sum_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Half Sum Pipeline Layout"),
            bind_group_layouts: &[&util_buffer_group_layout,&half_sum_group_layout],
            push_constant_ranges: &[]
        });

        let half_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Half Sum Pipeline"),
            layout: Some(&half_sum_pipeline_layout),
            module: &util_shader,
            entry_point: "half_sum",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let util_pipelines_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Util Pipelines Layout"),
            bind_group_layouts: &[&util_buffer_group_layout],
            push_constant_ranges: &[]
        });

        let square_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Square Pipeline"),
            layout: Some(&util_pipelines_layout),
            module: &util_shader,
            entry_point: "square_arr",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let diffusion_scale_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Diffusion Scale Pipeline Layout"),
            bind_group_layouts: &[&util_buffer_group_layout,&util_buffer_group_layout,&size_group_layout],
            push_constant_ranges: &[]
        });

        let diffusion_scale_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Diffusion Scale Pipeline"),
            layout: Some(&diffusion_scale_pipeline_layout),
            module: &util_shader,
            entry_point: "diffusion_scale",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let frame_apply_scale_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Frame Apply Scale Pipeline Layout"),
            bind_group_layouts: &[&size_group_layout,&diffuse_buffer_group_layout,&frame_group_layout,&mode_layout],
            push_constant_ranges: &[]
        });

        let frame_apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Frame Apply Pipeline"),
            layout: Some(&frame_apply_scale_pipeline_layout),
            module: &sim_shader,
            entry_point: "frame_apply",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let size_buffer = device.create_buffer_init(&BufferInitDescriptor{
            label: Some("Size Buffer"),
            contents: bytemuck::cast_slice(&[backing_width,backing_height]),
            usage: wgpu::BufferUsages::UNIFORM
        });

        let diffuse_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Diffuse Buffer"),
            size: backing_width  as u64 * backing_height as u64 * 4 * 4,
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

        let secondary_diffuse_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Secondary Diffuse Buffer"),
            size: backing_width  as u64 * backing_height as u64 * 4 * 4,
            usage: wgpu::BufferUsages::COPY_SRC  | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false
        });

        let secondary_velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Secondary Velocity Buffer"),
            size: backing_width as u64 * backing_height as u64 * 4 * 2,
            usage: wgpu::BufferUsages::COPY_SRC  | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false
        });

        let secondary_heat_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Secondary Heat Buffer"),
            size: backing_width as u64 * backing_height as u64 * 4,
            usage: wgpu::BufferUsages::COPY_SRC  | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false
        });

        let mode_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Wall Buffer"),
            size: ((backing_width as u64 * backing_height as u64) / 2).next_multiple_of(4), //stored as a bit array which is truly [u32]
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
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

        let gas_data_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: Some("Compute Bind Group"), 
            layout: &diffuse_buffer_group_layout, 
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: diffuse_buffer.as_entire_binding()
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

        let secondary_gas_data_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: Some("Secondary Diffuse Bind Group"), 
            layout: &diffuse_buffer_group_layout, 
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: secondary_diffuse_buffer.as_entire_binding()
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

        let mode_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Wall Bind Group"),
            layout: &mode_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: mode_buffer.as_entire_binding()
                }
            ]
        });

        let diffuse_view = diffuse.create_view(&wgpu::TextureViewDescriptor{ 
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
                    resource: wgpu::BindingResource::TextureView(&diffuse_view)
                },
                wgpu::BindGroupEntry{
                    binding: 11,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler)
                }
            ]
        });

        GPU{
            device,
            queue,
            surface,
            surface_config,

            decoder: AsyncVideoDecoder::new(backing_width,backing_height),
            frame_buffer: Vec::new().into_iter(),

            backing_width,
            backing_height,

            render_pipeline,
            diffusion_pipeline,
            movement_pipeline,
            line_pipeline,
            load_texture_pipeline,

            half_sum_pipeline,
            square_pipeline,
            diffusion_scale_pipeline,

            frame_apply_pipeline,

            load_texture_layout,
            line_group_layout,
            util_buffer_group_layout,
            half_sum_group_layout,
            frame_group_layout,
            
            diffuse_buffer,
            velocity_buffer,
            heat_buffer,
            secondary_diffuse_buffer,
            secondary_velocity_buffer,
            secondary_heat_buffer,
            diffuse_view,
            
            size_bind_group,
            gas_data_bind_group,
            secondary_gas_data_bind_group,
            mode_bind_group,
            fragment_bind_group
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.clamp(1, 2048);
        self.surface_config.height = height.clamp(1,2048);
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn render(&mut self, render_mode: u32) {
        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        let copy_mode = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Copy Mode Buffer"),
            contents: bytemuck::cast_slice(&[render_mode]),
            usage: wgpu::BufferUsages::UNIFORM
        });

        let load_texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("Load To Texture Bind Group"),
            layout: &self.load_texture_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 20,
                    resource: wgpu::BindingResource::TextureView(&self.diffuse_view)
                },
                wgpu::BindGroupEntry{
                    binding: 21,
                    resource: copy_mode.as_entire_binding()
                }
            ]
        });
        //encoder.copy_buffer_to_texture(
        //    wgpu::ImageCopyBufferBase{
        //        buffer: &self.diffuse_buffer,
        //        layout: wgpu::ImageDataLayout{ 
        //            offset: 0, 
        //            bytes_per_row: Some(self.backing_width * 4 * 4), 
        //            rows_per_image: None 
        //        }
        //    },
        //    wgpu::ImageCopyTexture{
        //        texture: &self.diffuse,
        //        mip_level: 0,
        //        origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
        //        aspect: wgpu::TextureAspect::All
        //    },
        //    wgpu::Extent3d{
        //        width: self.backing_width,
        //        height: self.backing_height,
        //        depth_or_array_layers: 1
        //    }
        //);


        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
                label: Some("Load Texture Pass"),
                timestamp_writes: None
            });

            compute_pass.set_pipeline(&self.load_texture_pipeline);
            compute_pass.set_bind_group(0, &self.size_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.gas_data_bind_group, &[]);
            compute_pass.set_bind_group(2, &load_texture_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.mode_bind_group, &[]);
            compute_pass.dispatch_workgroups(self.backing_width.div_ceil(8), self.backing_height.div_ceil(8), 1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment sim_shader targets
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
        
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.size_bind_group, &[]);
            render_pass.set_bind_group(1, &self.fragment_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn compute(&mut self,num_ticks: u32,apply_frame: bool) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        let mut frame_bind_group = None;
        loop {
            if apply_frame {
                if let Some(frame) = self.frame_buffer.next() {
                    {
                        let gpu_frame = self.device.create_buffer_init(&BufferInitDescriptor{
                            label: Some("GPU Frame"),
                            contents: frame.data(0),
                            usage: wgpu::BufferUsages::STORAGE
                        });
    
                        frame_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor{
                            label: Some("Frame Bind Group"),
                            layout: &self.frame_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry{
                                    binding: 30,
                                    resource: gpu_frame.as_entire_binding()
                                }
                            ]
                        }));
                    }
                    break;
                } else {
                    if let Some(buffer) = self.decoder.get_frame_buf() {
                        self.frame_buffer = buffer.into_iter();
                    } else {
                        break;
                    }
                }
            } else {
                break;
            }
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None 
            });
            
            for _ in 0..num_ticks {
                compute_pass.set_pipeline(&self.movement_pipeline);
                compute_pass.set_bind_group(0, &self.size_bind_group, &[]);
                compute_pass.set_bind_group(1, &self.gas_data_bind_group, &[]);
                compute_pass.set_bind_group(2, &self.secondary_gas_data_bind_group, &[]);
                compute_pass.set_bind_group(3, &self.mode_bind_group, &[]);
                compute_pass.dispatch_workgroups(self.backing_width.div_ceil(8), self.backing_height.div_ceil(8), 1);
                compute_pass.set_pipeline(&self.diffusion_pipeline);
                compute_pass.set_bind_group(0, &self.size_bind_group, &[]);
                compute_pass.set_bind_group(1, &self.secondary_gas_data_bind_group, &[]);
                compute_pass.set_bind_group(2, &self.gas_data_bind_group, &[]);
                compute_pass.set_bind_group(3, &self.mode_bind_group, &[]);
                compute_pass.dispatch_workgroups(self.backing_width.div_ceil(8), self.backing_height.div_ceil(8), 1);
                if let Some(frame_bind_group) = &frame_bind_group {
                    compute_pass.set_pipeline(&self.frame_apply_pipeline);
                    compute_pass.set_bind_group(0, &self.size_bind_group, &[]);
                    compute_pass.set_bind_group(1, &self.gas_data_bind_group, &[]);
                    compute_pass.set_bind_group(2, &frame_bind_group, &[]);
                    compute_pass.set_bind_group(3, &self.mode_bind_group, &[]);
                    compute_pass.dispatch_workgroups(self.backing_width.div_ceil(8), self.backing_height.div_ceil(8), 1);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn movement(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None 
            });       
            
            compute_pass.set_pipeline(&self.movement_pipeline);
            compute_pass.set_bind_group(0, &self.size_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.gas_data_bind_group, &[]);
            compute_pass.set_bind_group(2, &self.secondary_gas_data_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.mode_bind_group, &[]);
            compute_pass.dispatch_workgroups(self.backing_width.div_ceil(8), self.backing_height.div_ceil(8), 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.secondary_diffuse_buffer,
            0,
            &self.diffuse_buffer,
            0,
            self.backing_width as u64 * self.backing_height as u64 * 4 * 4
        );

        encoder.copy_buffer_to_buffer(
            &self.secondary_velocity_buffer,
            0,
            &self.velocity_buffer,
            0,
            self.backing_width as u64 * self.backing_height as u64 * 4 * 2
        );

        encoder.copy_buffer_to_buffer(
            &self.secondary_heat_buffer,
            0,
            &self.heat_buffer,
            0,
            self.backing_width as u64 * self.backing_height as u64 * 4
        );

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn diffusion(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None 
            });       
            
            compute_pass.set_pipeline(&self.diffusion_pipeline);
            compute_pass.set_bind_group(0, &self.size_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.gas_data_bind_group, &[]);
            compute_pass.set_bind_group(2, &self.secondary_gas_data_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.mode_bind_group, &[]);
            compute_pass.dispatch_workgroups(self.backing_width.div_ceil(8), self.backing_height.div_ceil(8), 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.secondary_diffuse_buffer,
            0,
            &self.diffuse_buffer,
            0,
            self.backing_width as u64 * self.backing_height as u64 * 4 * 4
        );

        encoder.copy_buffer_to_buffer(
            &self.secondary_velocity_buffer,
            0,
            &self.velocity_buffer,
            0,
            self.backing_width as u64 * self.backing_height as u64 * 4 * 2
        );

        encoder.copy_buffer_to_buffer(
            &self.secondary_heat_buffer,
            0,
            &self.heat_buffer,
            0,
            self.backing_width as u64 * self.backing_height as u64 * 4
        );

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn line(&mut self, line_points: &[Point], line_config: LineConfig) {
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

            let line_config_buffer = self.device.create_buffer_init(&BufferInitDescriptor{
                label: Some("Line Color Buffer"),
                contents: &line_config.as_byte_array(),
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
            compute_pass.set_bind_group(1, &self.gas_data_bind_group, &[]);
            compute_pass.set_bind_group(2, &line_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.mode_bind_group, &[]);
            compute_pass.dispatch_workgroups((line_points.len() as u32 -1).div_ceil(64), 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn sum_buffer(&self, mut pass: wgpu::ComputePass, buffer: &wgpu::BindGroup, block_size: u32) {
        let mut chunk_size = 1;
        while chunk_size < self.backing_width * self.backing_height {
            let half_sum_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("Half Sum Buffer"),
                contents: bytemuck::cast_slice(&[block_size,chunk_size]),
                usage: wgpu::BufferUsages::UNIFORM
            });
            chunk_size *= 2;

            let half_sum_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Half Sum Bing Group"),
                layout: &self.half_sum_group_layout,
                entries: &[
                    wgpu::BindGroupEntry{
                        binding: 10,
                        resource: half_sum_buffer.as_entire_binding()
                    }
                ]
            });

            pass.set_pipeline(&self.half_sum_pipeline);
            pass.set_bind_group(0, buffer, &[]);
            pass.set_bind_group(1, &half_sum_bind_group, &[]);
            pass.dispatch_workgroups(self.backing_width * self.backing_height.div_ceil(chunk_size *  64), 1, 1);
        }
    }

    fn print_diagnostics(&self) {
        let staging_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Diagnostic Staging Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false
        }));

        let diffuse_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("Diagnostic Diffuse Bind Group"),
            layout: &self.util_buffer_group_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: self.secondary_diffuse_buffer.as_entire_binding()
                }
            ]
        });

        let velocity_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("Diagnostic Velocity Bind Group"),
            layout: &self.util_buffer_group_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: self.secondary_velocity_buffer.as_entire_binding()
                }
            ]
        });

        let heat_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("Diagnostic Heat Bind Group"),
            layout: &self.util_buffer_group_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: self.secondary_heat_buffer.as_entire_binding()
                }
            ]
        });

        let mut total_energy = 0.0;
        for i in 0..4 {
            let (active_bind_group, active_buffer, main_buffer ,size, block_size)= match i {
                0 | 1 => (&velocity_bind_group, &self.secondary_velocity_buffer,&self.velocity_buffer,(self.backing_width, self.backing_height),2),
                2 => (&heat_bind_group, &self.secondary_heat_buffer,&self.heat_buffer,(self.backing_width, self.backing_height),1),
                3 => (&diffuse_bind_group, &self.secondary_diffuse_buffer,&self.diffuse_buffer,(self.backing_width, self.backing_height),4),
                _ => panic!("Error invalid i")
            };  

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Diagnostic Encoder"),
            });

            if i == 0 {
                encoder.copy_buffer_to_buffer(
                    &self.diffuse_buffer,
                    0,
                    &self.secondary_diffuse_buffer,
                    0,
                    self.backing_width as u64 * self.backing_height as u64 * 4 * 4
                );
            }

            if i != 3 {
                encoder.copy_buffer_to_buffer(
                    main_buffer,
                    0,
                    active_buffer, 
                    0, 
                    size.0 as u64 * size.1 as u64 * block_size * 4
                );
            }
            
    
        
            let mut diagnostic_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
                label: Some("Diagnostic Pass"),
                timestamp_writes: None
            });
            
            if i == 1 { // KE run
                diagnostic_pass.set_pipeline(&self.square_pipeline);
                diagnostic_pass.set_bind_group(0, &velocity_bind_group, &[]);
                diagnostic_pass.dispatch_workgroups((self.backing_width * self.backing_height * 2).div_ceil(64), 1, 1);
            }
            if i != 3 {
                diagnostic_pass.set_pipeline(&self.diffusion_scale_pipeline);
                diagnostic_pass.set_bind_group(0, active_bind_group, &[]);
                diagnostic_pass.set_bind_group(1, &diffuse_bind_group, &[]);
                diagnostic_pass.set_bind_group(2, &self.size_bind_group, &[]);
                diagnostic_pass.dispatch_workgroups(self.backing_width.div_ceil(8), self.backing_height.div_ceil(8), 1);
            }
            self.sum_buffer(diagnostic_pass, active_bind_group, block_size as u32);
    
            encoder.copy_buffer_to_buffer(
                active_buffer, 
                0, 
                &staging_buffer, 
                0, 
                block_size * 4
            );
            
            self.queue.submit(std::iter::once(encoder.finish()));
            
            let buffer_clone = staging_buffer.clone();
            let (tx,rx) = mpsc::channel();
            staging_buffer.slice(0..(block_size * 4)).map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
    
            if let Ok(res) = rx.recv() {
                res.map_err(|err| {println!("{}",err)}).unwrap();
                let view = buffer_clone.slice(0..(block_size * 4)).get_mapped_range();
                let vec: &[f32] = bytemuck::cast_slice(&view);
                match i {
                    0 => println!("\nMomentum: x: {}, y: {}",vec[0],vec[1]),
                    1 => {
                        println!("KE        x: {}, y: {}, total: {}",vec[0] / 2.0,vec[1] / 2.0,(vec[0] + vec[1]) / 2.0);
                        total_energy += (vec[0] + vec[1]) / 2.0;
                    }
                    2 => {
                        println!("Heat:        {}",vec[0]);
                        total_energy += vec[0];
                    }
                    3 => println!("Diffuse   R: {}, G: {}, B: {}, A: {}, Total (excl A): {}",vec[0],vec[1],vec[2],vec[3], vec[0] + vec[1] + vec[2]),
                    _ => panic!("Error: invalid i")
                }
                drop(view);
                buffer_clone.unmap();
            }
        }
        println!("Total Energy: {}", total_energy);
    }

    fn pixel_diagnostics(&self, point: Point) {
        let staging_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Diagnostic Staging Buffer"),
            size: 16 + 8 + 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false
        }));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Diagnostic Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.diffuse_buffer, 4 * 4 * (point.0 as u64 + self.backing_width as u64 * point.1 as u64), &staging_buffer, 0, 16);
        encoder.copy_buffer_to_buffer(&self.velocity_buffer, 2 * 4 * (point.0 as u64 + self.backing_width as u64 * point.1 as u64), &staging_buffer, 16, 8);
        encoder.copy_buffer_to_buffer(&self.heat_buffer, 4 * (point.0 as u64 + self.backing_width as u64 * point.1 as u64), &staging_buffer, 24, 4);

        self.queue.submit(std::iter::once(encoder.finish()));

        let (tx,rx) = mpsc::channel();
        staging_buffer.slice(0..(16 + 8 + 4)).map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(res) = rx.recv() {
            res.map_err(|err| {println!("{}",err)}).unwrap();
            let view = staging_buffer.slice(0..(16 + 8 + 4)).get_mapped_range();
            let data: &[f32] = bytemuck::cast_slice(&view);
            let total = data[0] + data[1] + data[2];
            println!("\nDiffuse R: {}, G: {}, B: {}, A: {}, Total (excl A): {}", data[0],data[1],data[2],data[3],total);
            println!("Velocity X: {}, Y: {}", data[4], data[5]);
            println!("Momentum X: {}, Y: {}", data[4] * total, data[5] * total);
            println!("KE X: {}, Y: {}, Total: {}", data[4] * data[4] * total / 2.0, data[5] * data[5] * total / 2.0, (data[4] * data[4] + data[5] * data[5]) * total / 2.0);
            println!("Heat: {}, HE: {}",data[6], data[6] * total);
            println!("Total Energy: {}", ((data[4] * data[4] + data[5] * data[5]) / 2.0 + data[6]) * total);
        }
    }

    fn frame_apply(&mut self) {
        loop {
            if let Some(frame) = self.frame_buffer.next() {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Frame Apply Encoder"),
                }); 

                {
                    let gpu_frame = self.device.create_buffer_init(&BufferInitDescriptor{
                        label: Some("GPU Frame"),
                        contents: frame.data(0),
                        usage: wgpu::BufferUsages::STORAGE
                    });

                    let frame_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor{
                        label: Some("Frame Bind Group"),
                        layout: &self.frame_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry{
                                binding: 30,
                                resource: gpu_frame.as_entire_binding()
                            }
                        ]
                    });
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Compute Pass"),
                        timestamp_writes: None 
                    });   
                    compute_pass.set_pipeline(&self.frame_apply_pipeline);
                    compute_pass.set_bind_group(0, &self.size_bind_group, &[]);
                    compute_pass.set_bind_group(1, &self.gas_data_bind_group, &[]);
                    compute_pass.set_bind_group(2, &frame_bind_group, &[]);
                    compute_pass.set_bind_group(3, &self.mode_bind_group, &[]);
                    compute_pass.dispatch_workgroups(self.backing_width.div_ceil(8), self.backing_height.div_ceil(8), 1);
                }
                self.queue.submit(std::iter::once(encoder.finish()));
                break;
            } else {
                if let Some(buffer) = self.decoder.get_frame_buf() {
                    self.frame_buffer = buffer.into_iter();
                } else {
                    break;
                }
            }
        }
    }

    fn set_file_decode(&mut self, path: String) {
        self.decoder.set_file_decode(path); 
        self.frame_buffer = Vec::new().into_iter();
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

enum GPUCommand {
    Resize{width: u32, height: u32},
    Compute{num_ticks: u32},
    Movement,
    Diffusion,
    ToggleLoop,
    SetTickRate{num_ticks: u32},
    SetRenderMode{new_render_mode: u32},
    Line{line_points: Vec<Point>, line_config: LineConfig},
    Diagnostic,
    PixelDiagnostic{point: Point},
    FrameApply,
    ToggleFrameApply,
    SetFileDecode{path: String}
}

struct GPUSize {
    surface_width: u32,
    surface_height: u32,
    backing_width: u32,
    backing_height: u32
}

struct VideoDecoder {
    ictx: ffmpeg_next::format::context::Input,
    decoder: ffmpeg_next::codec::decoder::Video,
    scaler: ffmpeg_next::software::scaling::context::Context,
    video_stream_index: usize,
    frame_buf: Vec<ffmpeg_next::util::frame::Video>,
    intermediate_buf_frame: ffmpeg_next::util::frame::Video
}

impl VideoDecoder {
    fn new(output_width: u32, output_height: u32, path: String) -> Self {
        ffmpeg_next::init().unwrap();

        let ictx = ffmpeg_next::format::input(&path).unwrap();
        let input = ictx.streams().best(ffmpeg_next::media::Type::Video).unwrap();
        let video_stream_index = input.index();

        let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(input.parameters()).unwrap();
        let decoder = context_decoder.decoder().video().unwrap();

        let scaler = ffmpeg_next::software::scaling::context::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg_next::format::Pixel::GRAYF32LE,
            output_width,
            output_height,
            ffmpeg_next::software::scaling::flag::Flags::BILINEAR,
        ).unwrap();

        VideoDecoder {
            ictx,
            decoder,
            scaler,
            video_stream_index,
            frame_buf: Vec::new(),
            intermediate_buf_frame: ffmpeg_next::util::frame::Video::empty()
        }
    }

    fn process_frame(&mut self) -> bool {
        if self.frame_buf.len() < 90 {
            let mut packets = self.ictx.packets().filter(|(stream,_)| stream.index() == self.video_stream_index);
            while self.decoder.receive_frame(&mut self.intermediate_buf_frame).is_err() {
                match packets.next() {
                    Some((_,packet)) => {
                        self.decoder.send_packet(&packet).unwrap();
                    }
                    None => {return false;}
                }
            }
            let mut processed_frame = ffmpeg_next::util::frame::Video::empty();
            self.scaler.run(&self.intermediate_buf_frame, &mut processed_frame).unwrap();
            self.frame_buf.push(processed_frame);
        }
        true
    }

    fn dump_frame_buf(&mut self) -> Vec<ffmpeg_next::util::frame::Video> {
        std::mem::replace(&mut self.frame_buf, Vec::new())
    }
}

struct AsyncVideoDecoder {
    decode_thread_sender: mpsc::Sender<VideoDecoderCommand>,
    decode_thread_reciever: mpsc::Receiver<Option<Vec<ffmpeg_next::util::frame::Video>>>
}

impl AsyncVideoDecoder {
    fn new(output_width: u32, output_height: u32) -> Self {
        let (signal_tx,signal_rx) = mpsc::channel();
        let (data_tx, data_rx) = mpsc::channel();

        std::thread::spawn(move || {
            let mut decoder: Option<VideoDecoder> = None;
            loop {
                match signal_rx.recv_timeout(std::time::Duration::from_millis(match decoder {Some(_) => 10, None => 100})) {
                    Err(e) => {
                        match e {
                            mpsc::RecvTimeoutError::Disconnected => {break;}
                            mpsc::RecvTimeoutError::Timeout => {
                                if let Some(decoder_inner) = &mut decoder {
                                    if !decoder_inner.process_frame() {
                                        data_tx.send(Some(decoder_inner.dump_frame_buf())).unwrap();
                                        println!("Finished Playing File");
                                        decoder = None;
                                    }
                                }
                            }
                        }
                    }
                    Ok(command) => {
                        match command {
                            VideoDecoderCommand::DumpFrameBuf => {
                                if let Some(decoder) = &mut decoder {
                                    data_tx.send(Some(decoder.dump_frame_buf())).unwrap();
                                } else {
                                    data_tx.send(None).unwrap();
                                }
                            }
                            VideoDecoderCommand::StartFileDecode { path } => {
                                decoder = Some(VideoDecoder::new(output_width, output_height, path))
                            }
                        }
                    }
                }
            }
        });

        AsyncVideoDecoder {
            decode_thread_sender: signal_tx,
            decode_thread_reciever: data_rx
        }
    }

    fn get_frame_buf(&self) -> Option<Vec<ffmpeg_next::util::frame::Video>> {
        let _ = self.decode_thread_sender.send(VideoDecoderCommand::DumpFrameBuf);
        match self.decode_thread_reciever.recv() {
            Err(_) => {
                None
            }
            Ok(out) => {
                out
            }
        }
    }

    fn set_file_decode(&self, path: String) {
        let _ = self.decode_thread_sender.send(VideoDecoderCommand::StartFileDecode { path });
    }
}

enum VideoDecoderCommand {
    DumpFrameBuf,
    StartFileDecode{path: String}
}

struct App {
    idx: usize,
    window_id: Option<WindowId>,
    window: Option<Arc<Window>>,

    mouse_held: bool,
    last_mouse_pos: Option<Point>,
    line_points: Vec<Point>,
    line_config: LineConfig,

    gpu: Option<mpsc::Sender<GPUCommand>>,
    gpu_size: Option<GPUSize>,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        } else if !event_loop.exiting() {
            let window = Arc::new(event_loop.create_window(Window::default_attributes()).unwrap());
            window.set_max_inner_size(Some(PhysicalSize{width: 2048, height: 2048}));
            let size = window.inner_size();
            println!("Enter Backing Scale: ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            input.pop();
            let backing_scale = input.parse::<f32>().unwrap();
            println!("Backing width: {}, Backing Height: {}", (size.width as f32 * backing_scale) as u32, (size.height as f32 * backing_scale) as u32);
            let mut gpu = pollster::block_on(GPU::new(window.clone(),(size.width as f32 * backing_scale) as u32, (size.height as f32 * backing_scale) as u32));
            self.window = Some(window);
            let (tx,rx) = mpsc::channel();
            std::thread::spawn(move || {
                let mut num_ticks_per_frame = 60;
                let mut render_mode = 0;
                let timer = std::time::Instant::now();
                let framerate = 30.0;
                let mut apply_frames = false;
                let mut next_frame_time: Option<std::time::Duration> = None;
                loop {
                    match rx.recv_timeout(next_frame_time.map_or(std::time::Duration::from_millis(20), |next_frame_time| next_frame_time - std::cmp::min(timer.elapsed(), next_frame_time))) {
                        Err(e) => {
                            match e {
                                mpsc::RecvTimeoutError::Disconnected => {break;}
                                mpsc::RecvTimeoutError::Timeout => {
                                    if let Some(next_frame_time) = &mut next_frame_time {
                                        gpu.compute(num_ticks_per_frame,apply_frames);
                                        gpu.render(render_mode);
                                        *next_frame_time += std::time::Duration::from_secs_f64(1.0/framerate);
                                    }
                                }
                            }
                        }
                        Ok(command) => {
                            match command {
                                GPUCommand::Resize { width, height } => {gpu.resize(width, height); gpu.render(render_mode)}
                                GPUCommand::Compute { num_ticks} => {gpu.compute(num_ticks,apply_frames); gpu.render(render_mode)}
                                GPUCommand::Movement => {gpu.movement(); gpu.render(render_mode)}
                                GPUCommand::Diffusion => {gpu.diffusion(); gpu.render(render_mode)}
                                GPUCommand::ToggleLoop => {next_frame_time = match next_frame_time {
                                    None => Some(timer.elapsed()),
                                    Some(_) => None,
                                }}
                                GPUCommand::SetTickRate { num_ticks } => {num_ticks_per_frame = num_ticks}
                                GPUCommand::SetRenderMode { new_render_mode } => {render_mode = new_render_mode; gpu.render(render_mode)}
                                GPUCommand::Line { line_points, line_config} => {gpu.line(&line_points,line_config); gpu.render(render_mode)}
                                GPUCommand::Diagnostic => {gpu.print_diagnostics()}
                                GPUCommand::PixelDiagnostic{point} => {gpu.pixel_diagnostics(point)}
                                GPUCommand::FrameApply => {gpu.frame_apply(); gpu.render(render_mode);}
                                GPUCommand::ToggleFrameApply => {apply_frames = !apply_frames;}
                                GPUCommand::SetFileDecode{path} => {gpu.set_file_decode(path);}
                            }
                        }
                    }
                }
            });
            self.gpu = Some(tx);
            self.gpu_size = Some(GPUSize{
                surface_width: size.width, 
                surface_height: size.height,
                backing_width: (size.width as f32 * backing_scale) as u32,
                backing_height: (size.height as f32 * backing_scale) as u32
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
                println!(
                    "--------------------------------------------------------- Window {} Destroyed",
                    self.idx
                );
                self.window_id = None;
                event_loop.exit();
                return;
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
                                        gpu.send(GPUCommand::Compute{num_ticks: 60}).unwrap();
                                    }
                                }
                                "f" => {
                                    if let Some(gpu) = &mut self.gpu {
                                        gpu.send(GPUCommand::ToggleLoop).unwrap();
                                    }
                                }
                                "d" => {
                                    if let Some(gpu) = &mut self.gpu {
                                        gpu.send(GPUCommand::Diagnostic).unwrap();
                                    }
                                }
                                "e" => {
                                    if let Some(gpu) = &mut self.gpu {
                                        gpu.send(GPUCommand::Movement).unwrap();
                                    }
                                }
                                "q" => {
                                    if let Some(gpu) = &mut self.gpu {
                                        gpu.send(GPUCommand::Diffusion).unwrap();
                                    }
                                }
                                "v" => {
                                    println!("Enter Tick Rate: ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    if let Some(gpu) = &mut self.gpu {
                                        gpu.send(GPUCommand::SetTickRate { num_ticks: input.parse::<u32>().unwrap()}).unwrap();
                                    }
                                }
                                "w" => {
                                    if let Some(gpu) = &self.gpu {
                                        if let Some(size) = &self.gpu_size {
                                            gpu.send(GPUCommand::Line{
                                                line_points: std::mem::replace(&mut vec![
                                                    Point(0,0),
                                                    Point(0,size.backing_height),
                                                    Point(size.backing_width,size.backing_height),
                                                    Point(size.backing_width,0),
                                                    Point(0,0)
                                                ], Vec::new()), 
                                                line_config: self.line_config.clone()
                                            }).unwrap();
                                        }
                                    }
                                }
                                "1" => {
                                    println!("Enter Red Density: ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    self.line_config.color[0] = input.parse::<f32>().unwrap();
                                }
                                "2" => {
                                    println!("Enter Green Density: ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    self.line_config.color[1] = input.parse::<f32>().unwrap();
                                }
                                "3" => {
                                    println!("Enter Blue Density: ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    self.line_config.color[2] = input.parse::<f32>().unwrap();
                                }
                                "4" => {
                                    println!("Enter X Velocity: ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    self.line_config.velocity[0] = input.parse::<f32>().unwrap();
                                }
                                "5" => {
                                    println!("Enter Y Velocity: ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    self.line_config.velocity[1] = input.parse::<f32>().unwrap();
                                }
                                "6" => {
                                    println!("Enter Heat: ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    let heat = input.parse::<f32>().unwrap();
                                    self.line_config.heat = heat * heat;
                                }
                                "7" => {
                                    println!("Enter Mode Target (0: color, 1: velocity, 2: heat, 3: wall): ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    let mode_target = input.parse::<u32>().unwrap();
                                    println!("Enter Mode: ");
                                    input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    let mode = input.parse::<u32>().unwrap();
                                    match (mode_target, mode) {
                                        (0,0) => {self.line_config.color_mode = ColorMode::Default},
                                        (0,1) => {self.line_config.color_mode = ColorMode::Static},
                                        (0,2) => {self.line_config.color_mode = ColorMode::Paint},
                                        (1,0) => {self.line_config.static_velocity = false},
                                        (1,1) => {self.line_config.static_velocity = true},
                                        (2,0) => {self.line_config.static_heat = false},
                                        (2,1) => {self.line_config.static_heat = true},
                                        (3,0) => {self.line_config.is_wall = false},
                                        (3,1) => {self.line_config.is_wall = true},
                                        (_,_) => println!("Unregonized mode target / mode")
                                    }
                                }
                                "8" => {
                                    println!("Enter Render Mode: ");
                                    let mut input = String::new();
                                    std::io::stdin().read_line(&mut input).unwrap();
                                    input.pop();
                                    let new_render_mode = input.parse::<u32>().unwrap();
                                    if let Some(gpu) = &mut self.gpu {
                                        gpu.send(GPUCommand::SetRenderMode { new_render_mode }).unwrap();
                                    }
                                }
                                "g" => {
                                    if let Some(gpu) = &mut self.gpu {
                                        gpu.send(GPUCommand::FrameApply).unwrap();
                                    }
                                }
                                "t" => {
                                    if let Some(gpu) = &mut self.gpu {
                                        gpu.send(GPUCommand::ToggleFrameApply).unwrap();
                                    }
                                }
                                "b" => {
                                    if let Some(gpu) = &mut self.gpu {
                                        println!("Enter Path");
                                        let mut input = String::new();
                                        std::io::stdin().read_line(&mut input).unwrap();
                                        input.pop();
                                        let _ = gpu.send(GPUCommand::SetFileDecode { path: input });
                                    }
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
                    let x = position.x * size.backing_width as f64 / size.surface_width as f64;
                    let y = position.y * size.backing_height as f64 / size.surface_height as f64;
                    self.last_mouse_pos = Some(Point(x as u32, y as u32));
                    if self.mouse_held {
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
                                        line_config: self.line_config.clone()
                                    }).unwrap();
                                }
                            }
                        }
                    }
                    MouseButton::Right => {
                        if state == ElementState::Pressed {
                            if let Some(gpu) = &self.gpu {
                                if let Some(point) = self.last_mouse_pos {
                                    gpu.send(GPUCommand::PixelDiagnostic { point }).unwrap();
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
        last_mouse_pos: None,
        line_points: Vec::new(), 
        line_config: LineConfig {
            color: [1.0,1.0,1.0,1.0], 
            velocity: [0.0, 0.0],
            heat: 0.0,
            color_mode: ColorMode::Default,
            static_velocity: false,
            static_heat: false,
            is_wall: false
        },

        gpu: None, 
        gpu_size: None,
    };
    event_loop.run_app(&mut app).unwrap();
    println!("test");
    println!("test");
}