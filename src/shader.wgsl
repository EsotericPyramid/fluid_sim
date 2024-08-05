@group(0)
@binding(0)
var<uniform> size: Size;

@group(1)
@binding(0)
var<storage,read_write> diffuse_storage: array<u32>;

@group(1)
@binding(10)
var diffuse: texture_2d<f32>;

@group(1)
@binding(11)
var texture_sampler: sampler;

@group(2)
@binding(0)
var<storage,read> line_points: array<vec2<u32>>;

@group(2)
@binding(10)
var<storage,read_write> secondary_diffuse_storage: array<u32>;

struct Size {
    backing_width: u32,
    backing_height: u32,
    pad_width: u32
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) tex_coords: vec2<f32>
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    switch in_vertex_index {
        case 0u: {
            out.clip_position = vec4<f32>(-1.0,1.0,0.0,1.0); 
            out.tex_coords = vec2<f32>(0.0,0.0);
        }
        case 1u: {
            out.clip_position = vec4<f32>(-1.0,-1.0,0.0,1.0); 
            out.tex_coords = vec2<f32>(0.0,1.0);
        }
        case 2u: {
            out.clip_position = vec4<f32>(1.0,1.0,0.0,1.0); 
            out.tex_coords = vec2<f32>(1.0,0.0);
        }
        case 3u: {
            out.clip_position = vec4<f32>(1.0,-1.0,0.0,1.0); 
            out.tex_coords = vec2<f32>(1.0,1.0);
        }
        case 4u: {
            out.clip_position = vec4<f32>(1.0,1.0,0.0,1.0); 
            out.tex_coords = vec2<f32>(1.0,0.0);
        }
        case 5u: {
            out.clip_position = vec4<f32>(-1.0,-1.0,0.0,1.0); 
            out.tex_coords = vec2<f32>(0.0,1.0);
        }
        default: {}
    }
    return out;
}

@fragment
fn fs_main(position: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(diffuse,texture_sampler,position.tex_coords);
}

fn unpack4xU8(rgba8_packed: u32) -> vec4<u32> {
    return vec4(
        rgba8_packed & 0x000000ff,
        (rgba8_packed >> 8) & 0x000000ff,
        (rgba8_packed >> 16) & 0x000000ff,
        (rgba8_packed >> 24) & 0x000000ff
    );
}

fn index_2d(point: vec2<u32>) -> u32 {
    return point.x + size.pad_width * point.y;
}

@compute
@workgroup_size(8,8)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    //var x = f32((global_id.x % 8)) - 4.0;
    //var y = f32((global_id.y % 8)) - 4.0;
    //let red = u32(sqrt(((x*x)+(y*y))/32.0) * 255.0);
    //x = f32((global_id.x % 40)) - 20.0;
    //y = f32((global_id.y % 40)) - 20.0;
    //let green = u32(sqrt(((x*x)+(y*y))/800.0) * 255.0);
    //x = f32((global_id.x % 200)) - 100.0;
    //y = f32((global_id.y % 200)) - 100.0;
    //let blue = u32(sqrt(((x*x)+(y*y))/20000.0) * 255.0);
    //diffuse_storage[global_id.x + size.pad_width * global_id.y] = pack4xU8(
    //    vec4(red,green,blue,255u) / 2 + 
    //    unpack4xU8(diffuse_storage[global_id.x + size.pad_width * global_id.y]) / 2
    //);
    let dimensions = vec2(size.backing_width,size.backing_height);
    let location = vec2(global_id.x,global_id.y) + dimensions;
    secondary_diffuse_storage[index_2d(location % dimensions)] = pack4xU8((
        unpack4xU8(diffuse_storage[index_2d(location % dimensions)]) +
        unpack4xU8(diffuse_storage[index_2d((location + vec2(1,0)) % dimensions)]) +
        unpack4xU8(diffuse_storage[index_2d((location - vec2(1,0)) % dimensions)]) +
        unpack4xU8(diffuse_storage[index_2d((location + vec2(0,1)) % dimensions)]) +
        unpack4xU8(diffuse_storage[index_2d((location - vec2(0,1)) % dimensions)]) 
    ) / 5);
}


@compute
@workgroup_size(64)
fn line_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= arrayLength(&line_points) - 1 {
        return;
    }
    let start_point = vec2<i32>(line_points[global_id.x]);
    let end_point = vec2<i32>(line_points[global_id.x + 1]);
    let num_iterations = max(abs(end_point.x - start_point.x),abs(end_point.y - start_point.y));
    let delta_x = f32(end_point.x - start_point.x);
    let delta_y = f32(end_point.y - start_point.y);
    let delta_point = vec2(delta_x,delta_y) / f32(num_iterations);
    var current_point = vec2<f32>(start_point);
    for (var i = 0; i < num_iterations; i++) {
        let rounded_current_point = vec2<u32>(round(current_point));
        diffuse_storage[rounded_current_point.x + size.pad_width * rounded_current_point.y] = pack4xU8(vec4(255u,255u,255u,255u));
        //diffuse_storage[(rounded_current_point.x + 1) + size.pad_width * rounded_current_point.y] = pack4xU8(vec4(255u,255u,255u,255u));
        //diffuse_storage[(rounded_current_point.x - 1) + size.pad_width * rounded_current_point.y] = pack4xU8(vec4(255u,255u,255u,255u));
        //diffuse_storage[rounded_current_point.x + size.pad_width * (rounded_current_point.y + 1)] = pack4xU8(vec4(255u,255u,255u,255u));
        //diffuse_storage[rounded_current_point.x + size.pad_width * (rounded_current_point.y - 1)] = pack4xU8(vec4(255u,255u,255u,255u));
        current_point += delta_point;
    }
    
}

 