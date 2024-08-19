@group(0) @binding(0) var<uniform> size: Size;


@group(1) @binding(0) var<storage,read_write> diffuse_storage: array<vec4<f32>>;
@group(1) @binding(1) var<storage,read_write> velocity_buffer: array<vec2<f32>>;
@group(1) @binding(2) var<storage,read_write> heat_buffer: array<f32>;

@group(1) @binding(10) var diffuse: texture_2d<f32>;
@group(1) @binding(11) var texture_sampler: sampler;


@group(2) @binding(0) var<storage,read_write> secondary_diffuse_storage: array<vec4<f32>>;
@group(2) @binding(1) var<storage,read_write> secondary_velocity_buffer: array<vec2<f32>>;
@group(2) @binding(2) var<storage,read_write> secondary_heat_buffer: array<f32>;

@group(2) @binding(10) var<storage,read> line_points: array<vec2<u32>>;
@group(2) @binding(11) var<uniform> line_config: LineConfig;


@group(3) @binding(0) var<storage,read_write> wall_buffer: array<u32>;

struct Size {
    backing_width: u32,
    backing_height: u32,
    pad_width: u32
}

struct LineConfig {
    color: vec4<f32>,
    velocity: vec2<f32>,
    heat: f32,
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
    // code for managing high pressures, puts the values on a curve --> as it increases, more increase is needed to produce an equal change
    //let full_base_diffuse = textureSample(diffuse,texture_sampler,position.tex_coords);
    //var base_diffuse = vec3(full_base_diffuse.r,full_base_diffuse.g,full_base_diffuse.b);
    //base_diffuse = 1.0 - exp2(-base_diffuse);
    //return vec4(base_diffuse,1.0);
    return textureSample(diffuse,texture_sampler,position.tex_coords);
}

fn pad_index_2d(point: vec2<u32>) -> u32 {
    return point.x + size.pad_width * point.y;
}

fn index_2d(point: vec2<u32>) -> u32 {
    return point.x + size.backing_width * point.y;
}

fn wall_buffer_get_2d(point: vec2<u32>) -> bool {
    let index = index_2d(point);
    return bool((wall_buffer[index / 8] >> (index % 8)) & 0x00000001);
}

fn wall_buffer_set_2d(point: vec2<u32>, val: bool) {
    let index = index_2d(point);
    let arr_index = index / 8;
    let shifted_val = u32(val) << (index % 8);
    wall_buffer[arr_index] &= (shifted_val ^ 0x11111111); // set targetted bit to 0
    wall_buffer[arr_index] |= shifted_val; //set targetted bit to val
}

//make sure to keep in sync with util_shader.wgsl
fn total_diffuse(diffuse: vec4<f32>) -> f32 {
    return diffuse.x + diffuse.y + diffuse.z;
} 

fn magnitude(velocity: vec2<f32>) -> f32 {
    return sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
}

fn sum(vector: vec2<f32>) -> f32 {
    return vector.x + vector.y;
}


@compute
@workgroup_size(8,8)
fn diffusion_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= size.backing_width | global_id.y >= size.backing_height {
        return;
    }

    let dimensions = vec2<i32>(i32(size.backing_width),i32(size.backing_height));
    let base_location = vec2<i32>(i32(global_id.x),i32(global_id.y)) + dimensions;

    var aggregate_diffuse = vec4(0.0,0.0,0.0,0.0);
    var aggregate_velocity = vec2(0.0,0.0);
    var expected_velocity_energy = vec2(0.0,0.0);
    var aggregate_heat = 0.0;

    var location = vec2<u32>(0u,0u);
    var heat = 0.0;
    var new_diffuse = vec4(0.0,0.0,0.0,0.0);
    var diffuse_total = 0.0;
    var velocity = vec2(0.0,0.0);

    location = vec2<u32>((base_location + vec2(-1,-1)) % dimensions);
    heat = heat_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * (heat / 16.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index_2d(location)] + vec2(inverseSqrt(2.0),inverseSqrt(2.0)) * sqrt(heat);
    aggregate_velocity += velocity * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;

    location = vec2<u32>((base_location + vec2(-1,0)) % dimensions);
    heat = heat_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * ((sqrt(heat) * (1.0 + sqrt(2.0)) - heat) / 8.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index_2d(location)] + ((vec2(1.0,0.0) * heat) + (vec2(inverseSqrt(2.0),0.0) * sqrt(heat)) * (sqrt(2.0 * heat) - heat)); // can be simplified
    aggregate_velocity += velocity * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += inverseSqrt(2.0) * heat * (sqrt(2.0 * heat) - heat) * diffuse_total;

    location = vec2<u32>((base_location + vec2(-1,1)) % dimensions);
    heat = heat_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * (heat / 16.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index_2d(location)] + vec2(inverseSqrt(2.0),-inverseSqrt(2.0)) * sqrt(heat);
    aggregate_velocity += velocity * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,-1)) % dimensions);
    heat = heat_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * ((sqrt(heat) * (1.0 + sqrt(2.0)) - heat) / 8.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index_2d(location)] + ((vec2(0.0,1.0) * heat) + (vec2(0.0,inverseSqrt(2.0)) * sqrt(heat)) * (sqrt(2.0 * heat) - heat)); // can be simplified
    aggregate_velocity += velocity * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += inverseSqrt(2.0) * heat * (sqrt(2.0 * heat) - heat) * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,0)) % dimensions);
    heat = heat_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * ((4.0 - sqrt(heat) * (2.0 + 2.0 * sqrt(2.0)) + heat) / 4.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index_2d(location)];
    aggregate_velocity += velocity * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,1)) % dimensions);
    heat = heat_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * ((sqrt(heat) * (1.0 + sqrt(2.0)) - heat) / 8.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index_2d(location)] + ((vec2(0.0,-1.0) * heat) + (vec2(0.0,-inverseSqrt(2.0)) * sqrt(heat)) * (sqrt(2.0 * heat) - heat)); // can be simplified
    aggregate_velocity += velocity * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += inverseSqrt(2.0) * heat * (sqrt(2.0 * heat) - heat) * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,-1)) % dimensions);
    heat = heat_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * (heat / 16.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index_2d(location)] + vec2(-inverseSqrt(2.0),inverseSqrt(2.0)) * sqrt(heat);
    aggregate_velocity += velocity * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,0)) % dimensions);
    heat = heat_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * ((sqrt(heat) * (1.0 + sqrt(2.0)) - heat) / 8.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index_2d(location)] + ((vec2(-1.0,0.0) * heat) + (vec2(-inverseSqrt(2.0),0.0) * sqrt(heat)) * (sqrt(2.0 * heat) - heat)); // can be simplified
    aggregate_velocity += velocity * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += inverseSqrt(2.0) * heat * (sqrt(2.0 * heat) - heat) * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,1)) % dimensions);
    heat = heat_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * (heat / 16.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index_2d(location)] + vec2(-inverseSqrt(2.0),-inverseSqrt(2.0)) * sqrt(heat);
    aggregate_velocity += velocity * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    


    location = vec2<u32>((base_location % dimensions));
    diffuse_total = total_diffuse(aggregate_diffuse);
    secondary_diffuse_storage[pad_index_2d(location)] = aggregate_diffuse;
    if diffuse_total > 0.0 {
        aggregate_velocity = clamp(aggregate_velocity / diffuse_total,vec2(-1.0,-1.0),vec2(1.0,1.0)); // clamped because I **BELIEVE** that it can be pushed to inifinite speeds at ~0 mass fringes
        secondary_velocity_buffer[index_2d(location)] = aggregate_velocity;
        secondary_heat_buffer[index_2d(location)] = clamp((aggregate_heat  / diffuse_total + sum(expected_velocity_energy  / diffuse_total - aggregate_velocity * aggregate_velocity)) / 2.0,0.0,1.0); // discreprency between real & expected vel energy becomes Heat
    } else {
        secondary_velocity_buffer[index_2d(location)] = vec2(0.0,0.0);
        secondary_heat_buffer[index_2d(location)] = 0.0;
    }
}


@compute
@workgroup_size(8,8)
fn movement_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= size.backing_width | global_id.y >= size.backing_height {
        return;
    }

    let dimensions = vec2<i32>(i32(size.backing_width),i32(size.backing_height));
    let base_location = vec2<i32>(i32(global_id.x),i32(global_id.y)) + dimensions;

    var aggregate_diffuse = vec4(0.0,0.0,0.0,0.0);
    var aggregate_velocity = vec2(0.0,0.0);
    var expected_velocity_energy = vec2(0.0,0.0);
    var aggregate_heat = 0.0;

    var location = vec2<u32>((base_location + vec2(-1,-1)) % dimensions);
    var velocity = velocity_buffer[index_2d(location)];
    var new_diffuse = diffuse_storage[pad_index_2d(location)] * (max(velocity.x,0.0) * max(velocity.y,0.0));
    var diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; // signed square
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index_2d(location)] * diffuse_total;

    location = vec2<u32>((base_location + vec2(-1,0)) % dimensions);
    velocity = velocity_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * (max(velocity.x,0.0) * (1.0 - abs(velocity.y)));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; // signed square
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index_2d(location)] * diffuse_total;

    location = vec2<u32>((base_location + vec2(-1,1)) % dimensions);
    velocity = velocity_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * (max(velocity.x,0.0) * max(-velocity.y,0.0));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; // signed square
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index_2d(location)] * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,-1)) % dimensions);
    velocity = velocity_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * ((1.0 - abs(velocity.x)) * max(velocity.y,0.0));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; // signed square
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index_2d(location)] * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,0)) % dimensions);
    velocity = velocity_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * ((1.0 - abs(velocity.x)) * (1.0 - abs(velocity.y)));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; // signed square
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index_2d(location)] * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,1)) % dimensions);
    velocity = velocity_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * ((1.0 - abs(velocity.x)) * max(-velocity.y,0.0));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; // signed square
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index_2d(location)] * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,-1)) % dimensions);
    velocity = velocity_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * (max(-velocity.x,0.0) * max(velocity.y,0.0));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; // signed square
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index_2d(location)] * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,0)) % dimensions);
    velocity = velocity_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * (max(-velocity.x,0.0) * (1.0 - abs(velocity.y)));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; // signed square
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index_2d(location)] * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,1)) % dimensions);
    velocity = velocity_buffer[index_2d(location)];
    new_diffuse = diffuse_storage[pad_index_2d(location)] * (max(-velocity.x,0.0) * max(-velocity.y,0.0));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; // signed square
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index_2d(location)] * diffuse_total;
    
    location = vec2<u32>((base_location % dimensions));
    diffuse_total = total_diffuse(aggregate_diffuse);
    secondary_diffuse_storage[pad_index_2d(location)] = aggregate_diffuse;
    if diffuse_total > 0.0 {
        aggregate_velocity /= diffuse_total;
        secondary_velocity_buffer[index_2d(location)] = aggregate_velocity;
        secondary_heat_buffer[index_2d(location)] = clamp((aggregate_heat  / diffuse_total + sum(expected_velocity_energy  / diffuse_total - aggregate_velocity * aggregate_velocity) / 2.0),0.0,1.0); // discreprency between real & expected vel energy becomes Heat
    } else {
        secondary_velocity_buffer[index_2d(location)] = vec2(0.0,0.0);
        secondary_heat_buffer[index_2d(location)] = 0.0;
    }
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
        diffuse_storage[rounded_current_point.x + size.pad_width * rounded_current_point.y] = line_config.color;
        velocity_buffer[index_2d(rounded_current_point)] = line_config.velocity;
        heat_buffer[index_2d(rounded_current_point)] = line_config.heat;
        //diffuse_storage[(rounded_current_point.x + 1) + size.pad_width * rounded_current_point.y] = pack4xU8(vec4(255u,255u,255u,255u));
        //diffuse_storage[(rounded_current_point.x - 1) + size.pad_width * rounded_current_point.y] = pack4xU8(vec4(255u,255u,255u,255u));
        //diffuse_storage[rounded_current_point.x + size.pad_width * (rounded_current_point.y + 1)] = pack4xU8(vec4(255u,255u,255u,255u));
        //diffuse_storage[rounded_current_point.x + size.pad_width * (rounded_current_point.y - 1)] = pack4xU8(vec4(255u,255u,255u,255u));
        current_point += delta_point;
    }
    
}

 