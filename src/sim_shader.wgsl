@group(0) @binding(0) var<uniform> size: Size;


@group(1) @binding(0) var<storage,read_write> diffuse_buffer: array<vec4<f32>>;
@group(1) @binding(1) var<storage,read_write> velocity_buffer: array<vec2<f32>>;
@group(1) @binding(2) var<storage,read_write> heat_buffer: array<f32>;

@group(1) @binding(10) var diffuse: texture_2d<f32>;
@group(1) @binding(11) var texture_sampler: sampler;


@group(2) @binding(0) var<storage,read_write> secondary_diffuse_buffer: array<vec4<f32>>;
@group(2) @binding(1) var<storage,read_write> secondary_velocity_buffer: array<vec2<f32>>;
@group(2) @binding(2) var<storage,read_write> secondary_heat_buffer: array<f32>;

@group(2) @binding(10) var<storage,read> line_points: array<vec2<u32>>;
@group(2) @binding(11) var<uniform> line_config: LineConfig;

@group(2) @binding(20) var diffuse_texture_storage: texture_storage_2d<rgba8unorm,write>;
@group(2) @binding(21) var<uniform> copy_mode: u32;


@group(3) @binding(0) var<storage,read_write> mode_buffer: array<atomic<u32>>;

const inverse_sqrt_2 = inverseSqrt(2.0);
const sqrt_2 = sqrt(2.0);
const sqrt_2_plus_1 = 1.0 + sqrt_2;
const pi = 3.141592658389;

struct Size {
    backing_width: u32,
    backing_height: u32,
    pad_width: u32
}

struct LineConfig {
    color: vec4<f32>,
    velocity: vec2<f32>,
    heat: f32,
    mode: u32, // 0: gas, 1: wall
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) tex_coords: vec2<f32>
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
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
    let full_base_diffuse = textureSample(diffuse,texture_sampler,position.tex_coords);
    var base_diffuse = vec3(full_base_diffuse.r,full_base_diffuse.g,full_base_diffuse.b);
    // code for managing high pressures, puts the values on a curve --> as it increases, more increase is needed to produce an equal change
    //base_diffuse = 1.0 - exp2(-base_diffuse);
    return vec4(base_diffuse,1.0);
}

fn pad_index_2d(point: vec2<u32>) -> u32 {
    return point.x + size.pad_width * point.y;
}

fn index_2d(point: vec2<u32>) -> u32 {
    return point.x + size.backing_width * point.y;
}

fn mode_buffer_get_2d(point: vec2<u32>) -> u32 {
    let bit_index = 4 * index_2d(point);
    return (atomicLoad(&mode_buffer[bit_index / 32]) >> (bit_index % 32)) & 0x0000000f;
}

fn mode_buffer_set_2d(point: vec2<u32>, val: u32) {
    let bit_index = 4 * index_2d(point);
    let arr_index = bit_index / 32;
    let shifted_val = (val & 0x0000000f) << (bit_index % 32);
    atomicAnd(&mode_buffer[arr_index], ((0x0000000fu << (bit_index % 32)) ^ 0xffffffffu)); // set targetted bits to 0
    atomicOr(&mode_buffer[arr_index], shifted_val); //set targetted bits to val
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
    
    let mode = mode_buffer_get_2d(vec2(global_id.x,global_id.y));
    // dont try to put gas in a wall
    if (mode & 0x00000003) == 0x00000003 {
        return;
    }

    let dimensions = vec2<i32>(i32(size.backing_width),i32(size.backing_height));
    let base_location = vec2<i32>(i32(global_id.x),i32(global_id.y)) + dimensions;

    var location = vec2<u32>((base_location + vec2(-1,-1)) % dimensions);
    var wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    var index = index_2d(location);
    var heat = heat_buffer[index];
    var new_diffuse = diffuse_buffer[pad_index_2d(location)] * (heat / 16.0);
    var diffuse_total = total_diffuse(new_diffuse);
    var aggregate_diffuse = new_diffuse;
    var velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    var aggregate_velocity = (velocity + vec2(inverse_sqrt_2,inverse_sqrt_2) * sqrt(heat)) * diffuse_total;
    var expected_velocity_energy = velocity * velocity * diffuse_total;
    var aggregate_heat = heat * diffuse_total;

    location = vec2<u32>((base_location + vec2(-1,0)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    heat = heat_buffer[index];
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * ((sqrt(heat) * (sqrt_2_plus_1) - heat) / 8.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    aggregate_velocity += (velocity + (vec2(heat * (2.0 - inverse_sqrt_2 * sqrt(heat)),0.0) )) * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat * diffuse_total;

    location = vec2<u32>((base_location + vec2(-1,1)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    heat = heat_buffer[index];
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * (heat / 16.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    aggregate_velocity += (velocity + vec2(inverse_sqrt_2,-inverse_sqrt_2) * sqrt(heat)) * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,-1)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    heat = heat_buffer[index];
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * ((sqrt(heat) * (sqrt_2_plus_1) - heat) / 8.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    aggregate_velocity += (velocity + (vec2(0.0,heat * (2.0 - inverse_sqrt_2 * sqrt(heat))) )) * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,0)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    heat = heat_buffer[index];
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * ((4.0 - sqrt(heat) * (2.0 * sqrt_2_plus_1) + heat) / 4.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    aggregate_velocity += velocity * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,1)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    heat = heat_buffer[index];
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * ((sqrt(heat) * (sqrt_2_plus_1) - heat) / 8.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    aggregate_velocity += (velocity + (vec2(0.0,heat * (inverse_sqrt_2 * sqrt(heat) - 2.0)) )) * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,-1)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    heat = heat_buffer[index];
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * (heat / 16.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    aggregate_velocity += (velocity + vec2(-inverse_sqrt_2,inverse_sqrt_2) * sqrt(heat)) * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,0)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    heat = heat_buffer[index];
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * ((sqrt(heat) * (sqrt_2_plus_1) - heat) / 8.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    aggregate_velocity += (velocity + (vec2(heat * (inverse_sqrt_2 * sqrt(heat) - 2.0),0.0) )) * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,1)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    heat = heat_buffer[index];
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * (heat / 16.0);
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    aggregate_velocity += (velocity + vec2(-inverse_sqrt_2,-inverse_sqrt_2) * sqrt(heat)) * diffuse_total;
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat * diffuse_total;
    

    let static_diffuse = bool(mode & 0x00000001);
    let paint_diffuse = bool((mode >> 1) & 0x00000001);
    let static_vel = bool((mode >> 2) & 0x00000001);
    let static_heat = bool((mode >> 3) & 0x00000001);
    location = vec2<u32>((base_location % dimensions));
    let previous_diffuse = diffuse_buffer[pad_index_2d(location)];
    aggregate_diffuse = select(aggregate_diffuse, previous_diffuse, static_diffuse);
    diffuse_total = total_diffuse(aggregate_diffuse);
    //paint step
    if total_diffuse(previous_diffuse) > 0.0 {
        aggregate_diffuse = select(aggregate_diffuse, previous_diffuse * (diffuse_total / total_diffuse(previous_diffuse)), paint_diffuse);
    }
    secondary_diffuse_buffer[pad_index_2d(location)] = aggregate_diffuse;
    index = index_2d(location);
    if diffuse_total > 0.0 {
        aggregate_velocity = clamp(aggregate_velocity / diffuse_total,vec2(-1.0,-1.0),vec2(1.0,1.0)); // clamped because I **BELIEVE** that it can be pushed to inifinite speeds at ~0 mass fringes
        secondary_velocity_buffer[index] = select(aggregate_velocity, velocity_buffer[index], static_vel);
        secondary_heat_buffer[index] = select(
            clamp((aggregate_heat  / diffuse_total + sum(expected_velocity_energy  / diffuse_total - aggregate_velocity * aggregate_velocity) / 2.0),0.0,1.0),  // discreprency between real & expected vel energy becomes Heat
            heat_buffer[index],
            static_heat
        );
    } else {
        secondary_velocity_buffer[index] = select(vec2(0.0,0.0),velocity_buffer[index],static_vel);
        secondary_heat_buffer[index] = select(0.0,heat_buffer[index],static_heat);
    }
}


@compute
@workgroup_size(8,8)
fn movement_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= size.backing_width | global_id.y >= size.backing_height {
        return;
    }

    let mode = mode_buffer_get_2d(vec2(global_id.x,global_id.y));
    // dont try to put gas in a wall
    if (mode & 0x00000003) == 0x00000003 {
        return;
    }

    let dimensions = vec2<i32>(i32(size.backing_width),i32(size.backing_height));
    let base_location = vec2<i32>(i32(global_id.x),i32(global_id.y)) + dimensions;

    var location = vec2<u32>((base_location + vec2(-1,-1)) % dimensions);
    var wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    var index = index_2d(location);
    var velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    var new_diffuse = diffuse_buffer[pad_index_2d(location)] * (max(velocity.x,0.0) * max(velocity.y,0.0));
    var diffuse_total = total_diffuse(new_diffuse);
    var aggregate_diffuse = new_diffuse;
    var aggregate_velocity = velocity * diffuse_total; 
    var expected_velocity_energy = velocity * velocity * diffuse_total;
    var aggregate_heat = heat_buffer[index] * diffuse_total;

    location = vec2<u32>((base_location + vec2(-1,0)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * (max(velocity.x,0.0) * (1.0 - abs(velocity.y)));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; 
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index] * diffuse_total;

    location = vec2<u32>((base_location + vec2(-1,1)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * (max(velocity.x,0.0) * max(-velocity.y,0.0));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; 
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index] * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,-1)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * ((1.0 - abs(velocity.x)) * max(velocity.y,0.0));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; 
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index] * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,0)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * ((1.0 - abs(velocity.x)) * (1.0 - abs(velocity.y)));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; 
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index] * diffuse_total;

    location = vec2<u32>((base_location + vec2(0,1)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * ((1.0 - abs(velocity.x)) * max(-velocity.y,0.0));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; 
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index] * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,-1)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * (max(-velocity.x,0.0) * max(velocity.y,0.0));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; 
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index] * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,0)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * (max(-velocity.x,0.0) * (1.0 - abs(velocity.y)));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; 
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index] * diffuse_total;

    location = vec2<u32>((base_location + vec2(1,1)) % dimensions);
    wall_collision = (mode_buffer_get_2d(location) & 0x00000003) == 0x00000003;
    location = select(location,vec2<u32>(base_location % dimensions),wall_collision);
    index = index_2d(location);
    velocity = velocity_buffer[index] * select(1.0,-1.0,wall_collision);
    new_diffuse = diffuse_buffer[pad_index_2d(location)] * (max(-velocity.x,0.0) * max(-velocity.y,0.0));
    diffuse_total = total_diffuse(new_diffuse);
    aggregate_diffuse += new_diffuse;
    aggregate_velocity += velocity * diffuse_total; 
    expected_velocity_energy += velocity * velocity * diffuse_total;
    aggregate_heat += heat_buffer[index] * diffuse_total;
    
    let static_diffuse = bool(mode & 0x00000001);
    let paint_diffuse = bool((mode >> 1) & 0x00000001);
    let static_vel = bool((mode >> 2) & 0x00000001);
    let static_heat = bool((mode >> 3) & 0x00000001);
    location = vec2<u32>((base_location % dimensions));
    let previous_diffuse = diffuse_buffer[pad_index_2d(location)];
    aggregate_diffuse = select(aggregate_diffuse, previous_diffuse, static_diffuse);
    diffuse_total = total_diffuse(aggregate_diffuse);
    //paint step
    if total_diffuse(previous_diffuse) > 0.0 {
        aggregate_diffuse = select(aggregate_diffuse, previous_diffuse * (diffuse_total / total_diffuse(previous_diffuse)), paint_diffuse);
    }
    secondary_diffuse_buffer[pad_index_2d(location)] = aggregate_diffuse;
    index = index_2d(location);
    if diffuse_total > 0.0 {
        aggregate_velocity /= diffuse_total;
        secondary_velocity_buffer[index] = select(aggregate_velocity, velocity_buffer[index], static_vel);
        secondary_heat_buffer[index] = select(
            clamp((aggregate_heat  / diffuse_total + sum(expected_velocity_energy  / diffuse_total - aggregate_velocity * aggregate_velocity) / 2.0),0.0,1.0),  // discreprency between real & expected vel energy becomes Heat
            heat_buffer[index],
            static_heat
        );
    } else {
        secondary_velocity_buffer[index] = select(vec2(0.0,0.0),velocity_buffer[index],static_vel);
        secondary_heat_buffer[index] = select(0.0,heat_buffer[index],static_heat);
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
        diffuse_buffer[pad_index_2d(rounded_current_point)] = line_config.color;
        diffuse_buffer[pad_index_2d(rounded_current_point + vec2(1,0))] = line_config.color;
        diffuse_buffer[pad_index_2d(rounded_current_point - vec2(1,0))] = line_config.color;
        diffuse_buffer[pad_index_2d(rounded_current_point + vec2(0,1))] = line_config.color;
        diffuse_buffer[pad_index_2d(rounded_current_point - vec2(0,1))] = line_config.color;
        velocity_buffer[index_2d(rounded_current_point)] = line_config.velocity;
        velocity_buffer[index_2d(rounded_current_point + vec2(1,0))] = line_config.velocity;
        velocity_buffer[index_2d(rounded_current_point - vec2(1,0))] = line_config.velocity;
        velocity_buffer[index_2d(rounded_current_point + vec2(0,1))] = line_config.velocity;
        velocity_buffer[index_2d(rounded_current_point - vec2(0,1))] = line_config.velocity;
        heat_buffer[index_2d(rounded_current_point)] = line_config.heat;
        heat_buffer[index_2d(rounded_current_point + vec2(1,0))] = line_config.heat;
        heat_buffer[index_2d(rounded_current_point - vec2(1,0))] = line_config.heat;
        heat_buffer[index_2d(rounded_current_point + vec2(0,1))] = line_config.heat;
        heat_buffer[index_2d(rounded_current_point - vec2(0,1))] = line_config.heat;
        mode_buffer_set_2d(rounded_current_point,line_config.mode);
        mode_buffer_set_2d(rounded_current_point + vec2(1,0),line_config.mode);
        mode_buffer_set_2d(rounded_current_point - vec2(1,0),line_config.mode);
        mode_buffer_set_2d(rounded_current_point + vec2(0,1),line_config.mode);
        mode_buffer_set_2d(rounded_current_point - vec2(0,1),line_config.mode);
        current_point += delta_point;
    }
}

 
@compute
@workgroup_size(8,8)
fn load_to_texture(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let location = vec2(global_id.x,global_id.y);
    var diffuse = vec4(0.0,0.0,0.0,0.0);
    switch copy_mode {
        case 0u: {
            diffuse = diffuse_buffer[pad_index_2d(location)];
        }
        case 1u: {
            let vel = velocity_buffer[index_2d(location)];
            let angle = atan2(vel.y,vel.x);
            let mag = sqrt(dot(vel,vel));
            diffuse = vec4(
                max((3.0/(2.0*pi)) * abs(angle) - 0.5,0.0) * mag,
                max(1.0 - (3.0/(2.0*pi)) * abs(angle + (pi/3.0)),0.0) * mag,
                max(1.0 - (3.0/(2.0*pi)) * abs(angle - (pi/3.0)),0.0) * mag,
                1.0
            );
        }
        case 2u: {
            let heat = heat_buffer[index_2d(location)];
            diffuse = vec4(heat,heat,heat,1.0);
        }
        default: {}
    }
    textureStore(diffuse_texture_storage,location,diffuse);
}