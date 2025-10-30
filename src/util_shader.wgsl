
@group(0) @binding(0) var<storage,read_write> buffer: array<f32>;

@group(1) @binding(0) var<storage,read_write> diffuse_buffer: array<f32>;

@group(1) @binding(10) var<uniform> half_sum_size: HalfSumSize;

@group(2) @binding(0) var<uniform> size: Size;

struct HalfSumSize {
    block_size: u32,
    chunk_size: u32
}

struct Size {
    backing_width: u32,
    backing_height: u32
}

fn index_2d(point: vec2<u32>) -> u32 {
    return point.x + size.backing_width * point.y;
}

//make sure to keep in sync with sim_shader.wgsl
fn total_diffuse(diffuse: vec4<f32>) -> f32 {
    return diffuse.x + diffuse.y + diffuse.z;
} 

/* intended use (_ represents unused value):
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    half_sum() [block_size = 1, chunk_size = 1]
    3, _, 7, _, 11, _, 15, _, 19, _
    half_sum() [block_size = 1, chunk_size = 2]
    10, _, _, _, 26, _, _, _, 19, _
    half_sum() [block_size = 1, chunk_size = 4]
    36, _, _, _, _, _, _, _, 19, _
    half_sum() [block_size = 1, chunk_size = 8]
    55, _, _, _, _, _, _, _, _, _

    then read out index 0

    use block size if summing vecs or structs (2 for vec2, 3 for vec3, etc.)
*/
@compute
@workgroup_size(64)
fn half_sum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x * half_sum_size.block_size * (2 * half_sum_size.chunk_size);
    let shifted_index = index + half_sum_size.block_size * half_sum_size.chunk_size;
    for (var i = 0u; i < half_sum_size.block_size; i++) {
        let left_index = index + i;
        let right_index = shifted_index + i;
        if right_index < arrayLength(&buffer) {
            buffer[left_index] = buffer[left_index] + buffer[right_index];
        }
    }
}

@compute
@workgroup_size(64)
fn square_arr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x < arrayLength(&buffer) {
        let old = buffer[global_id.x];
        buffer[global_id.x] = old * old;
    }
}

@compute
@workgroup_size(8,8)
fn diffusion_scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= size.backing_width | global_id.y >= size.backing_height {
        return;
    }

    let block_sizing = arrayLength(&buffer) / (size.backing_width * size.backing_height); //jank

    let location = vec2(global_id.x,global_id.y);
    
    let diffuse = vec4(
        diffuse_buffer[4 * index_2d(location) + 0],
        diffuse_buffer[4 * index_2d(location) + 1],
        diffuse_buffer[4 * index_2d(location) + 2],
        diffuse_buffer[4 * index_2d(location) + 3]
    );
    let diffuse_total = total_diffuse(diffuse);
    for (var i = 0u; i < block_sizing; i++) {
        buffer[block_sizing * index_2d(location) + i] *= diffuse_total;
    }

}

