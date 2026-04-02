#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in float light;
layout(location = 3) in uint tag;
layout(location = 4) in uint tex_id;

layout(location = 0) out vec2 tex_coords;
layout(location = 1) out float out_light;
layout(location = 2) flat out uint out_tex_id;

layout(set = 0, binding = 0) uniform VP_Data {
    mat4 vp;
} vp;

void main() {
    gl_Position = vp.vp * vec4(position, 1.0);
    tex_coords = uv;
    out_light = light;
    out_tex_id = tex_id;
}