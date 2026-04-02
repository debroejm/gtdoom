#version 460
#extension GL_EXT_nonuniform_qualifier : require

#ifdef WIREFRAME
    #extension GL_EXT_fragment_shader_barycentric : require

    float WireFrame(in float Thickness)
    {
        const vec3 BaryCoord = gl_BaryCoordEXT;

        const vec3 dBaryCoordX = dFdxFine(BaryCoord);
        const vec3 dBaryCoordY = dFdyFine(BaryCoord);
        const vec3 dBaryCoord  = sqrt(dBaryCoordX*dBaryCoordX + dBaryCoordY*dBaryCoordY);

        const vec3 dThickness = dBaryCoord * Thickness;

        const vec3 Remap = smoothstep(vec3(0.0), dThickness, BaryCoord);
        const float ClosestEdge = min(min(Remap.x, Remap.y), Remap.z);

        return 1.0 - ClosestEdge;
    }
#endif

layout(location = 0) in vec2 tex_coords;
layout(location = 1) in float light;
layout(location = 2) flat in uint tex_id;
layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler s;
layout(set = 1, binding = 1) uniform texture2D tex[];

void main() {
    vec4 color = texture(sampler2D(tex[nonuniformEXT(tex_id)], s), tex_coords);
    #ifdef WIREFRAME
        if( WireFrame(1.5) >= 0.8 ) {
            f_color = vec4(1.0, 0.0, 0.5, 1.0);
        } else {
            f_color = vec4(color.xyz * light, color.w);
        }
    #else
        f_color = vec4(color.xyz * light, color.w);
    #endif
}