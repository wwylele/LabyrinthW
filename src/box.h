
#include "Helpers.h"

Program BoxProgram() {

    Program box_program;
    const GLchar* box_vertex_shader = R"(
#version 150 core

uniform mat4 obj_tran;
uniform mat4 camera_tran;
uniform float tex_seed;
uniform mat4 shadow_tran;

const vec3 ijk[3] = vec3[3](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));

out vec4 p_world;
out vec4 p_shadow_v;
out vec4 n_world;
out vec2 texcoord;

void main()
{
    int face = gl_VertexID / 6;
    int in_face_id = gl_VertexID % 6;
    int axis = face / 2;
    float negate = 2.0 * (0.5 - float(face % 2));

    vec3 perm[3] = vec3[3](
        ijk[axis], ijk[(axis + 1) % 3], ijk[(axis + 2) % 3]
    );

    float tx = tex_seed * tex_seed * face;
    float ty = tex_seed * face;
    float tdx = 1.0 + sin(tex_seed * tex_seed + face) * 0.25;
    float tdy = 1.0 + cos(1.5 * tex_seed - face) * 0.25;

    vec3 position;
    if (in_face_id == 0 || in_face_id == 3) {
        position = (perm[0] + perm[1] + perm[2]) * negate;
        texcoord = vec2(tx, ty);
    } else if (in_face_id == 1) {
        position = perm[0] * negate - perm[1] + perm[2];
        texcoord = vec2(tx + tdx, ty);
    } else if (in_face_id == 2 || in_face_id == 4) {
        position = (perm[0] - perm[1] - perm[2]) * negate;
        texcoord = vec2(tx + tdx, ty + tdy);
    } else if (in_face_id == 5) {
        position = perm[0] * negate + perm[1] - perm[2];
        texcoord = vec2(tx, ty + tdy);
    }

    vec4 world_pos = obj_tran * vec4(position, 1.0);

    p_world = world_pos;
    n_world = obj_tran * vec4(perm[0] * negate, 0.0);

    vec4 proj_pos = camera_tran * world_pos;

    gl_Position = proj_pos;
    p_shadow_v = shadow_tran * world_pos;
}
)";
    const GLchar* box_fragment_shader = R"(
#version 150 core

in vec4 p_world;
in vec4 n_world;
in vec2 texcoord;
in vec4 p_shadow_v;

out vec4 outColor;

uniform sampler2D box_texture;
uniform sampler2D shadow;
uniform vec3 camera_pos;
uniform vec3 light_source;

vec2 shadow_dev[9] = vec2[](
  vec2(0.0, 0.0),
  vec2(-1.0, 0.0),
  vec2(1.0, 0.0),
  vec2(0.0, 1.0),
  vec2(0.0, -1.0),
  vec2(-1.0, 1.0),
  vec2(1.0, 1.0),
  vec2(-1.0, -1.0),
  vec2(1.0, -1.0)
);

void main()
{
    vec4 p_shadow = p_shadow_v;
    p_shadow /= p_shadow.w;
    p_shadow *= 0.5;
    p_shadow += vec4(0.5, 0.5, 0.5, 0.5);

    vec3 v = normalize(camera_pos - p_world.xyz);
    vec3 l = normalize(light_source - p_world.xyz);
    vec3 n = normalize(n_world.xyz);
    vec3 h = normalize(l + v);
    vec3 vert_color = texture(box_texture, texcoord).xyz;
    vec3 diffuse = vert_color * max(0.0, dot(n, l)) * 0.5;
    vec3 specular = vec3(0.2, 0.2, 0.2) * pow(max(0.0, dot(n, h)), 2);
    vec3 amb = vert_color * 0.5;

    float shadow_coef = 1.0;
    for (int i = 0; i < 9; ++i) {
        if (texture(shadow, p_shadow.xy + shadow_dev[i] / 1024.0).x < (p_shadow.z - 0.0001)) {
            shadow_coef -= 1.0 / 9;
        }
    }

    outColor = vec4(clamp((diffuse + specular) * shadow_coef + amb, 0.0, 1.0), 1.0);
}
)";

    box_program.init(box_vertex_shader,box_fragment_shader,"outColor");

    return box_program;
}
