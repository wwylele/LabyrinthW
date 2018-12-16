
#include "Helpers.h"

Program SphereProgram() {

    Program sphere_program;
    const GLchar* sphere_vertex_shader = R"(
#version 150 core

uniform vec3 obj_color;
uniform vec3 camera_pos;
uniform mat4 obj_tran;
uniform mat4 inv_obj;
uniform mat4 camera_tran;

out vec3 vert_color;
out vec3 e; // ray origin in object space
out vec3 d; // ray direction in objct space

const vec3 ijk[3] = vec3[3](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));

void main()
{
    int face = gl_VertexID / 6;
    int in_face_id = gl_VertexID % 6;
    int axis = face / 2;
    float negate = 2.0 * (0.5 - float(face % 2));

    vec3 perm[3] = vec3[3](
        ijk[axis], ijk[(axis + 1) % 3], ijk[(axis + 2) % 3]
    );

    vec3 position;
    if (in_face_id == 0 || in_face_id == 3) {
        position = (perm[0] + perm[1] + perm[2]) * negate;
    } else if (in_face_id == 1) {
        position = perm[0] * negate - perm[1] + perm[2];
    } else if (in_face_id == 2 || in_face_id == 4) {
        position = (perm[0] - perm[1] - perm[2]) * negate;
    } else if (in_face_id == 5) {
        position = perm[0] * negate + perm[1] - perm[2];
    }

    e = (inv_obj * vec4(camera_pos, 1.0)).xyz;
    d = position - e;

    vert_color = obj_color;

    vec4 world_pos = obj_tran * vec4(position, 1.0);
    vec4 proj_pos = camera_tran * world_pos;

    gl_Position = proj_pos;
}
)";
    const GLchar* sphere_fragment_shader = R"(
#version 150 core

in vec3 vert_color;
in vec3 e;
in vec3 d;

out vec4 outColor;

uniform vec3 camera_pos;
uniform mat4 obj_tran;
uniform mat4 inv_obj;
uniform mat4 camera_tran;
uniform vec3 light_source;
uniform sampler2D shadow;
uniform mat4 shadow_tran;

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
    float a = dot(d, d);
    float b = 2 * dot(d, e);
    float c = dot(e, e) - 1;
    float det = b * b - 4 * a * c;
    if (det < 0.0)
        discard;

    float t = (-b - sqrt(det)) / (2.0 * a);
    if (t < 0.0)
        discard;

    vec3 p = e + d * t;

    vec4 p_world = obj_tran * vec4(p, 1.0);
    vec4 n_world = transpose(inv_obj) * vec4(p, 0.0);
    vec4 p_proj = camera_tran * p_world;
    vec4 p_shadow = shadow_tran * p_world;
    p_shadow /= p_shadow.w;
    p_shadow *= 0.5;
    p_shadow += vec4(0.5, 0.5, 0.5, 0.5);
    gl_FragDepth = p_proj.z / p_proj.w * 0.5 + 0.5;

    vec3 v = normalize(camera_pos - p_world.xyz);
    vec3 l = normalize(light_source - p_world.xyz);
    vec3 n = normalize(n_world.xyz);
    vec3 h = normalize(l + v);
    vec3 diffuse = vert_color * 0.8 * max(0.0, dot(n, l));
    vec3 specular = vert_color * pow(max(0.0, dot(n, h)), 50);
    vec3 amb = vert_color * 0.2;

    float shadow_coef = 1.0;
    for (int i = 0; i < 9; ++i) {
        if (texture(shadow, p_shadow.xy + shadow_dev[i] / 1024.0).x < (p_shadow.z - 0.0004)) {
            shadow_coef -= 1.0 / 9;
        }
    }

    outColor = vec4(clamp((diffuse + specular) * shadow_coef + amb, 0.0, 1.0), 1.0);
}
)";

    sphere_program.init(sphere_vertex_shader,sphere_fragment_shader,"outColor");

    return sphere_program;
}
