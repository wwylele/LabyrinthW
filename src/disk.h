
#include "Helpers.h"

Program DiskProgram() {

    Program disk_program;
    const GLchar* disk_vertex_shader = R"(
#version 150 core

uniform mat4 obj_tran;
uniform mat4 camera_tran;

out vec2 circle_coord;

void main()
{
    int in_face_id = gl_VertexID;
    vec2 coord;
    if (in_face_id == 0 || in_face_id == 3) {
        coord = vec2(-1.0, -1.0);
    } else if (in_face_id == 1) {
       coord = vec2(1.0, -1.0);
    } else if (in_face_id == 2 || in_face_id == 4) {
        coord = vec2(1.0, 1.0);
    } else if (in_face_id == 5) {
        coord = vec2(-1.0, 1.0);
    }

    circle_coord = coord;
    gl_Position = camera_tran * obj_tran * vec4(coord, 0.0, 1.0);
}
)";
    const GLchar* disk_fragment_shader = R"(
#version 150 core

in vec2 circle_coord;
out vec4 outColor;

uniform sampler2D disk_texture;

void main()
{
    if (dot(circle_coord, circle_coord) >= 1.0)
        discard;
    outColor = vec4(texture(disk_texture, (circle_coord + vec2(1.0, 1.0)) * 0.5).xyz * 0.5, 1.0);
}
)";

    disk_program.init(disk_vertex_shader,disk_fragment_shader,"outColor");

    return disk_program;
}
