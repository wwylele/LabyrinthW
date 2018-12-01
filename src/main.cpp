// This example is heavily based on the tutorial at https://open.gl

// OpenGL Helpers to reduce the clutter
#include "Helpers.h"

// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>

// Linear Algebra Library
#include <Eigen/Core>
#include <Eigen/Dense>

// Timer
#include <chrono>

#include <fstream>
#include <iostream>
#include <tuple>
#include <thread>
#include <array>

struct Object {
    Eigen::Projective3f linear = Eigen::Projective3f::Identity();
    Eigen::Projective3f translate = Eigen::Projective3f::Identity();
    Eigen::Vector3f color = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
};

std::vector<Object> objects;

const std::size_t INVALID = ~0;
std::size_t current = INVALID;


const std::array<Eigen::Vector3f, 10> color_table{{
    {0.8f, 0.2f, 0.2f},
    {0.8f, 0.5f, 0.2f},
    {0.8f, 0.8f, 0.2f},
    {0.2f, 0.8f, 0.2f},
    {0.2f, 0.8f, 0.8f},
    {0.5f, 0.5f, 0.5f},
    {0.5f, 0.1f, 0.1f},
    {0.1f, 0.5f, 0.1f},
    {0.1f, 0.1f, 0.5f},
    {1.0f, 1.0f, 1.0f},
}};

const float pi = 3.14159265359;
float camera_r = 5.0f;
float camera_theta = 0.0f;
float camera_phi = pi / 2;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS)
        return;
    switch (key)
    {
    case GLFW_KEY_1: {
        Object object;
        objects.push_back(object);
        break;
    }
    case GLFW_KEY_F1:
    case GLFW_KEY_F2:
    case GLFW_KEY_F3:
    case GLFW_KEY_F4:
    case GLFW_KEY_F5:
    case GLFW_KEY_F6:
    case GLFW_KEY_F7:
    case GLFW_KEY_F8:
    case GLFW_KEY_F9:
    case GLFW_KEY_F10:
        if (current != INVALID) {
            int index = key - GLFW_KEY_F1;
            objects[current].color = color_table[index];
        }
        break;
    case GLFW_KEY_DELETE:
        if (current != INVALID) {
            objects.erase(objects.begin() + current);
            current = INVALID;
        }
        break;
    }
}


float y_over_x = 1.0;

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    y_over_x = (float)height / width;
    glViewport(0, 0, width, height);
}


Eigen::Projective3f CameraTransform(const Eigen::Vector3f& e) {
    Eigen::Vector3f w = -e.normalized();
    Eigen::Vector3f t = Eigen::Vector3f(0, 1.0f, 0);
    Eigen::Vector3f u = t.cross(w).normalized();
    Eigen::Vector3f v = w.cross(u);
    Eigen::Projective3f tr;
    tr.matrix().col(0) = Eigen::Vector4f(u(0), u(1), u(2), 0.0f);
    tr.matrix().col(1) = Eigen::Vector4f(v(0), v(1), v(2), 0.0f);
    tr.matrix().col(2) = Eigen::Vector4f(w(0), w(1), w(2), 0.0f);
    tr.matrix().col(3) = Eigen::Vector4f(e(0), e(1), e(2), 1.0f);
    return tr.inverse();
}

Eigen::Projective3f OrthProjection(float mx, float my, float zn, float zf) {
    Eigen::Projective3f tr;
    tr.matrix().col(0) = Eigen::Vector4f(1 / mx, 0, 0, 0);
    tr.matrix().col(1) = Eigen::Vector4f(0, 1 / my, 0, 0);
    tr.matrix().col(2) = Eigen::Vector4f(0, 0, 2 / (zf - zn), 0);
    tr.matrix().col(3) = Eigen::Vector4f(0, 0, -(zn + zf) / (zf - zn), 1);
    return tr;
}

Eigen::Projective3f Perspective(float mx, float my, float zn, float zf) {
    Eigen::Projective3f tr;
    tr.matrix().col(0) = Eigen::Vector4f(zn, 0, 0, 0);
    tr.matrix().col(1) = Eigen::Vector4f(0, zn, 0, 0);
    tr.matrix().col(2) = Eigen::Vector4f(0, 0, zn + zf, 1);
    tr.matrix().col(3) = Eigen::Vector4f(0, 0, -zf * zn, 0);
    return OrthProjection(mx, my, zn, zf) * tr;
}

Eigen::Vector3f GetCameraPos() {
    Eigen::Vector3f camera {
        std::sin(camera_phi) * std::cos(camera_theta),
        std::cos(camera_phi),
        std::sin(camera_phi) * std::sin(camera_theta),
    };
    camera *= camera_r;
    return camera;
}

Eigen::Projective3f GetCameraMatrix() {
    Eigen::Vector3f camera = GetCameraPos();
    auto camera_tran = CameraTransform(camera);
    auto proj = Perspective(0.5f, 0.5f * y_over_x, 1.0f, 1000.0f);
    return proj * camera_tran;
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    // Get the position of the mouse in the window
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Get the size of the window
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // Convert screen position to world coordinates
    double x = ((xpos/double(width))*2)-1;
    double y = (((height-1-ypos)/double(height))*2)-1; // NOTE: y axis is flipped in glfw

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        Eigen::Vector4f e{(float)x, (float)y, -1.0f, 1.0f};
        Eigen::Vector4f d{0.0f, 0.0f, 1.0f, 0.0f};
        Eigen::Vector4f e1d = e + d;
        Eigen::Projective3f camera_tran = GetCameraMatrix();
        Eigen::Projective3f rev = camera_tran.inverse();
        e = rev * e;
        e1d = rev * e1d;

        bool picked = false;
        std::size_t pick_index;
        float mt;
        for (std::size_t i = 0; i < objects.size(); ++i) {
            auto& object = objects[i];
            Eigen::Projective3f local_rev = (object.translate * object.linear).inverse();
            Eigen::Vector4f le = local_rev * e;
            Eigen::Vector4f le1d = local_rev * e1d;
            Eigen::Vector3f re = Eigen::Vector3f(le(0), le(1), le(2)) / le(3);
            Eigen::Vector3f re1d = Eigen::Vector3f(le1d(0), le1d(1), le1d(2)) / le1d(3);
            Eigen::Vector3f rd = re1d - re;

            // Sphere approximation
            float a = rd.dot(rd);
            float b = 2 * rd.dot(re);
            float c = re.dot(re) - 1;
            float det = b * b - 4 * a * c;
            if (det < 0.0)
                continue; //  Don't botther checking triangles if the sphere test fails

            float t = (-b - std::sqrt(det)) / (2 * a);
            if (t < 0.0)
                continue;

            if (!picked || t < mt) {
                picked = true;
                pick_index = i;
                mt = t;
            }
        }

        if (picked) {
            current = pick_index;
        } else {
            current = INVALID;
        }
    }
}

int main(void)
{
    GLFWwindow* window;

    // Initialize the library
    if (!glfwInit())
        return -1;

    // Activate supersampling
    glfwWindowHint(GLFW_SAMPLES, 8);

    // Ensure that we get at least a 3.2 context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

    // On apple we have to load a core profile with forward compatibility
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(500, 500, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    #ifndef __APPLE__
      glewExperimental = true;
      GLenum err = glewInit();
      if(GLEW_OK != err)
      {
        /* Problem: glewInit failed, something is seriously wrong. */
       fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      }
      glGetError(); // pull and savely ignonre unhandled errors like GL_INVALID_ENUM
      fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    #endif

    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    // Initialize the OpenGL Program
    // A program controls the OpenGL pipeline and it must contains
    // at least a vertex shader and a fragment shader to be valid
    Program program;
    const GLchar* vertex_shader = R"(
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
    const GLchar* fragment_shader = R"(
#version 150 core

in vec3 vert_color;
in vec3 e;
in vec3 d;

out vec4 outColor;

uniform vec3 camera_pos;
uniform mat4 obj_tran;
uniform mat4 inv_obj;
uniform mat4 camera_tran;

const vec3 light_source = vec3(10.0, 10.0, 10.0);

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
    gl_FragDepth = p_proj.z / p_proj.w * 0.5 + 0.5;

    vec3 v = normalize(camera_pos - p_world.xyz);
    vec3 l = normalize(light_source - p_world.xyz);
    vec3 n = normalize(n_world.xyz);
    vec3 h = normalize(l + v);
    vec3 diffuse = vert_color * 0.8 * max(0.0, dot(n, l));
    vec3 specular = vert_color * pow(max(0.0, dot(n, h)), 50);
    outColor = vec4(clamp(diffuse + specular + vert_color * 0.05, 0.0, 1.0), 1.0);
}
)";

    program.init(vertex_shader,fragment_shader,"outColor");
    program.bind();

    VertexArrayObject vao;
    vao.init();
    vao.bind();

    // Save the current time --- it will be used to dynamically change the triangle color
    auto t_start = std::chrono::high_resolution_clock::now();
    auto key_t_prev = t_start;

    // Register the keyboard callback
    glfwSetKeyCallback(window, key_callback);

    // Register the mouse callback
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Update viewport
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glEnable(GL_DEPTH_TEST);

    int uniform_vert_color = program.uniform("obj_color");
    int uniform_obj_tran = program.uniform("obj_tran");
    int uniform_camera_tran = program.uniform("camera_tran");
    int uniform_inv_obj = program.uniform("inv_obj");
    int uniform_camera_pos = program.uniform("camera_pos");

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        // Bind your program
        program.bind();

        // Set the uniform value depending on the time difference
        auto t_now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();

        const auto key_sample_period = std::chrono::duration_cast
            <std::chrono::high_resolution_clock::duration>
            (std::chrono::milliseconds(10));


        // limit frame rate;
        std::this_thread::sleep_until(key_t_prev + key_sample_period);

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            camera_theta += 0.03f;
        }

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            camera_theta -= 0.03f;
        }

        const float lock_prevent = 0.1f;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            camera_phi += 0.03f;
            camera_phi = std::min(camera_phi, pi - lock_prevent);
        }

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            camera_phi -= 0.03f;
            camera_phi = std::max(camera_phi, lock_prevent);
        }

        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            camera_r -= 0.03f;
            camera_r = std::max(camera_r, 0.1f);
        }

        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            camera_r += 0.03f;
        }


        if (current != INVALID) {
            const float tu = 0.05f;

            Eigen::Vector3f forward = -GetCameraPos().normalized();
            Eigen::Vector3f left = forward.cross(Eigen::Vector3f(0.0f, 1.0f, 0.0f)).normalized();
            Eigen::Vector3f up = left.cross(forward);

            /// Object translation
            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
                objects[current].translate = Eigen::Translation3f(tu * left) * objects[current].translate;
            }
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
                objects[current].translate = Eigen::Translation3f(-tu * left) * objects[current].translate;
            }
            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
                objects[current].translate = Eigen::Translation3f(tu * up) * objects[current].translate;
            }
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
                objects[current].translate = Eigen::Translation3f(-tu * up) * objects[current].translate;
            }
            if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
                objects[current].translate = Eigen::Translation3f(tu * forward) * objects[current].translate;
            }
            if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
                objects[current].translate = Eigen::Translation3f(-tu * forward) * objects[current].translate;
            }

            /// Object scaling
            if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
                objects[current].linear = Eigen::Scaling(1.1f, 1.1f, 1.1f) * objects[current].linear;
            }
            if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) {
                objects[current].linear = Eigen::Scaling(0.9f, 0.9f, 0.9f) * objects[current].linear;
            }

            /// Object rotation
            if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
                objects[current].linear = Eigen::AngleAxisf(0.05, left)
                    * objects[current].linear;
            }
            if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS) {
                objects[current].linear = Eigen::AngleAxisf(-0.05, left)
                    * objects[current].linear;
            }
            if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
                objects[current].linear = Eigen::AngleAxisf(0.05, up)
                    * objects[current].linear;
            }
            if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
                objects[current].linear = Eigen::AngleAxisf(-0.05, up)
                    * objects[current].linear;
            }
            if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) {
                objects[current].linear = Eigen::AngleAxisf(0.05, forward)
                    * objects[current].linear;
            }
            if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS) {
                objects[current].linear = Eigen::AngleAxisf(-0.05, forward)
                    * objects[current].linear;
            }
        }


        key_t_prev += key_sample_period;

        // Clear the framebuffer
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClearDepth(1.0f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        Eigen::Projective3f camera_tran = GetCameraMatrix();
        Eigen::Vector3f camera_pos = GetCameraPos();

        glUniformMatrix4fv(uniform_camera_tran, 1, GL_FALSE, camera_tran.data());
        glUniform3fv(uniform_camera_pos, 1, camera_pos.data());

        for (std::size_t i = 0; i < objects.size(); ++i) {
            auto& object = objects[i];

            Eigen::Projective3f obj_tran = object.translate * object.linear;
            glUniformMatrix4fv(uniform_obj_tran, 1, GL_FALSE, obj_tran.data());

            Eigen::Projective3f inv = obj_tran.inverse();
            glUniformMatrix4fv(uniform_inv_obj, 1, GL_FALSE, inv.data());

            glUniform3fv(uniform_vert_color, 1, object.color.data());
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Deallocate opengl memory
    program.free();
    vao.free();

    // Deallocate glfw internals
    //glfwTerminate();
    return 0;
}
