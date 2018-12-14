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
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "sphere.h"
#include "box.h"
#include "disk.h"
#include "cylinder.h"

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
float camera_r = 50.0f;
float camera_theta = pi / 2;
float camera_phi = pi / 2;



Eigen::Vector2d tilt{0.0, 0.0};

Eigen::Vector2d reset_ball_p{14, 14}, reset_ball_v{0.0, 0.0};
Eigen::Vector2d ball_p = reset_ball_p, ball_v = reset_ball_v;

constexpr double ball_r = 1.0;

bool falling = false;
std::size_t falling_hole;
double falling_distance;
double falling_angle;
double falling_av, falling_rv;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS)
        return;
    switch (key)
    {
    case GLFW_KEY_ENTER:
        falling = false;
        ball_p = reset_ball_p;
        ball_v = reset_ball_v;
        break;
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



constexpr float BoardXScale = 20.0f;
constexpr float BoardYScale = 20.0f;
constexpr float BoardZScale = 1.0f;

struct Box {
    float x_scale = 1.0f;
    float y_scale = 1.0f;
    float x_pos = 1.0f;
    float y_pos = 1.0f;
    float rotation = 0.0f;

    Box(float x_scale, float y_scale, float x_pos, float y_pos, float rotation):
        x_scale(x_scale), y_scale(y_scale), x_pos(x_pos), y_pos(y_pos), rotation(rotation) {}

    Eigen::Affine3f GetMatrix() const {
        Eigen::Affine3f tran = Eigen::Affine3f::Identity();
        tran = Eigen::Scaling(x_scale, y_scale, BoardZScale) * tran;
        tran = Eigen::AngleAxisf(rotation, Eigen::Vector3f(0.0, 0.0, 1.0f)) * tran;
        tran = Eigen::Translation3f(x_pos, y_pos, BoardZScale) * tran;
        return tran;
    }
};

std::vector<Box> boxes {
    Box{1.0f, 10.0f, 11.0f, 5.0f, 0.0f},
    Box{8.0f, 0.2f, 8.0f, -13.0f, 0.0f},
    Box{8.0f, 0.8f, 4.0f, -7.0f, 1.0f},
    Box{4.0f, 0.4f, -2.0f, -10.4f, 0.0f},
    Box{0.6f, 12.0f, -7.0f, -0.0f, 0.0f},

    Box{0.2f, 0.3f, -10.0f, -10.2f, 0.0f},
    Box{0.3f, 0.1f, -16.0f, -9.8f, 0.0f},
    Box{0.3f, 0.3f, -9.5f, -8.4f, 0.0f},
    Box{0.2f, 0.2f, -13.0f, -7.4f, 0.0f},
    Box{0.4f, 0.3f, -11.2f, -5.5f, 0.0f},
    Box{0.2f, 0.3f, -17.4f, -3.9f, 0.0f},
    Box{0.2f, 0.1f, -15.1f, -1.4f, 0.0f},
    Box{0.3f, 0.3f, -11.9f, 1.8f, 0.0f},
    Box{0.2f, 0.2f, -9.7f, 3.3f, 0.0f},
    Box{0.3f, 0.3f, -13.7f, 4.8f, 0.0f},
    Box{0.4f, 0.2f, -10.4f, 6.8f, 0.0f},
    Box{0.3f, 0.3f, -16.6f, 7.6f, 0.0f},
    Box{0.2f, 0.1f, -15.1f, 10.5f, 0.0f},
    Box{0.1f, 0.3f, -13.1f, 12.7f, 0.0f},
    Box{0.2f, 0.1f, -12.8f, 11.0f, 0.0f},

    Box{7.5f, 1.0f, 0.0f, 13.5f, 0.0f},

    Box{2.5f, 0.2f, 0.0f, 7.0f, 0.2f},
    Box{2.5f, 0.2f, 0.0f, -1.0f, 0.2f},
    Box{0.2f, 2.5f, 4.0f, 3.0f, 0.2f},
    Box{0.2f, 2.5f, -4.0f, 3.0f, 0.2f},
};

std::vector<Eigen::Vector2d> holes {
    {0.0, 3.0},
    {2.0, 2.0},
    {-2.0, 4.0},

    {11.0, 16.5},
    {17.0, -9.0},
    {15.0, -9.6},
    {13.0, -10.2},
    {11.0, -10.8},
    {5.0, -17},
    {-5.0, -17},
    {-5.0, -12},

    {-11.0, -1.0},
};

void AddEdges() {
    boxes.push_back(Box(1.0f, BoardYScale - 2.0f, BoardXScale - 1.0f, 0.0f, 0.0f));
    boxes.push_back(Box(1.0f, BoardYScale - 2.0f, -BoardXScale + 1.0f, 0.0f, 0.0f));
    boxes.push_back(Box(BoardXScale, 1.0f, 0.0f, BoardYScale - 1.0f, 0.0f));
    boxes.push_back(Box(BoardXScale,1.0f, 0.0f, -BoardYScale + 1.0f, 0.0f));
}

void move_callback(GLFWwindow* window, double xpos, double ypos) {
    // Get the size of the window
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    xpos -= width / 2.0;
    ypos -= height / 2.0;

    int unit = std::max(width, height) / 2.0;

    tilt = -Eigen::Vector2d{xpos, ypos} / unit * 0.3;
}

Eigen::Affine3f GetBallMatrix() {
    double delta = ball_r;
    if (falling) {
        if (falling_distance <= 0.0) {
            delta = -ball_r;
        } else {
            double a = ball_r - falling_distance;
            delta = std::sqrt(ball_r * ball_r - a * a);
        }
    }

    return Eigen::Translation3f((float)ball_p(0), (float)ball_p(1), (float)delta)
     * Eigen::Scaling((float)ball_r, (float)ball_r, (float)ball_r) ;
}

std::tuple<double, Eigen::Vector2d> PointToSegment(Eigen::Vector2d p, Eigen::Vector2d a, Eigen::Vector2d b) {
    Eigen::Vector2d ap = p - a;
    Eigen::Vector2d ab = b - a;
    float f = ap.dot(ab) / ab.dot(ab);
    if (f <= 0.0 || f >= 1.0)
        return std::make_tuple<double, Eigen::Vector2d>(10000.0f, {});
    return std::make_tuple<double, Eigen::Vector2d>((ap - ab * f).norm(), a + ab * f);
}

Eigen::Vector2d ProjectVector(Eigen::Vector2d v, Eigen::Vector2d a, Eigen::Vector2d b) {
    Eigen::Vector2d ab = b - a;
    return  v.dot(ab) / ab.dot(ab) * ab;
}

void StepFrame() {
    const double dt = 0.1;

    if (falling) {
        if (falling_distance <= 0.0) {
            falling_distance = 0.0;
            ball_p = holes[falling_hole];
            return;
        }

        falling_distance -= dt * falling_rv;
        falling_angle += dt * falling_av;
        ball_p = holes[falling_hole] + falling_distance *
            Eigen::Vector2d(std::cos(falling_angle), std::sin(falling_angle));
        return;
    }

    Eigen::Vector2d ball_a = tilt.normalized() * std::sin(tilt.norm());

    ball_v += ball_a * dt;

    Eigen::Vector2d new_p = ball_p + ball_v * dt;

    // filter v

    bool detected = false;
    for (Box& box : boxes) {
        bool collide = false;
        Eigen::Vector4d v[4]{
            {1.0, 1.0, 0.0, 1.0},
            {1.0, -1.0, 0.0, 1.0},
            {-1.0, -1.0, 0.0, 1.0},
            {-1.0, 1.0, 0.0, 1.0},
        };

        Eigen::Affine3d tran = box.GetMatrix().cast<double>();
        Eigen::Vector4d vv[4] {
            tran * v[0],
            tran * v[1],
            tran * v[2],
            tran * v[3],
        };

        Eigen::Vector2d a[4] {
            {vv[0](0), vv[0](1)},
            {vv[1](0), vv[1](1)},
            {vv[2](0), vv[2](1)},
            {vv[3](0), vv[3](1)},
        };

        Eigen::Vector2d b[4] {
            a[1],
            a[2],
            a[3],
            a[0],
        };

        for (int i = 0; i < 4; ++i) {
            float p, q;
            Eigen::Vector2d pg, qg;
            std::tie(p, pg) = PointToSegment(ball_p, a[i], b[i]);
            std::tie(q, qg) = PointToSegment(new_p, a[i], b[i]);
            if (p <= ball_r && q < p) {
                ball_p += (ball_r - p) / p * (ball_p - pg);

                if (detected) {
                    ball_v = {0.0, 0.0};
                    goto outer;
                }

                Eigen::Vector2d ab = b[i] - a[i];
                ball_v = ProjectVector(ball_v, a[i], b[i]);

                collide = true;
            }
        }

        for (int i = 0; i < 4; ++i) {
            Eigen::Vector2d ap = ball_p - a[i];
            Eigen::Vector2d aq = new_p - a[i];
            if (ap.norm() <= ball_r && aq.norm() < ap.norm()) {
                if (detected) {
                    ball_v = {0.0, 0.0};
                    goto outer;
                }

                Eigen::Vector2d ar = Eigen::Rotation2Dd(pi / 2) * ap;
                ball_v = ProjectVector(ball_v, {0.0, 0.0}, ar);
                collide = true;
            }
        }


        if (collide) {
            detected = true;
        }
    }

    outer:

    ball_p += ball_v * dt;

    for (std::size_t i = 0; i < holes.size(); ++i) {
        if ((holes[i] - ball_p).norm() > ball_r) continue;

        falling = true;
        falling_hole = i;
        falling_distance = ball_r;
        Eigen::Vector2d rp = ball_p - holes[i];
        falling_angle = std::atan2(rp(1), rp(0));
        Eigen::Vector2d rv = ProjectVector(ball_v, ball_p, holes[i]);
        falling_rv = rv.norm();
        Eigen::Vector2d av = ball_v - rv;
        falling_av = (rp(0) * av(1) - rp(1) * av(0)) / rp.dot(rp);

        break;
    }
}

Eigen::Affine3f GetBoardMatrix() {
    Eigen::Affine3f tran = Eigen::Affine3f::Identity();
    tran = Eigen::Scaling(BoardXScale, BoardYScale, BoardZScale) * tran;
    tran = Eigen::Translation3f(0, 0, -BoardZScale ) * tran;
    return tran;
}

Eigen::Affine3f GetGlobalMatrix() {
    Eigen::Vector2d axis = tilt.normalized();
    return Eigen::Translation3f(0, 0, -BoardZScale) *
    Eigen::AngleAxisd(tilt.norm(), Eigen::Vector3d(-axis(1), axis(0), 0.0)).cast<float>()
         * Eigen::Translation3f(0, 0, BoardZScale);
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
    window = glfwCreateWindow(1000, 1000, "Hello World", NULL, NULL);
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


    Program sphere_program = SphereProgram();
    Program cylinder_program = CylinderProgram();
    Program box_program = BoxProgram();
    Program disk_program = DiskProgram();

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

    glfwSetCursorPosCallback(window, move_callback);

    // Update viewport
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glEnable(GL_DEPTH_TEST);


    int x,y,n;
    unsigned char *data = stbi_load("wood.png", &x, &y, &n, 4);
    if (!data) {
        printf("failed to load wood texture file\n");
        return -1;
    }
    GLuint wood;
    glGenTextures(1, &wood);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, wood);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    stbi_image_free(data);

    data = stbi_load("check.png", &x, &y, &n, 4);
    if (!data) {
        printf("failed to load check texture file\n");
        return -1;
    }
    GLuint check;
    glGenTextures(1, &check);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, check);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    stbi_image_free(data);

    int sphere_vert_color = sphere_program.uniform("obj_color");
    int sphere_obj_tran = sphere_program.uniform("obj_tran");
    int sphere_camera_tran = sphere_program.uniform("camera_tran");
    int sphere_inv_obj = sphere_program.uniform("inv_obj");
    int sphere_camera_pos = sphere_program.uniform("camera_pos");

    int cylinder_vert_color = cylinder_program.uniform("obj_color");
    int cylinder_obj_tran = cylinder_program.uniform("obj_tran");
    int cylinder_camera_tran = cylinder_program.uniform("camera_tran");
    int cylinder_inv_obj = cylinder_program.uniform("inv_obj");
    int cylinder_camera_pos = cylinder_program.uniform("camera_pos");

    int box_obj_tran = box_program.uniform("obj_tran");
    int box_camera_tran = box_program.uniform("camera_tran");
    int box_camera_pos = box_program.uniform("camera_pos");
    int box_texture = box_program.uniform("box_texture");
    int box_tex_seed = box_program.uniform("tex_seed");

    int disk_obj_tran = disk_program.uniform("obj_tran");
    int disk_camera_tran = disk_program.uniform("camera_tran");
    int disk_texture = disk_program.uniform("disk_texture");

    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CW);
    glCullFace(GL_BACK);

    glEnable(GL_STENCIL_TEST);

    AddEdges();

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {

        // Set the uniform value depending on the time difference
        auto t_now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();

        const auto key_sample_period = std::chrono::duration_cast
            <std::chrono::high_resolution_clock::duration>
            (std::chrono::milliseconds(10));


        // limit frame rate;
        std::this_thread::sleep_until(key_t_prev + key_sample_period);

        StepFrame();

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
            camera_r -= 0.2f;
            camera_r = std::max(camera_r, 0.1f);
        }

        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            camera_r += 0.2f;
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
        glClearStencil(0);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

        Eigen::Projective3f camera_tran = GetCameraMatrix();
        Eigen::Vector3f camera_pos = GetCameraPos();

        disk_program.bind();
        glUniformMatrix4fv(disk_camera_tran, 1, GL_FALSE, camera_tran.data());

        glStencilFunc(GL_ALWAYS, 1, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        glStencilMask(0xFF);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glDepthMask(GL_FALSE);
        for (auto& hole : holes) {
            glUniformMatrix4fv(disk_obj_tran, 1, GL_FALSE,
                (GetGlobalMatrix() *
                Eigen::Translation3f((float)hole(0), (float)hole(1), 0.0f) *
                Eigen::Scaling((float)ball_r, (float)ball_r, 1.0f)).data()
            );
            glDrawArrays(GL_TRIANGLES, 0, 6); // holes
        }


        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);

        box_program.bind();

        glUniform1i(box_texture, 0);

        glUniformMatrix4fv(box_camera_tran, 1, GL_FALSE, camera_tran.data());
        glUniform3fv(box_camera_pos, 1, camera_pos.data());

        glUniform1f(box_tex_seed, 0.0);
        glUniformMatrix4fv(box_obj_tran, 1, GL_FALSE, (GetGlobalMatrix() * GetBoardMatrix()).data());
        glStencilFunc(GL_EQUAL, 0, 0xFF);
        glDrawArrays(GL_TRIANGLES, 0, 36); // main board

        glStencilFunc(GL_ALWAYS, 0, 0);

        cylinder_program.bind();

        glUniformMatrix4fv(cylinder_camera_tran, 1, GL_FALSE, camera_tran.data());
        glUniform3fv(cylinder_camera_pos, 1, camera_pos.data());
        Eigen::Vector3f color{0.5f, 0.3f, 0.0f};
        glUniform3fv(cylinder_vert_color, 1, color.data());
        for (auto& hole : holes) {
            Eigen::Affine3f obj_tran = (GetGlobalMatrix() *
                Eigen::Translation3f((float)hole(0), (float)hole(1), -BoardZScale) *
                Eigen::Scaling((float)ball_r, (float)ball_r, BoardZScale));

            Eigen::Affine3f inv_obj = obj_tran.inverse();
            glUniformMatrix4fv(cylinder_obj_tran, 1, GL_FALSE, obj_tran.data());
            glUniformMatrix4fv(cylinder_inv_obj, 1, GL_FALSE, inv_obj.data());
            glDrawArrays(GL_TRIANGLES, 0, 36); // holes wall
        }

        disk_program.bind();
        glUniformMatrix4fv(disk_camera_tran, 1, GL_FALSE, camera_tran.data());
        glUniform1i(disk_texture, 1); // the first one uses checker board
        for (auto& hole : holes) {
            glUniformMatrix4fv(disk_obj_tran, 1, GL_FALSE,
                (GetGlobalMatrix() *
                Eigen::Translation3f((float)hole(0), (float)hole(1), -2 * BoardZScale) *
                Eigen::Scaling((float)ball_r, (float)ball_r, 1.0f)).data()
            );
            glDrawArrays(GL_TRIANGLES, 0, 6); // hole bottom

            glUniform1i(disk_texture, 0); // others uses wood
        }

        box_program.bind();

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, wood);
        glUniform1i(box_texture, 0);

        for (std::size_t i = 0; i< boxes.size(); ++i) {
            glUniform1f(box_tex_seed, boxes[i].x_pos * boxes[i].y_pos);
            glUniformMatrix4fv(box_obj_tran, 1, GL_FALSE, (GetGlobalMatrix() * boxes[i].GetMatrix()).data());
            glDrawArrays(GL_TRIANGLES, 0, 36); // blocks
        }


        sphere_program.bind();
        glUniformMatrix4fv(sphere_camera_tran, 1, GL_FALSE, camera_tran.data());
        glUniform3fv(sphere_camera_pos, 1, camera_pos.data());

        Eigen::Projective3f ball_tran = GetGlobalMatrix() * GetBallMatrix();
        glUniformMatrix4fv(sphere_obj_tran, 1, GL_FALSE, ball_tran.data());

        Eigen::Projective3f inv = ball_tran.inverse();
        glUniformMatrix4fv(sphere_inv_obj, 1, GL_FALSE, inv.data());

        Eigen::Vector3f ball_color{0.8f, 0.8f, 0.8f};
        glUniform3fv(sphere_vert_color, 1, ball_color.data());
        glDrawArrays(GL_TRIANGLES, 0, 36);

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Deallocate opengl memory
    sphere_program.free();
    vao.free();

    // Deallocate glfw internals
    //glfwTerminate();
    return 0;
}
