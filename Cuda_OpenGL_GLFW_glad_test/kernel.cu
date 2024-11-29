
#include <glad/glad.h>
#include "shaders.hpp"

#include "logic.cpp"
#include "configuration.h"

#include "thrust/sort.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include <glm/glm.hpp>

#include <stdio.h>

// ####################################################################################################################################################################################
    // SHADERS

const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in float aRadius;
    uniform mat4 uProjection;

    void main() {
        gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
        gl_PointSize = aRadius;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec4 uColor;

    void main() {
        FragColor = uColor;
    }
)";


// ####################################################################################################################################################################################


#define MY_ERROR(source) perror(source), fprintf("file: %s, line: %d\n", __LINE__, __FILE__), exit(-1)
#define ERROR_CUDA(status) do { \
            if(status != cudaSuccess) \
            { \
                fprintf(stderr, "error: %s\n",  cudaGetErrorString(status)); \
                fprintf(stderr, "file: %s, line: %d\n", __FILE__, __LINE__); \
                exit(-1); \
            } \
        } while(0) \

__global__ void updateParticlesKernel(particles_gpu p, float wwidth, float wheight, float cell_size, float grid_width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // particle index
    if (i >= p.size)
        return;

    p.x[i] += p.vx[i];
    p.y[i] += p.vy[i];
    p.vy[i] -= p.g * p.m[i];

    // check window bounds
    if (p.x[i] >= wwidth)
        p.vx[i] = -abs(p.vx[i]);
    if (p.x[i] <= 0)
        p.vx[i] = abs(p.vx[i]);

    if (p.y[i] >= wheight)
        p.vy[i] = -abs(p.vy[i]);
    if (p.y[i] <= 0)
        p.vy[i] = abs(p.vy[i]);

    int cell_x = p.x[i] / cell_size;
    int cell_y = p.y[i] / cell_size;
    p.cell[i] = cell_x * grid_width + cell_y;
}

__global__ void particlesCollisionKernel(particles_gpu p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // particle index
    if (i >= p.size)
        return;

    int cell = p.cell[i];
    int j = i + 1;
    int k = i;
    while (k < p.size && p.cell[k] <= cell + 1)
        ++k;


    for (; j < k; ++j) {
        float dist_x = p.x[i] - p.x[j];
        float dist_y = p.y[i] - p.y[j];
        float dist = sqrt(dist_x * dist_x + dist_y * dist_y);
        if (dist <= p.radius[i] + p.radius[j]) {
            float norm_x = dist_x / dist;
            float norm_y = dist_y / dist;

            float rel_vx = p.vx[i] - p.vx[j];
            float rel_vy = p.vy[i] - p.vy[j];

            float rel_v = rel_vx * norm_x + rel_vy * norm_y;
            if (rel_v > 0) // are moving in opposite direction
                continue;
            float impulse = (2.0f * rel_v) / (p.m[i] * p.m[j]);

            p.vx[i] -= impulse * p.m[i] * norm_x;
            p.vx[i] -= impulse * p.m[i] * norm_y;

            p.vx[j] -= impulse * p.m[j] * norm_x;
            p.vx[j] -= impulse * p.m[j] * norm_y;
        }
    }
}


#include <GLFW/glfw3.h> // Will drag system OpenGL headers

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Main code
int main(int, char**)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.3 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // ####################################################################################################################################################################################
    // INITIALIZE AND CONFIGURE

    ERROR_CUDA(cudaSetDevice(0));

    // load config
    configuration config;
    config.load_configuration();

    particles p;
    p.initialize(config);

    int wwidth = config.starting_wwidth;
    int wheigth = config.starting_wheigth;

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(wwidth, wheigth,
        "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD!" << std::endl;
        return -1;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");
#endif
    ImGui_ImplOpenGL3_Init(glsl_version);

    //std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    //std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

    // ####################################################################################################################################################################################
    // SHADERS

    GLfloat vertices[] =
    {
        -0.5f, -0.5f * float(sqrt(3)) / 3, 0.0f, // Lower left corner
        0.5f, -0.5f * float(sqrt(3)) / 3, 0.0f, // Lower right corner
        0.0f, 0.5f * float(sqrt(3)) * 2 / 3, 0.0f // Upper corner
    };

    // Compile shaders
    GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);

    // Link shaders into a program
    GLuint particleShaderProgram = linkProgram(vertexShader, fragmentShader);

    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // ####################################################################################################################################################################################
    // DRAWING???

    GLuint vao, vbo[3];
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Position buffer
    glGenBuffers(1, &vbo[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, p.gpu.size * sizeof(glm::vec2), p.h_pos.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)0);
    glEnableVertexAttribArray(0);

    // Radius buffer
    glGenBuffers(1, &vbo[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, p.gpu.size * sizeof(float), p.h_radius.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // Color buffer
    glGenBuffers(1, &vbo[2]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, p.gpu.size * sizeof(glm::vec4), p.color.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
    glEnableVertexAttribArray(2);

    // ####################################################################################################################################################################################























    // Our state
    ImVec4 clear_color = ImVec4(0.f, 0.f, 0.f, 1.00f);

    // Main loop
#ifdef __EMSCRIPTEN__
    // For an Emscripten build we are disabling file-system access, so let's not attempt to do a fopen() of the imgui.ini file.
    // You may manually call LoadIniSettingsFromMemory() to load settings from your own storage.
    io.IniFilename = nullptr;
    EMSCRIPTEN_MAINLOOP_BEGIN
#else
    while (!glfwWindowShouldClose(window))
#endif
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        glfwGetWindowSize(window, &wwidth, &wheigth);

        int cell_size = 50;
        int grid_width = (int)ceil(wwidth / cell_size);

        dim3 blocks = dim3(p.gpu.size / 16 + 1);
        dim3 threads = dim3(16);
        updateParticlesKernel<<<blocks, threads>>>(p.gpu, wwidth, wheigth, cell_size, grid_width);
        ERROR_CUDA(cudaGetLastError());
        ERROR_CUDA(cudaDeviceSynchronize());

        //thrust::sort_by_key(
        //    p.d_cell.begin(), p.d_cell.end(), // Key vector
        //    thrust::make_zip_iterator(
        //        thrust::make_tuple(
        //            p.d_x.begin(),
        //            p.d_y.begin(),
        //            p.d_vx.begin(),
        //            p.d_vy.begin(),
        //            p.d_m.begin(),
        //            p.d_radius.begin()
        //        )
        //    ) // Values as a zip iterator
        //);

        //ERROR_CUDA(cudaGetLastError());
        //ERROR_CUDA(cudaDeviceSynchronize());

        /*particlesCollisionKernel<<<blocks, threads>>>(p);
        ERROR_CUDA(cudaGetLastError());
        ERROR_CUDA(cudaDeviceSynchronize());*/

        p.copy_results_back();

        {
            ImGui::Begin("Dynamic settings");
            ImGui::Text("Only background color is changeable for now.");

            ImGui::ColorEdit3("Change background", (float*)&clear_color); // Edit 3 floats representing a color

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glUseProgram(particleShaderProgram);
        glBindVertexArray(vao);
        // Update buffers if necessary
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferSubData(GL_ARRAY_BUFFER, 0, p.gpu.size * sizeof(glm::vec2), p.h_pos.data());

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glBufferSubData(GL_ARRAY_BUFFER, 0, p.gpu.size * sizeof(float), p.h_radius.data());

        glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
        glBufferSubData(GL_ARRAY_BUFFER, 0, p.gpu.size * sizeof(glm::vec4), p.color.data());

        // Draw particles
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, p.gpu.size);

        glfwSwapBuffers(window);
    }
#ifdef __EMSCRIPTEN__
    EMSCRIPTEN_MAINLOOP_END;
#endif

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    //glDeleteVertexArrays(1, &VAO);
    //glDeleteBuffers(1, &VBO);
    //glDeleteProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
