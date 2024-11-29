
#define SHADER_TESTING
#define SHADER_TEST_1

#include <glad/glad.h>
#include "shaders.hpp"

#ifndef SHADER_TESTING

#include "logic.cpp"
#include <glm/glm.hpp>

#endif

#include "configuration.h"

#include "thrust/sort.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include <stdio.h>

// loading shaders from files
#include <string>
#include <fstream>
#include <streambuf>

// ####################################################################################################################################################################################
    // SHADERS

class shader_files {
private:
    std::string vertexShaderSource;
    std::string fragmentShaderSource;

    void loadFromFile(std::string& str, std::string& file)
    {
        std::ifstream t(file);

        t.seekg(0, std::ios::end);
        vertexShaderSource.reserve(t.tellg());
        t.seekg(0, std::ios::beg);

        str.assign((std::istreambuf_iterator<char>(t)),
            std::istreambuf_iterator<char>());
    }

public:
    const char* vertexShaderSourceC;
    const char* fragmentShaderSourceC;

    shader_files(std::string vertexFile, std::string fragmentFile)
    {
        loadFromFile(vertexShaderSource, vertexFile);
        vertexShaderSourceC = vertexShaderSource.c_str();

        loadFromFile(fragmentShaderSource, fragmentFile);
        fragmentShaderSourceC = fragmentShaderSource.c_str();
    }
};


#ifdef SHADER_TEST_1

#define VERTEX_SHADER_SOURCE "shaders/simple.vert"
#define FRAGMENT_SHADER_SOURCE "shaders/simple.frag"

#endif

// ####################################################################################################################################################################################

#define MY_ERROR(source) perror(source), fprintf("file: %s, line: %d\n", __LINE__, __FILE__), exit(-1)
#define ERROR_CUDA(status) do { \
            if(status != cudaSuccess) \
            { \
                fprintf(stderr, "error: %s\n",  cudaGetErrorString(status)); \
                fprintf(stderr, "file: %s, line: %d\n", __FILE__, __LINE__); \
                exit(-1); \
            } \
        } while(0)

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
    // ( ... )
    // ( some #ifdef and #else )

    // GL 3.3 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

    // ####################################################################################################################################################################################
    // INITIALIZE AND CONFIGURE

#ifndef SHADER_TESTING
    ERROR_CUDA(cudaSetDevice(0));
#endif

    // load config
    configuration config;
    config.load_configuration();

#ifndef SHADER_TESTING
    particles p;
    p.initialize(config);
#endif

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

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD!" << std::endl;
        return -1;
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

    // ####################################################################################################################################################################################
    // SHADERS

#ifdef SHADER_TEST_1

    shader_files shader_files(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);

    // Compile shaders
    GLuint vertexShader = compileShader(shader_files.vertexShaderSourceC, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(shader_files.fragmentShaderSourceC, GL_FRAGMENT_SHADER);

    // Link shaders into a program
    GLuint particleShaderProgram = linkProgram(vertexShader, fragmentShader);

    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // ####################################################################################################################################################################################
    // DRAWING???

    GLfloat vertices[] =
    {
        -0.5f, -0.5f * float(sqrt(3)) / 3, 0.0f, // Lower left corner
        0.5f, -0.5f * float(sqrt(3)) / 3, 0.0f, // Lower right corner
        0.0f, 0.5f * float(sqrt(3)) * 2 / 3, 0.0f // Upper corner
    };


    // Create reference containers for the Vartex Array Object and the Vertex Buffer Object
    GLuint VAO, VBO;

    // Generate the VAO and VBO with only 1 object each
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    // Make the VAO the current Vertex Array Object by binding it
    glBindVertexArray(VAO);

    // Bind the VBO specifying it's a GL_ARRAY_BUFFER
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // Introduce the vertices into the VBO
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Configure the Vertex Attribute so that OpenGL knows how to read the VBO
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // Enable the Vertex Attribute so that OpenGL knows to use it
    glEnableVertexAttribArray(0);

    // Bind both the VBO and VAO to 0 so that we don't accidentally modify the VAO and VBO we created
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // ####################################################################################################################################################################################

#endif




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

#ifdef SHADER_TEST_1
        // Tell OpenGL which Shader Program we want to use
        glUseProgram(particleShaderProgram);
        // Bind the VAO so OpenGL knows to use it
        glBindVertexArray(VAO);
        // Draw the triangle using the GL_TRIANGLES primitive
        glDrawArrays(GL_TRIANGLES, 0, 3);
#endif

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
#ifdef __EMSCRIPTEN__
    EMSCRIPTEN_MAINLOOP_END;
#endif

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

#ifdef SHADER_TEST_1
    // Delete all the objects we've created
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(particleShaderProgram);
#endif

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
