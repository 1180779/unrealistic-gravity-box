
#include <glad/glad.h>

#include "error_macros.hpp"
#include "configuration.hpp"
#include "logic.hpp"
#include "shaders.hpp"

#include <stdio.h>

#include "thrust/sort.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h" // openGL interopperability

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"


#define BLOCK_SIZE 16

#define VERTEX_SHADER_SOURCE "shaders/cuda.vert"
#define FRAGMENT_SHADER_SOURCE "shaders/cuda.frag"
//#define GEOMETRY_SHADER_SOURCE "shaders/cuda.geom"
#define GEOMETRY_SHADER_SOURCE "shaders/cuda_circle.geom"

#pragma region KERNELS

__global__ void updateParticlesKernel(particles_gpu p, float wwidth, float wheight, float cell_size, int grid_width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // particle index
    if (i >= p.size)
        return;

    p.x[i] += p.vx[i];
    p.y[i] += p.vy[i];
    p.vy[i] -= p.g * p.m[i];

    // check window bounds
    if (p.x[i] >= wwidth - p.radius)
        p.vx[i] = -abs(p.vx[i]);
    if (p.x[i] <= 0.f + p.radius)
        p.vx[i] = abs(p.vx[i]);

    //if (p.y[i] >= wheight)
    //    p.vy[i] = -abs(p.vy[i]);
    if (p.y[i] <= 0.f + p.radius)
        p.vy[i] = abs(p.vy[i]);

    int cell_x = p.x[i] / cell_size;
    int cell_y = p.y[i] / cell_size;
    p.cell[i] = cell_x + cell_y * grid_width;
    p.index[i] = i;
}

__global__ void lowerBoundKernel(int *keys, int *res, int cell_count, int *cell, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= cell_count + CELL_BUFFER)
        return;
    int key = i; // keys[i];

    int low = 0;
    int high = size;
    while (low < high) {
        int mid = low + (high - low) / 2;  // Avoid overflow
        if (cell[mid] < key) {
            low = mid + 1;
        }
        else {
            high = mid;
        }
    }
    res[i] = low;
}

__global__ void reorderParticleData(particles_gpu p, particles_temp_gpu p_temp) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // particle index
    if (i >= p.size)
        return;
    int index = p.index[i];
    p_temp.temp_x[i] = p.x[index];
    p_temp.temp_y[i] = p.y[index];
    p_temp.temp_vx[i] = p.vx[index];
    p_temp.temp_vy[i] = p.vy[index];
    p_temp.temp_m[i] = p.m[index];
    p_temp.temp_color[i] = p.color[index];
}

__global__ void copyParticleDataBack(particles_gpu p, particles_temp_gpu p_temp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // particle index
    if (i >= p.size)
        return;
    p.x[i] = p_temp.temp_x[i];
    p.y[i] = p_temp.temp_y[i];
    p.vx[i] = p_temp.temp_vx[i];
    p.vy[i] = p_temp.temp_vy[i];
    p.m[i] = p_temp.temp_m[i];
    p.color[i] = p_temp.temp_color[i];
}

__device__ void particlesCollisionCheck(particles_gpu p, int i, int j) 
{
    float dist_x = p.x[i] - p.x[j];
    float dist_y = p.y[i] - p.y[j];
    float dist = sqrt(dist_x * dist_x + dist_y * dist_y);
    if (dist <= 2 * p.radius) {
        float contact_angle = atan2(dist_y, dist_x);

        float vi_norm = p.vx[i] * cos(contact_angle) + p.vy[i] * sin(contact_angle);
        float vi_tang = -p.vx[i] * sin(contact_angle) + p.vy[i] * cos(contact_angle);

        float vj_norm = p.vx[j] * cos(contact_angle) + p.vy[j] * sin(contact_angle);
        float vj_tang = -p.vx[j] * sin(contact_angle) + p.vy[j] * cos(contact_angle);

        float vi_norm_new = vi_norm * (p.m[i] - p.m[j]) + 2 * p.m[j] * vj_norm / (p.m[i] + p.m[j]);
        float vj_norm_new = vj_norm * (p.m[j] - p.m[i]) + 2 * p.m[i] * vi_norm / (p.m[i] + p.m[j]);

        p.vx[i] = vi_norm_new * cos(contact_angle) - vi_tang * sin(contact_angle);
        p.vy[i] = vi_norm_new * sin(contact_angle) + vi_tang * cos(contact_angle);

        p.vx[j] = vj_norm_new * cos(contact_angle) - vj_tang * sin(contact_angle);
        p.vy[j] = vj_norm_new * sin(contact_angle) + vj_tang * cos(contact_angle);
    }
}

__global__ void particlesCollisionKernel(particles_gpu p, int cell_size, int grid_width, int grid_height, int cell_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // particle index

    if (i >= p.size)
        return;

    // check if collision even works
    //for (int j = i + 1; j < p.size; ++j)
    //    particlesCollisionCheck(p, i, j);
    //return;


    int cell = p.cell[i];
    if (cell < 0 || cell >= cell_count)
        return;

    // the same cell
    int start;
    int end; 
    end = p.cell_indexes[cell + 1];
    
        for (int j = i + 1; j < end; ++j)
            particlesCollisionCheck(p, i, j);

    // there is a row of cells above
    if (cell + grid_width < cell_count) {
        // there is a column of cells on the left
        if (cell % grid_width != 0) {
            // cell above left 
            start = p.cell_indexes[cell + grid_width - 1];
            end = p.cell_indexes[cell + grid_width];
                for (int j = start; j < end; ++j)
                    particlesCollisionCheck(p, i, j);
        }

    //    // cell above
        start = p.cell_indexes[cell + grid_width];
        end = p.cell_indexes[cell + grid_width + 1];
            for (int j = start; j < end; ++j)
                particlesCollisionCheck(p, i, j);

    //    // check if column of cells on the right
        if ((cell + 1) % grid_width != 0) {
            // cell above right
            start = p.cell_indexes[cell + grid_width + 1];
            end = p.cell_indexes[cell + grid_width + 2];
                for (int j = start; j < end; ++j)
                    particlesCollisionCheck(p, i, j);
        }
    }

    // check if column of cells on the right
    if ((cell + 1) % grid_width != 0) {
        // cell right
        start = p.cell_indexes[cell + 1];
        end = p.cell_indexes[cell + 2];
            for (int j = start; j < end; ++j)
                particlesCollisionCheck(p, i, j);
    }
}

#pragma endregion KERNELS

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

    int wwidth = config.wwidth;
    int wheigth = config.wheigth;

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

    std::cout << "Loading shader files..." << std::endl;
    shader_files shader_files(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE, GEOMETRY_SHADER_SOURCE);
    shader_files.print();

    std::cout << "Compiling vertex shader..." << std::endl;
    GLuint vertexShader = compileShader(shader_files.vertexShaderSourceC, GL_VERTEX_SHADER);
    std::cout << "Compiling fragment shader..." << std::endl;
    GLuint fragmentShader = compileShader(shader_files.fragmentShaderSourceC, GL_FRAGMENT_SHADER);
    std::cout << "Compiling geometry shader..." << std::endl;
    GLuint geomertyShader = compileShader(shader_files.geometryShaderSourceC, GL_GEOMETRY_SHADER);

    // Link shaders
    GLuint particleShaderProgram = linkProgram(vertexShader, fragmentShader, geomertyShader);

    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteShader(geomertyShader);
    std::cout << "Shaders linked!" << std::endl;

    ImVec4 clear_color = ImVec4(0.f, 0.f, 0.f, 1.00f);
    bool startSimulation = false;
    
    // Configuration loop
    while (!glfwWindowShouldClose(window) && !startSimulation)
    {
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
#define CONFIG_WINDOW_WIDTH 400
#define CONFIG_WINDOW_HEIGHT 180
#define FIELD_WIDTH 150
            ImGui::SetNextWindowSize(ImVec2(CONFIG_WINDOW_WIDTH, CONFIG_WINDOW_HEIGHT));
            ImGui::Begin("Simulation configuration", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

            ImGui::SetNextItemWidth(FIELD_WIDTH);
            ImGui::InputInt("Particles count", &config.count, 1000, 10000);
            if (config.count <= 0)
                config.count = 1000;

            ImGui::SetNextItemWidth(FIELD_WIDTH);
            ImGui::InputFloat("Maximum abs velocity in x axis", &config.maxabs_starting_xvelocity, 0.05f, 0.02f);
            if (config.maxabs_starting_xvelocity <= 0.f)
                config.maxabs_starting_xvelocity = 0.05f;

            ImGui::SetNextItemWidth(FIELD_WIDTH);
            ImGui::InputFloat("Maximum abs velocity in y axis", &config.maxabs_starting_yvelocity, 0.05f, 0.02f);
            if (config.maxabs_starting_yvelocity <= 0.f)
                config.maxabs_starting_yvelocity = 0.05f;

            ImGui::SetNextItemWidth(FIELD_WIDTH);
            ImGui::InputFloat("Acceleration (g)", &config.g, 0.005f, 0.01f);
            ImGui::SetNextItemWidth(FIELD_WIDTH);
            ImGui::InputFloat("Particle radius", &config.radius, 0.5f, 1.0f);
            if (config.radius <= 0.5f)
                config.radius = 0.5f;

            startSimulation = ImGui::Button("Start simulation");

            ImGui::End();
        }

        {
            ImGui::SetNextWindowSize(ImVec2(200, 100));
            ImGui::Begin("Presets", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);


            bool standartPreset = false;
            standartPreset = ImGui::Button("Standard Preset");
            if (standartPreset) {
                config = configuration::preset::standard();
            }

            bool collisionsPreset = false;
            collisionsPreset = ImGui::Button("Collisions Preset");
            if (collisionsPreset) {
                config = configuration::preset::collisions();
            }
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
        glfwSwapBuffers(window);
    }

    std::cout << "Initializing particles..." << std::endl;
    particles p;
    p.initialize(config);
    partciles_temp p_temp;
    p_temp.initalize(p.gpu.size);

    std::cout << "Initializing buffers..." << std::endl;
    p.initialize(config);

    GLuint VAO, VBO[3];

    glGenVertexArrays(1, &VAO);
    glGenBuffers(3, VBO);


    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)* p.gpu.size, p.h_x.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)* p.gpu.size, p.h_y.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * p.gpu.size, p.color.data(), GL_DYNAMIC_DRAW);

    // register the buffers with CUDA
    cudaGraphicsResource* cudaResourceX;
    ERROR_CUDA(cudaGraphicsGLRegisterBuffer(&cudaResourceX, VBO[0], cudaGraphicsMapFlagsNone));
    cudaGraphicsResource* cudaResourceY;
    ERROR_CUDA(cudaGraphicsGLRegisterBuffer(&cudaResourceY, VBO[1], cudaGraphicsMapFlagsNone));
    cudaGraphicsResource* cudaResourceColor;
    ERROR_CUDA(cudaGraphicsGLRegisterBuffer(&cudaResourceColor, VBO[2], cudaGraphicsMapFlagsNone));

    // VBO[0]: x pos
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(0); // location 0 (x)

    // VBO[1]: y pos
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); // location 1 (y)

    // VBO[2]: color
    glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2); // location 2 (color)

    glUseProgram(particleShaderProgram);
    glBindVertexArray(VAO);

    // map VBOs for CUDA use
    p.mapFromVBO(cudaResourceX, p.gpu.x);
    p.mapFromVBO(cudaResourceY, p.gpu.y);
    p.mapFromVBO(cudaResourceColor, p.gpu.color);

    // get uniform location
    GLint screenWidthLoc = glGetUniformLocation(particleShaderProgram, "screenWidth");
    GLint screenHeightLoc = glGetUniformLocation(particleShaderProgram, "screenHeight");
    GLint radiusyLocation = glGetUniformLocation(particleShaderProgram, "radius_x");
    GLint radiusxLocation = glGetUniformLocation(particleShaderProgram, "radius_y");

    p.cell_size = std::min(
        static_cast<int>(config.radius * 4), 
        std::min(wwidth, wheigth) );

    std::cout << "Cell size = " << p.cell_size << std::endl;

    std::cout << "Starting simulation..." << std::endl;

    // Main loop
    while (!glfwWindowShouldClose(window))
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

        int grid_width = static_cast<int>(
            ceil(static_cast<double>(wwidth) / static_cast<double>(p.cell_size)) );
        int grid_heigth = static_cast<int>(
            ceil(static_cast<double>(wheigth) / static_cast<double>(p.cell_size)));
        p.cell_count = grid_width * grid_heigth;
        
        // uniform data for shaders
        glUniform1f(screenWidthLoc, wwidth);
        glUniform1f(screenHeightLoc, wheigth);
        glUniform1f(radiusxLocation, p.gpu.radius / wwidth * 2.0f);
        glUniform1f(radiusyLocation, p.gpu.radius / wheigth * 2.0f);

        dim3 blocks = dim3(p.gpu.size / BLOCK_SIZE + 1);
        dim3 threads = dim3(BLOCK_SIZE);

        // update particles positions
        updateParticlesKernel<<<blocks, threads>>>(p.gpu, wwidth, wheigth, p.cell_size, grid_width);
        ERROR_CUDA(cudaGetLastError());
        ERROR_CUDA(cudaDeviceSynchronize());
        
        //p.copy_back(grid_width, grid_heigth);

        // sort indexes and copy the data back and forth
        thrust::sort_by_key(p.d_cell.begin(), p.d_cell.end(), p.d_index.begin());
        ERROR_CUDA(cudaGetLastError());
        ERROR_CUDA(cudaDeviceSynchronize());
        
        reorderParticleData<<<blocks, threads>>>(p.gpu, p_temp.gpu);
        ERROR_CUDA(cudaGetLastError());
        ERROR_CUDA(cudaDeviceSynchronize());

        copyParticleDataBack<<<blocks, threads>>>(p.gpu, p_temp.gpu);
        ERROR_CUDA(cudaGetLastError());
        ERROR_CUDA(cudaDeviceSynchronize());

        dim3 blocks_lb = dim3(p.cell_count / BLOCK_SIZE + 1);
        dim3 threads_lb = dim3(BLOCK_SIZE);
        int unique_count = p.getCellIndexesPart1();
        lowerBoundKernel<<<blocks_lb, threads_lb>>>(
            thrust::raw_pointer_cast(p.d_indices.data()),
            thrust::raw_pointer_cast(p.d_cell_indexes_final.data()),
            p.cell_count,
            p.gpu.cell,
            p.gpu.size);
        ERROR_CUDA(cudaGetLastError());
        ERROR_CUDA(cudaDeviceSynchronize());
        //p.getCellIndexesPart2(unique_count);
        //p.copy_back(grid_width, grid_heigth);

        // collisions
        particlesCollisionKernel<<<blocks, threads>>>(p.gpu, p.cell_size, grid_width, grid_heigth, p.cell_count);
        ERROR_CUDA(cudaGetLastError());
        ERROR_CUDA(cudaDeviceSynchronize());


        {
            ImGui::SetNextWindowSize(ImVec2(300, 80));
            ImGui::Begin("Dynamic settings", NULL, ImGuiWindowFlags_NoResize);

            ImGui::SetNextItemWidth(150);
            ImGui::ColorEdit3("Change background", (float*)&clear_color); // Edit 3 floats representing a color

            ImGui::Text("Average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawArrays(GL_POINTS, 0, p.gpu.size);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // unmap and unregister buffers for CUDA use
    p.unmap(cudaResourceX);
    p.unmap(cudaResourceY);
    p.unmap(cudaResourceColor);
    ERROR_CUDA(cudaGraphicsUnregisterResource(cudaResourceX));
    ERROR_CUDA(cudaGraphicsUnregisterResource(cudaResourceY));
    ERROR_CUDA(cudaGraphicsUnregisterResource(cudaResourceColor));

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // delete openGL buffers
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO[0]);
    glDeleteBuffers(1, &VBO[1]);
    glDeleteBuffers(1, &VBO[2]);
    glDeleteProgram(particleShaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
