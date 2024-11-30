
#include "error_macros.hpp"
#include "logic.hpp"

#include <glad/glad.h>
#include "cuda_gl_interop.h" // openGL interoperability
#include <glm/glm.hpp>

void partciles_temp::initalize(int size) 
{
    temp_x = thrust::device_vector<float>(size);
    temp_y = thrust::device_vector<float>(size);
    temp_vx = thrust::device_vector<float>(size);
    temp_vy = thrust::device_vector<float>(size);
    temp_m = thrust::device_vector<float>(size);
    temp_color = thrust::device_vector<glm::vec4>(size);

    gpu.temp_x = thrust::raw_pointer_cast(temp_x.data());
    gpu.temp_y = thrust::raw_pointer_cast(temp_y.data());
    gpu.temp_vx = thrust::raw_pointer_cast(temp_vx.data());
    gpu.temp_vy = thrust::raw_pointer_cast(temp_vy.data());
    gpu.temp_m = thrust::raw_pointer_cast(temp_m.data());
    gpu.temp_color = thrust::raw_pointer_cast(temp_color.data());
}

void particles::initialize(const configuration& config)
{
    gpu.size = config.count;
    gpu.g = config.g;
    gpu.radius = config.radius;

    h_index = thrust::host_vector<int>(gpu.size);
    h_x = thrust::host_vector<float>(gpu.size);
    h_y = thrust::host_vector<float>(gpu.size);
    h_vx = thrust::host_vector<float>(gpu.size);
    h_vy = thrust::host_vector<float>(gpu.size);
    h_m = thrust::host_vector<float>(gpu.size);
    color.resize(gpu.size);

    d_index = thrust::device_vector<float>(gpu.size);
    d_vx = thrust::device_vector<float>(gpu.size);
    d_vy = thrust::device_vector<float>(gpu.size);
    d_m = thrust::device_vector<float>(gpu.size);
    d_cell = thrust::device_vector<int>(gpu.size);

    gpu.index = thrust::raw_pointer_cast(d_index.data());
    gpu.vx = thrust::raw_pointer_cast(d_vx.data());
    gpu.vy = thrust::raw_pointer_cast(d_vy.data());
    gpu.m = thrust::raw_pointer_cast(d_m.data());
    gpu.cell = thrust::raw_pointer_cast(d_cell.data());

    srand((unsigned int)time(0));
    for (int i = 0; i < config.count; ++i) {
        h_index[i] = i;
        h_m[i] = 1.f;

        h_x[i] = rand() % (int)(config.wwidth - 2 * gpu.radius) + gpu.radius;
        h_y[i] = rand() % (int)(config.wheigth - 2 * gpu.radius) + gpu.radius;

        h_vx[i] = -config.maxabs_starting_xvelocity + static_cast <float> (rand()) /
            (static_cast <float> (RAND_MAX / (2 * config.maxabs_starting_xvelocity)));;
        h_vy[i] = -config.maxabs_starting_yvelocity + static_cast <float> (rand()) /
            (static_cast <float> (RAND_MAX / (2 * config.maxabs_starting_yvelocity)));;

        color[i] = glm::vec4((rand() % 256) / 255.0f, (rand() % 256) / 255.0f, (rand() % 256) / 255.0f, 1);
    }

    thrust::copy(h_index.begin(), h_index.end(), d_index.begin());
    thrust::copy(h_vx.begin(), h_vx.end(), d_vx.begin());
    thrust::copy(h_vy.begin(), h_vy.end(), d_vy.begin());
    thrust::copy(h_m.begin(), h_m.end(), d_m.begin());
}

void particles::unmap(cudaGraphicsResource*& cudaResource)
{
    ERROR_CUDA(cudaGraphicsUnmapResources(1, &cudaResource, 0));
}

// for debugging purposes 
void particles::copy_back() {
    float* raw_x = h_x.data();
    ERROR_CUDA(cudaMemcpy(raw_x, gpu.x, sizeof(float) * gpu.size, cudaMemcpyDeviceToHost));
    ERROR_CUDA(cudaGetLastError());
    ERROR_CUDA(cudaDeviceSynchronize());

    float* raw_y = h_y.data();
    ERROR_CUDA(cudaMemcpy(raw_y, gpu.y, sizeof(float) * gpu.size, cudaMemcpyDeviceToHost));
    ERROR_CUDA(cudaGetLastError());
    ERROR_CUDA(cudaDeviceSynchronize());
        
    float* raw_vy = h_vy.data();
    ERROR_CUDA(cudaMemcpy(raw_vy, gpu.vy, sizeof(float) * gpu.size, cudaMemcpyDeviceToHost));
    ERROR_CUDA(cudaGetLastError());
    ERROR_CUDA(cudaDeviceSynchronize());
        
    for (int i = 0; i < gpu.size; ++i) {
        std::cout << "i = " << i << " | x = " << h_x[i] << ", y = " << h_y[i]  << ", vy = " << h_vy[i] << std::endl;
    }
}