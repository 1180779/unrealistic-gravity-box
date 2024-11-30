
#include <glad/glad.h>

#include "error_macros.h"
#include "configuration.h"

#include "imgui/imgui.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h" // openGL interoperability

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <glm/glm.hpp>

#include <vector>

struct particles_gpu {
    int size;
    int g;
    float radius;

    float* x;
    float* y;
    float* vx;
    float* vy;
    float* m;
    int* cell;
    GLfloat* color;
};

struct particles {
    particles_gpu gpu;

    thrust::host_vector<float> h_x;
    thrust::host_vector<float> h_y;
    thrust::host_vector<float> h_vx;
    thrust::host_vector<float> h_vy;
    thrust::host_vector<float> h_m;
    std::vector<glm::vec4> color;

    thrust::device_vector<float> d_vx;
    thrust::device_vector<float> d_vy;
    thrust::device_vector<float> d_m;
    thrust::device_vector<int> d_cell;

    void initialize(const configuration& config)
    {
        gpu.size = config.particles_count;
        gpu.g = config.g;
        gpu.radius = config.radius;

        h_x = thrust::host_vector<float>(gpu.size);
        h_y = thrust::host_vector<float>(gpu.size);
        h_vx = thrust::host_vector<float>(gpu.size);
        h_vy = thrust::host_vector<float>(gpu.size);
        h_m = thrust::host_vector<float>(gpu.size);
        color.resize(gpu.size);


        d_vx = thrust::device_vector<float>(gpu.size);
        d_vy = thrust::device_vector<float>(gpu.size);
        d_m = thrust::device_vector<float>(gpu.size);
        d_cell = thrust::device_vector<int>(gpu.size);

        gpu.vx = thrust::raw_pointer_cast(d_vx.data());
        gpu.vy = thrust::raw_pointer_cast(d_vy.data());
        gpu.m = thrust::raw_pointer_cast(d_m.data());
        gpu.cell = thrust::raw_pointer_cast(d_cell.data());

        srand((unsigned int)time(0));
        for (int i = 0; i < config.particles_count; ++i) {
            h_m[i] = 1.f;

            h_x[i] = rand() % (int)(config.starting_wwidth - 2 * gpu.radius) + gpu.radius;
            h_y[i] = rand() % (int)(config.starting_wheigth - 2 * gpu.radius) + gpu.radius;

            h_vx[i] = -config.maxabs_starting_velocity + static_cast <float> (rand()) /
                (static_cast <float> (RAND_MAX / (2 * config.maxabs_starting_velocity)));;
            h_vy[i] = -config.maxabs_starting_velocity + static_cast <float> (rand()) /
                (static_cast <float> (RAND_MAX / (2 * config.maxabs_starting_velocity)));;

            color[i] = glm::vec4((rand() % 256) / 255.0f, (rand() % 256) / 255.0f, (rand() % 256) / 255.0f, 1);
            //std::cout << "i = " << i << ", vx = " << h_vx[i] << ", vy = " << h_vy[i] << std::endl;
        }

        thrust::copy(h_vx.begin(), h_vx.end(), d_vx.begin());
        thrust::copy(h_vy.begin(), h_vy.end(), d_vy.begin());
        thrust::copy(h_m.begin(), h_m.end(), d_m.begin());
    }

    void mapFromVBO(cudaGraphicsResource*& cudaResource, float*& dest)
    {
        ERROR_CUDA(cudaGraphicsMapResources(1, &cudaResource, 0));
        void* devPtr;
        size_t size;
        ERROR_CUDA(cudaGraphicsResourceGetMappedPointer(&devPtr, &size, cudaResource));

        //std::cout << "mapped (size = " << size << ") bytes to cuda" << std::endl;
        if (size < gpu.size * sizeof(float))
            MY_ERROR("cudaGraphicsResourceGetMappedPointer: returned size is too small");
        dest = static_cast<float*>(devPtr);
    }

    void unmap(cudaGraphicsResource*& cudaResource)
    {
        // Step 4: Unmap the buffer for OpenGL
        ERROR_CUDA(cudaGraphicsUnmapResources(1, &cudaResource, 0));
    }

    // for debugging purposes 
    void copy_back() {
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
};
