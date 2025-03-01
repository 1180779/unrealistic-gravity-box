
#ifndef _LOGIC_1180779_
#define _LOGIC_1180779_

//#define DEBUG

#include "configuration.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <glm/glm.hpp>

#define CELL_BUFFER 2 // aditional entries in 

struct particles_temp_gpu 
{
    float* temp_x;
    float* temp_y;
    float* temp_vx;
    float* temp_vy;
    float* temp_m;
    int* temp_cell;
    glm::vec4* temp_color;
};

struct partciles_temp 
{
    particles_temp_gpu gpu;

    thrust::device_vector<float> temp_x;
    thrust::device_vector<float> temp_y;
    thrust::device_vector<float> temp_vx;
    thrust::device_vector<float> temp_vy;
    thrust::device_vector<float> temp_m;
    thrust::device_vector<int> temp_cell;
    thrust::device_vector<glm::vec4> temp_color;

    void initalize(int size);
};

struct particles_gpu 
{
    int size;
    float g;
    float radius;

    int* index;
    float* x;
    float* y;
    float* vx;
    float* vy;
    float* m;
    int* cell;

    int* cell_indexes;
    glm::vec4* color;
};

struct particles 
{
    int cell_count;
    int cell_size;
    particles_gpu gpu;

    thrust::host_vector<int> h_index;
    thrust::host_vector<float> h_x;
    thrust::host_vector<float> h_y;
    thrust::host_vector<float> h_vx;
    thrust::host_vector<float> h_vy;
    thrust::host_vector<float> h_m;
    std::vector<glm::vec4> color;

    thrust::device_vector<int> d_index;
    thrust::device_vector<float> d_vx;
    thrust::device_vector<float> d_vy;
    thrust::device_vector<float> d_m;
    thrust::device_vector<int> d_cell;



    // get cells first occurances
    thrust::device_vector<int> d_cell_keys;     // keys (size of particle count)
    thrust::device_vector<int> d_cell_indexes;  // indexes (size of particle count)
    thrust::device_vector<int> d_indices;       // [0, 1, ..., particle count - 1]

    thrust::host_vector<int> h_cell_keys;
    thrust::host_vector<int> h_cell_indexes;
    thrust::host_vector<int> h_cell_indexes_final;

    thrust::device_vector<int> d_cell_indexes_final; // final indexes (size of number of cells)

    void initialize(const configuration& config);

    template <typename T>
    void mapFromVBO(cudaGraphicsResource*& cudaResource, T*& dest)
    {
        ERROR_CUDA(cudaGraphicsMapResources(1, &cudaResource, 0));
        void* devPtr;
        size_t size;
        ERROR_CUDA(cudaGraphicsResourceGetMappedPointer(&devPtr, &size, cudaResource));

        if (size < gpu.size * sizeof(T))
            ERROR_("cudaGraphicsResourceGetMappedPointer: returned size is too small");
        dest = static_cast<T*>(devPtr);
    }

    void unmap(cudaGraphicsResource*& cudaResource);

    int getCellIndexesPart1();
#ifdef DEBUG
    void getCellIndexesPart2(int unique_count);

    // for debugging purposes 
    void copy_back(int grid_width, int grid_heigth);
#endif
};

#endif
