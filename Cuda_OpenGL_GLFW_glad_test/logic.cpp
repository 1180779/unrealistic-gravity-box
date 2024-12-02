
#include "error_macros.hpp"
#include "logic.hpp"

#include <glad/glad.h>
#include "cuda_gl_interop.h" // openGL interoperability
#include "thrust/sequence.h"
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
    // copy configuration
    gpu.size = config.count;
    gpu.g = config.g;
    gpu.radius = config.radius;

    // allocate host vectors
    h_index = thrust::host_vector<int>(gpu.size);
    h_x = thrust::host_vector<float>(gpu.size);
    h_y = thrust::host_vector<float>(gpu.size);
    h_vx = thrust::host_vector<float>(gpu.size);
    h_vy = thrust::host_vector<float>(gpu.size);
    h_m = thrust::host_vector<float>(gpu.size);
    color.resize(gpu.size);

    // allocate device vectors
    d_index = thrust::device_vector<float>(gpu.size);
    d_vx = thrust::device_vector<float>(gpu.size);
    d_vy = thrust::device_vector<float>(gpu.size);
    d_m = thrust::device_vector<float>(gpu.size);
    d_cell = thrust::device_vector<int>(gpu.size);

    d_cell_indexes = thrust::device_vector<int>(gpu.size);
    d_cell_keys = thrust::device_vector<int>(gpu.size);
    d_indices = thrust::device_vector<int>(gpu.size);
    thrust::sequence(d_indices.begin(), d_indices.end());

    // get raw pointers to use on gpu
    gpu.index = thrust::raw_pointer_cast(d_index.data());
    gpu.vx = thrust::raw_pointer_cast(d_vx.data());
    gpu.vy = thrust::raw_pointer_cast(d_vy.data());
    gpu.m = thrust::raw_pointer_cast(d_m.data());
    gpu.cell = thrust::raw_pointer_cast(d_cell.data());
    gpu.cell_indexes = thrust::raw_pointer_cast(d_cell_indexes_final.data());

    // get random particle data
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

    // copy particle data to gpu
    thrust::copy(h_index.begin(), h_index.end(), d_index.begin());
    thrust::copy(h_vx.begin(), h_vx.end(), d_vx.begin());
    thrust::copy(h_vy.begin(), h_vy.end(), d_vy.begin());
    thrust::copy(h_m.begin(), h_m.end(), d_m.begin());
}

void particles::unmap(cudaGraphicsResource*& cudaResource)
{
    ERROR_CUDA(cudaGraphicsUnmapResources(1, &cudaResource, 0));
}

void particles::getCellIndexes() {
    // reduce by key -> get the unique keys and their indexes 
    // from particle cell data
    auto end = thrust::reduce_by_key(
        d_cell.begin(),
        d_cell.end(),
        d_indices.begin(),
        d_cell_keys.begin(),
        d_cell_indexes.begin(),
        thrust::equal_to<int>(),
        thrust::minimum<int>()
    );

    // resize vectors if necessary
    int unique_count = end.first - d_cell_keys.begin();
    if (h_cell_keys.size() < unique_count)
        h_cell_keys.resize(unique_count);
    if (h_cell_indexes.size() < unique_count)
        h_cell_indexes.resize(unique_count);

    if (h_cell_indexes_final.size() < cell_count)
        h_cell_indexes_final.resize(cell_count);
    if (d_cell_indexes_final.size() < cell_count) {
        d_cell_indexes_final.resize(cell_count);
        gpu.cell_indexes = thrust::raw_pointer_cast(d_cell_indexes_final.data());
    }
    
    // copy the results to cpu
    ERROR_CUDA(cudaMemcpy(thrust::raw_pointer_cast(h_cell_keys.data()), 
        thrust::raw_pointer_cast(d_cell_keys.data()), 
        sizeof(int) * unique_count, cudaMemcpyDeviceToHost));
    ERROR_CUDA(cudaGetLastError());
    ERROR_CUDA(cudaDeviceSynchronize());

    ERROR_CUDA(cudaMemcpy(thrust::raw_pointer_cast(h_cell_indexes.data()),
        thrust::raw_pointer_cast(d_cell_indexes.data()),
        sizeof(int) * unique_count, cudaMemcpyDeviceToHost));
    ERROR_CUDA(cudaGetLastError());
    ERROR_CUDA(cudaDeviceSynchronize());

    // change partial data to indexes of all cells
    printf("\nunique_count = % 5d. \n%10s", unique_count, "Keys:");
    for (int i = 0; i < unique_count; ++i)
        printf("%5d", h_cell_keys[i]);
    printf("\n%10s", "Indexes:");
    for (int i = 0; i < unique_count; ++i)
        printf("%5d", h_cell_indexes[i]);
    printf("\n");

    int j = unique_count - 1;
    for (int i = cell_count - 1; i >= 0; --i) {
        int temp = h_cell_keys[j];
        if (h_cell_keys[j] != i) {
            if (i + 1 < cell_count)
                h_cell_indexes_final[i] = h_cell_indexes_final[i + 1];
            else
                h_cell_indexes_final[i] = gpu.size - 1;
            continue;
        }
        h_cell_indexes_final[i] = h_cell_indexes[j];
        --j;
    }

    // copy final cell indexes back to gpu
    ERROR_CUDA(cudaMemcpy(thrust::raw_pointer_cast(d_cell_indexes_final.data()),
        thrust::raw_pointer_cast(h_cell_indexes_final.data()),
        sizeof(int) * cell_count, cudaMemcpyHostToDevice));
    ERROR_CUDA(cudaGetLastError());
    ERROR_CUDA(cudaDeviceSynchronize());
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
        
    float* raw_vx = h_vx.data();
    ERROR_CUDA(cudaMemcpy(raw_vx, gpu.vx, sizeof(float) * gpu.size, cudaMemcpyDeviceToHost));
    ERROR_CUDA(cudaGetLastError());
    ERROR_CUDA(cudaDeviceSynchronize());

    float* raw_vy = h_vy.data();
    ERROR_CUDA(cudaMemcpy(raw_vy, gpu.vy, sizeof(float) * gpu.size, cudaMemcpyDeviceToHost));
    ERROR_CUDA(cudaGetLastError());
    ERROR_CUDA(cudaDeviceSynchronize());

    int* raw_cell = new int[gpu.size];
    ERROR_CUDA(cudaMemcpy(raw_cell, gpu.cell, sizeof(int) * gpu.size, cudaMemcpyDeviceToHost));
    ERROR_CUDA(cudaGetLastError());
    ERROR_CUDA(cudaDeviceSynchronize());

    printf("\n\nCELL DATA(cell_size = %d), (d_indexes size = %d)\n%10s", cell_size, d_cell_indexes_final.size(), "keys:");
    for (int i = 0; i < cell_count; ++i) {
        printf("%5d ", i);
    }
    printf("\n%10s", "indexes:");
    for (int i = 0; i < cell_count; ++i) {
        printf("%5d ", h_cell_indexes_final[i]);
    }

    printf("\nPARTICLE DATA: \n");
    for (int i = 0; i < gpu.size; ++i) {
        printf("i = %5d | cell = %5d | x = %5.3f | y = %5.3f | vx = %5.3f | vy = %5.3f\n", i, raw_cell[i], h_x[i], h_y[i], h_vx[i], h_vy[i]);
    }
    delete[] raw_cell;
}