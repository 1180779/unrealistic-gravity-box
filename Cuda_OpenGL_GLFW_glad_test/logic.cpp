
#include "configuration.h"

#include "imgui/imgui.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <glm/glm.hpp>

#include <vector>

struct particles_gpu {
    int size;
    int g;

    float* x;
    float* y;
    float* vx;
    float* vy;
    float* m;
    float* radius;
    int* cell;
};

struct particles {
    particles_gpu gpu;

    thrust::host_vector<float> h_x;
    thrust::host_vector<float> h_y;
    thrust::host_vector<float> h_vx;
    thrust::host_vector<float> h_vy;
    thrust::host_vector<float> h_m;
    thrust::host_vector<float> h_radius;
    std::vector<glm::vec4> color;
    std::vector<glm::vec2> h_pos;

    thrust::device_vector<float> d_x;
    thrust::device_vector<float> d_y;
    thrust::device_vector<float> d_vx;
    thrust::device_vector<float> d_vy;
    thrust::device_vector<float> d_m;
    thrust::device_vector<float> d_radius;
    thrust::device_vector<int> d_cell;

    void initialize(const configuration &config) 
    {
        gpu.size = config.particles_count;
        gpu.g = config.g;

        h_x = thrust::host_vector<float>(gpu.size);
        h_y = thrust::host_vector<float>(gpu.size);
        h_vx = thrust::host_vector<float>(gpu.size);
        h_vy = thrust::host_vector<float>(gpu.size);
        h_m = thrust::host_vector<float>(gpu.size);
        h_radius = thrust::host_vector<float>(gpu.size);
        color.resize(gpu.size);
        h_pos = std::vector<glm::vec2>(gpu.size);

        d_x = thrust::device_vector<float>(gpu.size);
        d_y = thrust::device_vector<float>(gpu.size);
        d_vx = thrust::device_vector<float>(gpu.size);
        d_vy = thrust::device_vector<float>(gpu.size);
        d_m = thrust::device_vector<float>(gpu.size);
        d_radius = thrust::device_vector<float>(gpu.size);
        d_cell = thrust::device_vector<int>(gpu.size);

        gpu.x = thrust::raw_pointer_cast(d_x.data());
        gpu.y = thrust::raw_pointer_cast(d_y.data());
        gpu.vx = thrust::raw_pointer_cast(d_vx.data());
        gpu.vy = thrust::raw_pointer_cast(d_vy.data());
        gpu.m = thrust::raw_pointer_cast(d_m.data());
        gpu.radius = thrust::raw_pointer_cast(d_radius.data());
        gpu.cell = thrust::raw_pointer_cast(d_cell.data());

        srand((unsigned int)time(0));
        for (int i = 0; i < config.particles_count; ++i) {
            h_radius[i] = 1.f;
            h_m[i] = 1.f;

            h_x[i] = rand() % (int)(config.starting_wwidth - 2 * h_radius[i]) + h_radius[i];
            h_y[i] = rand() % (int)(config.starting_wheigth - 2 * h_radius[i]) + h_radius[i];

            h_vx[i] = -config.maxabs_starting_velocity + static_cast <float> (rand()) / 
                (static_cast <float> (RAND_MAX / (2*config.maxabs_starting_velocity)));;
            h_vy[i] = -config.maxabs_starting_velocity + static_cast <float> (rand()) /
                (static_cast <float> (RAND_MAX / (2 * config.maxabs_starting_velocity)));;

            color[i] = glm::vec4((rand() % 256)/255.0f, (rand() % 256)/255.0f, (rand() % 256)/255.0f, 1);
            //std::cout << "i = " << i << ", vx = " << h_vx[i] << ", vy = " << h_vy[i] << std::endl;
        }

        thrust::copy(h_x.begin(), h_x.end(), d_x.begin());
        thrust::copy(h_y.begin(), h_y.end(), d_y.begin());
        thrust::copy(h_vx.begin(), h_vx.end(), d_vx.begin());
        thrust::copy(h_vy.begin(), h_vy.end(), d_vy.begin());
        thrust::copy(h_m.begin(), h_m.end(), d_m.begin());
        thrust::copy(h_radius.begin(), h_radius.end(), d_radius.begin());
    }

    void copy_results_back() 
    {
        thrust::copy(d_x.begin(), d_x.end(), h_x.begin());
        thrust::copy(d_y.begin(), d_y.end(), h_y.begin());

        //thrust::transform(
        //    d_x.begin(), d_x.end(),
        //    h_pos.begin(),
        //    [] (float x) {
        //        return glm::vec2(x, 0.0f); // Only set x; y remains 0 for now
        //    }
        //);

        //// Fill the y-components of host_positions
        //thrust::transform(
        //    d_y.begin(), d_y.end(),
        //    h_pos.begin(),
        //    [] (float y) {
        //        return glm::vec2(0.0f, y); // Add y; x remains unchanged
        //    },
        //    thrust::plus<glm::vec2>() // Combine with existing values
        //);
    }
};
