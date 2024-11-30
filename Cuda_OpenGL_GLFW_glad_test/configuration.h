#pragma once
class configuration
{
public:
    int starting_wwidth;
    int starting_wheigth;

    float g;
    float maxabs_starting_velocity; // in pixels per frame
    float radius;

    int particles_count;

    void load_configuration();
};

