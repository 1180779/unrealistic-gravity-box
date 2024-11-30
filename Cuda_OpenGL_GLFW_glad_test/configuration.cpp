#include "configuration.h"

void configuration::load_configuration() 
{
    starting_wwidth = 1280;
    starting_wheigth = 720;

    g = -0.2f; //9.81f;
    radius = 1;

    particles_count = 30; // 10000

    maxabs_starting_velocity = 0.5f;
}