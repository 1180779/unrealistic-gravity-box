#include "configuration.h"

void configuration::load_configuration() 
{
    starting_wwidth = 1280;
    starting_wheigth = 720;

    g = -0.4f; //9.81f;

    particles_count = 100; // 10000

    maxabs_starting_velocity = 2;
}