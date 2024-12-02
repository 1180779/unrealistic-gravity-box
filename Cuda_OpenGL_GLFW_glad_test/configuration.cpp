#include "configuration.hpp"

void configuration::load_configuration() 
{
    *this = configuration::preset::collisions();

    wwidth = 1280;
    wheigth = 720;
}

configuration configuration::preset::standard()
{
    configuration config;
    config.count = 100000;
    config.radius = 1.0f;

    config.g = 0.4f;
    config.maxabs_starting_xvelocity = 2.0f;
    config.maxabs_starting_yvelocity = 2.0f;
    return config;
}

configuration configuration::preset::collisions() 
{
    configuration config;
    config.count = 4;
    config.radius = 25.0f;

    config.g = 0.4f;
    config.maxabs_starting_xvelocity = 5.0f;
    config.maxabs_starting_yvelocity = 2.0f;
    return config;
}
