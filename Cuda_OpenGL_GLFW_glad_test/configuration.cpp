#include "configuration.h"

void configuration::load_configuration() 
{
    wwidth = 1280;
    wheigth = 720;

    g = 0.1f; //9.81f;
    radius = 1.0f;

    count = 100000; // 10000

    maxabs_starting_xvelocity = 0.5f;
    maxabs_starting_yvelocity = 0.5f;
}