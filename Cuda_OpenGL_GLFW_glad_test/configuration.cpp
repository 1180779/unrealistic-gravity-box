#include "configuration.hpp"

//#define SMALL
//#define BIG_SPEEDY

void configuration::load_configuration() 
{
    wwidth = 1280;
    wheigth = 720;
    g = 0.1f; //9.81f;

#ifdef SMALL
    radius = 1.0f;

    count = 100000;

    maxabs_starting_xvelocity = 2.0f;
    maxabs_starting_yvelocity = 2.0f;

#elif defined BIG_SPEEDY

    radius = 25.0f;

    count = 4;

    maxabs_starting_xvelocity = 10.0f;
    maxabs_starting_yvelocity = 2.0f;

#else

    radius = 25.0f;

    count = 4;

    maxabs_starting_xvelocity = 5.0f;
    maxabs_starting_yvelocity = 2.0f;

#endif

}