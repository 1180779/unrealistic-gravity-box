
#ifndef _CONFIGURATION_1180779_
#define _CONFIGURATION_1180779_

#define DEBUG

class configuration
{
public:
    int wwidth;
    int wheigth;

    float g;
    float maxabs_starting_xvelocity; // in pixels per frame
    float maxabs_starting_yvelocity; // in pixels per frame
    float radius;

    int count;

    void load_configuration();
};

#endif
