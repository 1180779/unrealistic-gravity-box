
#ifndef _CONFIGURATION_1180779_
#define _CONFIGURATION_1180779_

#include <string>

class configuration
{
public:
    class preset {
    public:
        static configuration standard();
        static configuration collisions();
    };

    void load_from_file(std::string filename);

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
