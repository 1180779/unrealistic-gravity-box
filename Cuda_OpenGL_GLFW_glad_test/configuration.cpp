#include "configuration.hpp"

#include <iostream>
#include <fstream>
#include <string>

void configuration::load_configuration() 
{
    *this = configuration::preset::standard();

    wwidth = 1280;
    wheigth = 720;
}

static void erase_white(std::string &str) 
{
    int i = 0;
    while (i < str.size() && (str[i] == ' ' || str[i] == '\t')) {
        ++i;
    }
    str.erase(0, i);
}

static std::string nextstr_to_space(std::string &str) 
{
    erase_white(str);
    int i = 0;
    while (i < str.size() && str[i] != ' ')
        ++i;
    if (i == str.size())
        return str.substr();

    std::string res = str.substr(0, i);
    str = str.substr(i + 1, str.size() - i - 1);
    erase_white(str);
    return res;
}

void configuration::load_from_file(std::string filename) 
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Could not open configuration file (\'config.txt\')" << std::endl;
        return;
    }

    std::cout << "READING FILE: " << std::endl;
    std::string line;
    while (std::getline(file, line)) {
        if (line.size() == 0)
            continue;
        if (line[0] == '#')
            continue;

        std::string option = nextstr_to_space(line);
        if (option == "wwidth") {
            wwidth = std::stoi(line);
        }
        else if (option == "wheigth") {
            wheigth = std::stoi(line);
        }
        else if (option == "g") {
            g = std::stof(line);
        }
        else if (option == "maxabs_vx") {
            maxabs_starting_xvelocity = std::stof(line);
        }
        else if (option == "maxabs_vy") {
            maxabs_starting_yvelocity = std::stof(line);
        }
        else if (option == "radius") {
            radius = std::stof(line);
        }
        else if (option == "count") {
            count = std::stoi(line);
        }
    }

    file.close();
}

configuration configuration::preset::standard()
{
    configuration config;
    config.count = 100000;
    config.radius = 1.0f;

    config.g = 0.4f;
    config.maxabs_starting_xvelocity = 2.0f;
    config.maxabs_starting_yvelocity = 2.0f;

    config.wwidth = 1280;
    config.wheigth = 720;
    return config;
}

configuration configuration::preset::collisions() 
{
    configuration config;
    config.count = 15;
    config.radius = 25.0f;

    config.g = 0.4f;
    config.maxabs_starting_xvelocity = 5.0f;
    config.maxabs_starting_yvelocity = 2.0f;

    config.wwidth = 1280;
    config.wheigth = 720;
    return config;
}
