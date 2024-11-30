
#ifndef _SHADERS_1180779_
#define _SHADERS_1180779_

#include <glad/glad.h>

#include <iostream>

class shader_files {
private:
    std::string vertexShaderSource;
    std::string fragmentShaderSource;
    std::string geometryShaderSource;

public:
    const char* vertexShaderSourceC;
    const char* fragmentShaderSourceC;
    const char* geometryShaderSourceC;

    void loadFromFile(std::string& str, std::string& file);
    shader_files(std::string vertexFile, std::string fragmentFile, std::string geometryShader = "");
    void print();
};

GLuint compileShader(const char* source, GLenum shaderType);
GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader, GLuint geometryShader = -1);

#endif
