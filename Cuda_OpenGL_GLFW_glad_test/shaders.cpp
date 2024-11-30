
#include "shaders.hpp"

#include <iostream>

// loading shaders from files
#include <string>
#include <fstream>
#include <streambuf>

void shader_files::loadFromFile(std::string& str, std::string& file)
{
    std::ifstream t(file);

    t.seekg(0, std::ios::end);
    vertexShaderSource.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());
}

shader_files::shader_files(std::string vertexFile, std::string fragmentFile, std::string geometryShader)
{
    loadFromFile(vertexShaderSource, vertexFile);
    vertexShaderSourceC = vertexShaderSource.c_str();

    loadFromFile(fragmentShaderSource, fragmentFile);
    fragmentShaderSourceC = fragmentShaderSource.c_str();

    geometryShaderSourceC = NULL;
    if (geometryShader != "") {
        loadFromFile(geometryShaderSource, geometryShader);
        geometryShaderSourceC = geometryShaderSource.c_str();
    }
}

void shader_files::print()
{
    std::cout << "vertex shader: \"\n";
    std::cout << vertexShaderSourceC << "\"\n\n";

    std::cout << "fragment shader: \"\n";
    std::cout << fragmentShaderSourceC << "\"\n\n";

    if (geometryShaderSourceC != NULL) {
        std::cout << "geometry shader: \"\n";
        std::cout << geometryShaderSourceC << "\"\n\n";
    }
}



GLuint compileShader(const char* source, GLenum shaderType) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Check for compilation errors
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader Compilation Failed:\n" << infoLog << std::endl;
    }

    return shader;
}

GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader, GLuint geometryShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    if (geometryShader != -1)
        glAttachShader(program, geometryShader);
    glLinkProgram(program);

    // Check for linking errors
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Program Linking Failed:\n" << infoLog << std::endl;
    }

    return program;
}