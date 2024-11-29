#pragma once

#include <glad/glad.h>

GLuint compileShader(const char* source, GLenum shaderType);
GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader);
