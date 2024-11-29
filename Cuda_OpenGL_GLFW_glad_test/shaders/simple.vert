#version 330 core
layout (location = 0) in vec2 aPos;
void main()
{
    float r = 0.05;
    vec2 v1 = vec2(aPos.x - r, aPos.y - r);
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
};