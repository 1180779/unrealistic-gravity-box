#version 330 core
layout(location = 0) in float xPos;     // x position
layout(location = 1) in float yPos;     // y position
layout(location = 2) in vec4 aColor;    // r, g, b, a (color data)

uniform float screenWidth;  // Screen width passed from CPU
uniform float screenHeight; // Screen height passed from CPU

out vec4 vertexColor; // Pass color to the fragment shader

void main() {
    // Normalize xPos and yPos to [-1, 1]
    float normX = (xPos / screenWidth) * 2.0 - 1.0;
    float normY = (yPos / screenHeight) * 2.0 - 1.0;

    gl_Position = vec4(normX, normY, 0.0, 1.0); // Set position in clip space
    vertexColor = aColor;               // Pass color to the next stage
}