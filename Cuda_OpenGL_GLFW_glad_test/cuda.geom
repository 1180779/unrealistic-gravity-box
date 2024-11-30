#version 330 core

layout(points) in;                    // Accept points as input
layout(triangle_strip, max_vertices = 4) out; // Emit a square

in vec4 pointColor[];                 // Color from vertex shader
out vec4 vertexColor;                 // Pass color to fragment shader

uniform float radius; // Radius of the square (in normalized device coordinates)

void main() {
    vec4 center = gl_in[0].gl_Position; // Center of the square
    vertexColor = pointColor[0];        // Use the input color

    // Emit vertices for a square
    gl_Position = center + vec4(-radius, -radius, 0.0, 0.0); EmitVertex();
    gl_Position = center + vec4( radius, -radius, 0.0, 0.0); EmitVertex();
    gl_Position = center + vec4(-radius,  radius, 0.0, 0.0); EmitVertex();
    gl_Position = center + vec4( radius,  radius, 0.0, 0.0); EmitVertex();
    EndPrimitive();
}