#version 330 core

layout(points) in;                    // Accept points as input
layout(triangle_strip, max_vertices = 4) out; // Emit a square

in vec4 vertexColor[];                    // Color passed from the vertex shader
out vec4 fragColor; 

uniform float radius_x;
uniform float radius_y;

void main() {
    vec4 center = gl_in[0].gl_Position; // Center of the square
    fragColor = vertexColor[0];        // Use the input color

    // Emit vertices for a square
    gl_Position = center + vec4(-radius_y, -radius_x, 0.0, 0.0); 
    EmitVertex();
    gl_Position = center + vec4(-radius_y, radius_x, 0.0, 0.0); 
    EmitVertex();
    gl_Position = center + vec4(radius_y,  -radius_x, 0.0, 0.0); 
    EmitVertex();
    gl_Position = center + vec4( radius_y,  radius_x, 0.0, 0.0); 
    EmitVertex();

    EndPrimitive();
}