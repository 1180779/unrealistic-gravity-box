#version 330 core

layout(points) in;                    // Accept points as input
layout(triangle_strip, max_vertices = 32) out;

in vec4 vertexColor[];                    // Color passed from the vertex shader
out vec4 fragColor; 

uniform float radius_x;
uniform float radius_y;

const int NUM_SEGMENTS = 15; // Number of segments to approximate the circle

void main() {
    float angleIncrement = 2.0 * 3.14159265 / float(NUM_SEGMENTS);
    vec4 center = gl_in[0].gl_Position; 
    fragColor = vertexColor[0];        // Use the input color

    // Generate circle perimeter vertices
    for (int i = 0; i <= NUM_SEGMENTS; i++) {
        float angle = angleIncrement * float(i);
        vec2 offset = vec2(cos(angle) * radius_y, sin(angle) * radius_x);
        vec2 pos = vec2(center) + offset;

        if(i % 3 == 0) {
            gl_Position = vec4(vec2(center), 0.0, 1.0);
            EmitVertex();
        }

        gl_Position = vec4(pos, 0.0, 1.0);
        EmitVertex();
    }

    EndPrimitive();
}