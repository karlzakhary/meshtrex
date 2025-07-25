#version 450

layout(location = 0) out vec3 fragColor;

const vec3 vertices[] =
{
vec3(0, 0.5, 0),
vec3(0.5, -0.5, 0),
vec3(-0.5, -0.5, 0),
};

vec3 colors[3] = vec3[](
vec3(1.0, 0.0, 0.0),
vec3(0.0, 1.0, 0.0),
vec3(0.0, 0.0, 1.0)
);

void main() {
    gl_Position = vec4(vertices[gl_VertexIndex], 1.0);
    fragColor = colors[gl_VertexIndex];
}
