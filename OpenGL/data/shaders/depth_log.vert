#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;
out vec3 Normal;
out vec3 wPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float Near;
uniform float Far;

varying float zr;

// Take the matrix A, return A(i,j) (i-th row, j-th column)
float el(mat4 A, int i, int j)
{
    return A[j][i]; // because of the column-major ordering: https://stackoverflow.com/questions/13633395/how-do-you-access-the-individual-elements-of-a-glsl-mat4
}

void main()
{
    TexCoords = aTexCoords;  
    vec4 eye_coords = view * model * vec4(aPos, 1.0);
    gl_Position = projection * eye_coords;
    zr = gl_Position.z;

    gl_Position.z = 2.0*log(gl_Position.w/Near)/log(Far/Near) - 1; 
    gl_Position.z *= gl_Position.w;

    Normal = mat3(transpose(inverse(model))) * aNormal;
    wPos = vec3(model * vec4(aPos, 1.0f));
}