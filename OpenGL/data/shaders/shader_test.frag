#version 330 core
in vec4 vCol;
out vec4 FragColor;
uniform float amplitude;
void main()
{
   FragColor = amplitude * vCol;
}