#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aCol;

out vec4 vCol;
uniform float offset;

void main()
{
	gl_Position = vec4(aPos.x+offset, -aPos.y, aPos.z, 1.0f);
	vCol = aCol;
}