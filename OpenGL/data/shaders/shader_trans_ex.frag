#version 330 core

out vec4 col;
in vec3 ourCol;
in vec2 ourTexCoord;

uniform sampler2D ourTexture;
uniform sampler2D ourTexture2;

uniform float darkening;

void main()
{
//	float darkening =4 *  sqrt(pow(ourTexCoord.x-.5f,2)+pow(ourTexCoord.y-.5f,2));
//	col = mix(texture(ourTexture, ourTexCoord), texture(ourTexture2, vec2(ourTexCoord.x,1-ourTexCoord.y)), darkening) * vec4(1.0f, 0.0f, 0.0f, 1.0f);	 Lazily using the same texture coordinates
	col = vec4(ourCol, 1.0);
}