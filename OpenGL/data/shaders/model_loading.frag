#version 330 core

struct Light{
	vec3 color;
	vec3 wDir;
};

struct Material{
	sampler2D texture_diffuse1;
	sampler2D texture_specular1;
};

out vec4 FragColor;
in vec2 TexCoords;
in vec3 Normal;
in vec3 wPos;

uniform vec3 camPos;
uniform Material material;
uniform Light light;

void main()
{    
   	vec3 normlightdir = normalize(-light.wDir);
	vec3 n = normalize(Normal);

	// diffuse
	vec3 diffuse_sh = max(dot(normlightdir, n),0) * light.color;

	// specular
	float c = 20.0;
	vec3 viewdir = normalize(camPos-wPos);
	vec3 refl = reflect(-normlightdir, n);	// the first vector should point to the fragment
	float spec = pow(max(dot(refl,viewdir),0.0), c);
	vec3 spec_sh = spec * light.color;
	
	vec3 res = vec3(texture(material.texture_diffuse1, TexCoords))*diffuse_sh+vec3(texture(material.texture_specular1, TexCoords))*spec_sh;
//	vec3 res = vec3(0.0f, 0.0f, 1.0f);
	FragColor = vec4(res, 1.0f);
}