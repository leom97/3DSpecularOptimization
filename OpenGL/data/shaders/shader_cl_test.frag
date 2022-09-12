#version 330 core

struct Material{
	vec3 ambient_color;
	vec3 diffuse_color;
	vec3 specular_color;
	float shininess;
};

struct Light{
	vec3 color;
	vec3 wPos;
};

uniform vec3 objCol;
uniform vec3 camPos;	// in world coordinates
uniform Material material;
uniform Light light;

out vec4 col;
in vec3 Normal;
in vec3 wPos;

void main()
{
	// ambient
	vec3 amb_sh = light.color;

	// diffuse
	vec3 normLightDir = normalize(light.wPos-wPos);
	vec3 diffuse_sh = max(dot(normLightDir, Normal),0) * light.color;

	// specular
	vec3 viewDir = normalize(camPos-wPos);
	vec3 refl = reflect(-normLightDir, Normal);	// the first vector should point to the fragment
	float spec = pow(max(dot(refl,viewDir),0.0), material.shininess);
	vec3 spec_sh = spec * light.color;
	
	vec3 res = material.ambient_color*amb_sh + material.diffuse_color*diffuse_sh+material.specular_color*spec_sh;
	col = vec4(res, 1.0f);
}