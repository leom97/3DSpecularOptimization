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

uniform int normal_map;	// controls whether the color will be the normals or not

uniform float Near;
uniform float Far;
// Take as input the depth in NDC, further (linearly) mapped to [0,1], and get back the z value in eye coordinates (camera coordinates)
float retrieve_depth(float z_01)
{
	float z = z_01 * 2.0 - 1.0; // back to NDC 
    return (2.0 * Near * Far) / (Far + Near - z * (Far - Near));	
}

void main()
{    

	if (normal_map==0)
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
		FragColor = vec4(res, 1.0f);
	}
	else{
		FragColor = vec4(normalize(Normal)*0.5f+0.5f, 1.0f);
	}

	// Note: if we're in depth map mode, this will not happen, and the .z component of the fragment will be returned instead
	// So, it is fine to leave this code as is

}