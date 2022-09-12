#version 330 core

struct Light{
	float intensity;
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
uniform int albedo_mode;	// controls whether the color will be the normals or not

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

		// shading
		float shad = max(dot(normlightdir * light.intensity, n),0);

		// specular
		float c = 200.0;
		vec3 viewdir = normalize(camPos-wPos);
		vec3 refl = reflect(-normlightdir, n);	// the first vector should point to the fragment
		vec3 h = normalize(viewdir+normlightdir);

		float spec = (c+2)*pow(max(dot(h,n),0), c);	// Note, this will complain, because of definition of powers of negative numbers. I clamp to get 0 and not nan
		
		vec3 albedo_diff = vec3(texture(material.texture_diffuse1, TexCoords));
		vec3 albedo_spec = vec3(texture(material.texture_specular1, TexCoords));
		vec3 res = (albedo_diff + spec * albedo_spec)*shad;

		if (albedo_mode==0)	FragColor = vec4(res, 1.0f);
		else if (albedo_mode == 1) FragColor = vec4(albedo_diff, 1.0f);
		else FragColor = vec4(albedo_spec, 1.0f);
	}
	else{
		FragColor = vec4(normalize(Normal)*0.5f+0.5f, 1.0f);
	}

	// Note: if we're in depth map mode, this will not happen, and the .z component of the fragment will be returned instead
	// So, it is fine to leave this code as is

}