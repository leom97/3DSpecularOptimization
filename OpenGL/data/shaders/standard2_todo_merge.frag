#version 330 core
out vec4 FragColor;

in VS_OUT {
    vec3 wPos;
    vec3 Normal;
    vec2 TexCoords;
    vec4 FragPosLightSpace;
} fs_in;

struct Material{
	sampler2D texture_diffuse1;
	sampler2D texture_specular1;
};

uniform sampler2D shadowMap;

struct Light{
	vec3 color;
	vec3 wDir;
};

uniform vec3 camPos;
uniform Light light;
uniform Material material;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // calculate bias (based on depth map resolution and slope)
    vec3 normal = normalize(fs_in.Normal);
    vec3 lightDir = normalize(-light.wDir);
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    // check whether current frag pos is in shadow
    // float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth  ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    
    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(projCoords.z > 1.0)
        shadow = 0.0;
        
    return shadow;
}

void main()
{           

    vec3 normlightdir = normalize(-light.wDir);
	vec3 n = normalize(fs_in.Normal);

	// diffuse
	vec3 diffuse_sh = max(dot(normlightdir, n),0) * light.color;

	// specular
	float c = 20.0;
	vec3 viewdir = normalize(camPos-fs_in.wPos);
	vec3 refl = reflect(-normlightdir, n);	// the first vector should point to the fragment
	float spec = pow(max(dot(refl,viewdir),0.0), c);
	vec3 spec_sh = spec * light.color;
	
    float shadow = ShadowCalculation(fs_in.FragPosLightSpace);                      
    vec3 diffuse_alb = vec3(texture(material.texture_diffuse1, fs_in.TexCoords));
    vec3 specular_alb = vec3(texture(material.texture_specular1, fs_in.TexCoords));
	vec3 res = (diffuse_alb*diffuse_sh+specular_alb*spec_sh);
    res = res * (1.0 - shadow);
	FragColor = vec4(res, 1.0f);    

}