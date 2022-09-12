#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D depthMap;
uniform float near_plane;
uniform float far_plane;
uniform int reverse;

// required when using a perspective projection matrix
float LinearizeDepth(float depth, int reverse)  // depth is gl_FragCoord.z
{   
    if (reverse==0)
    {
        float z = depth * 2.0 - 1.0; // Back to NDC 
        return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));    // Note: there should be a minus sign to get the real eye depth, but it would be <0 and I couldn't color it	
//        return near_plane * exp(depth * log(far_plane/near_plane));
    }
    else
    {
        float z = depth; // Back to NDC 
        return (near_plane * far_plane) / (near_plane + z * (far_plane - near_plane));
    }
}

void main()
{             
    float depthValue = texture(depthMap, TexCoords).r;
    FragColor = vec4(vec3(LinearizeDepth(depthValue, reverse) / far_plane), 1.0); // perspective
//    FragColor = vec4(vec3(depthValue), 1.0); // orthographic
}