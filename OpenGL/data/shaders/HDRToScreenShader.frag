#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D HDRTexture;

void main()
{             
    vec3 hdrColor = texture(HDRTexture, TexCoords).rgb;
    // reinhard (the most basic, no need for fancy stuff)
    vec3 result = hdrColor / (hdrColor + vec3(1.0));    // i.e. remap the color to [0, 1] for correct display
    FragColor = vec4(result, 1.0);
}