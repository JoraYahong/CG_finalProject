#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 surfaceNormal;


uniform sampler2D texture_diffuse1;
uniform sampler2D specular;


void main()
{    
	
    FragColor = texture(texture_diffuse1, TexCoords);
	
}