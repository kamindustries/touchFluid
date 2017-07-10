out vec4 fragColor;

uniform float halfReciprocalGridScale; // 0.5 / gridScale


void main()
{
	ivec2 TC = ivec2(gl_FragCoord.xy);
	
    vec2 vLeft = texelFetch(sTD2DInputs[0], TC-ivec2(-1,0), 0).xy;
    vec2 vRight = texelFetch(sTD2DInputs[0], TC-ivec2(1,0), 0).xy;
    vec2 vTop = texelFetch(sTD2DInputs[0], TC-ivec2(0,1), 0).xy;
    vec2 vBottom = texelFetch(sTD2DInputs[0], TC-ivec2(0,-1), 0).xy;
    
    fragColor = vec4(halfReciprocalGridScale * ((vRight.y - vLeft.y) - (vTop.x - vBottom.x)));
}


