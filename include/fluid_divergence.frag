out vec4 fragColor;

uniform float halfInverseCell;

int VELOCITY = 0;
int OBSTACLE = 1;

void main()
{
    ivec2 T = ivec2(gl_FragCoord.xy);

    // Find neighboring velocities:
    vec2 vN = texelFetchOffset(sTD2DInputs[VELOCITY], T, 0, ivec2(0, 1)).xy;
    vec2 vS = texelFetchOffset(sTD2DInputs[VELOCITY], T, 0, ivec2(0, -1)).xy;
    vec2 vE = texelFetchOffset(sTD2DInputs[VELOCITY], T, 0, ivec2(1, 0)).xy;
    vec2 vW = texelFetchOffset(sTD2DInputs[VELOCITY], T, 0, ivec2(-1, 0)).xy;

    // Find neighboring obstacles:
    vec3 oN = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(0, 1)).xyz;
    vec3 oS = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(0, -1)).xyz;
    vec3 oE = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(1, 0)).xyz;
    vec3 oW = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(-1, 0)).xyz;

    // Use obstacle velocities for solid cells:
    if (oN.b > 0) vN = oN.xy;
    if (oS.b > 0) vS = oS.xy;
    if (oE.b > 0) vE = oE.xy;
    if (oW.b > 0) vW = oW.xy;

    fragColor = vec4( halfInverseCell*halfInverseCell * (vE.x - vW.x + vN.y - vS.y) ) ;

}
