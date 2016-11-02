out vec4 fragColor;

uniform float alpha;
uniform float inverseBeta;

int PRESSURE = 0;
int DIVERGENCE = 1;
int OBSTACLE = 2;

void main()
{
    ivec2 T = ivec2(gl_FragCoord.xy);

    // Find neighboring pressure:
    vec4 pN = texelFetchOffset(sTD2DInputs[PRESSURE], T, 0, ivec2(0, 1));
    vec4 pS = texelFetchOffset(sTD2DInputs[PRESSURE], T, 0, ivec2(0, -1));
    vec4 pE = texelFetchOffset(sTD2DInputs[PRESSURE], T, 0, ivec2(1, 0));
    vec4 pW = texelFetchOffset(sTD2DInputs[PRESSURE], T, 0, ivec2(-1, 0));
    vec4 pC = texelFetch(sTD2DInputs[PRESSURE], T, 0);

    // Find neighboring obstacles:
    vec3 oN = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(0, 1)).xyz;
    vec3 oS = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(0, -1)).xyz;
    vec3 oE = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(1, 0)).xyz;
    vec3 oW = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(-1, 0)).xyz;

    // Use center pressure for solid cells:
    if (oN.b > 0) pN = pC;
    if (oS.b > 0) pS = pC;
    if (oE.b > 0) pE = pC;
    if (oW.b > 0) pW = pC;

    vec4 bC = texelFetch(sTD2DInputs[DIVERGENCE], T, 0);
    float a = -1.25*1.25;
    fragColor = (pW + pE + pS + pN + alpha * bC) * inverseBeta;
}
