out vec4 fragColor;

uniform float gradientScale;

int VELOCITY = 0;
int PRESSURE = 1;
int OBSTACLE = 2;
void main()
{
    ivec2 T = ivec2(gl_FragCoord.xy);

    vec3 oC = texelFetch(sTD2DInputs[OBSTACLE], T, 0).xyz;
    if (oC.b > 0) {
        fragColor = vec4(oC.xy, 0., 0.);
        return;
    }

    // Find neighboring pressure:
    float pN = texelFetchOffset(sTD2DInputs[PRESSURE], T, 0, ivec2(0, 1)).r;
    float pS = texelFetchOffset(sTD2DInputs[PRESSURE], T, 0, ivec2(0, -1)).r;
    float pE = texelFetchOffset(sTD2DInputs[PRESSURE], T, 0, ivec2(1, 0)).r;
    float pW = texelFetchOffset(sTD2DInputs[PRESSURE], T, 0, ivec2(-1, 0)).r;
    float pC = texelFetch(sTD2DInputs[PRESSURE], T, 0).r;

    // Find neighboring obstacles:
    vec3 oN = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(0, 1)).xyz;
    vec3 oS = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(0, -1)).xyz;
    vec3 oE = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(1, 0)).xyz;
    vec3 oW = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(-1, 0)).xyz;

    // Use center pressure for solid cells:
    vec2 obstV = vec2(0);
    vec2 vMask = vec2(1);

    if (oN.b > 0) { pN = pC; obstV.y = oN.y; vMask.y = 0; }
    if (oS.b > 0) { pS = pC; obstV.y = oS.y; vMask.y = 0; }
    if (oE.b > 0) { pE = pC; obstV.x = oE.x; vMask.x = 0; }
    if (oW.b > 0) { pW = pC; obstV.x = oW.x; vMask.x = 0; }

    vec2 oldV = texelFetch(sTD2DInputs[VELOCITY], T, 0).xy;
    vec2 grad = vec2(pE - pW, pN - pS) * gradientScale * 1;
    vec2 newV = oldV - grad;
    fragColor = vec4((vMask * newV) + obstV, 0., 0.);
}
