out vec4 fragColor;

uniform vec2 res;
uniform vec2 invRes;
uniform float dt;
uniform vec2 xLen;
uniform vec2 diffRateAB;

int CHEM = 0;

void main()
{
    ivec2 T = ivec2(gl_FragCoord.xy);

    // Find neighboring velocities:
    vec4 chem = texelFetch(sTD2DInputs[CHEM], T, 0);


    float xLength = gl_FragCoord.x/xLen.x;
		float yLength = gl_FragCoord.y/xLen.y;
		float dx = xLength/gl_FragCoord.x;
		float dy = yLength/gl_FragCoord.y;
		vec2 alpha = diffRateAB * dt / (dx*dy);

		vec4 lap = (-4.0f * texelFetch(sTD2DInputs[CHEM], T, 0) +
                        texelFetch(sTD2DInputs[CHEM], ivec2(T.x+1,T.y), 0) +
                        texelFetch(sTD2DInputs[CHEM], ivec2(T.x-1,T.y), 0) +
                        texelFetch(sTD2DInputs[CHEM], ivec2(T.x,T.y+1), 0) +
                        texelFetch(sTD2DInputs[CHEM], ivec2(T.x,T.y-1), 0) );
    lap.xy *= alpha;
    fragColor = lap;
}
