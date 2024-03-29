out vec4 fragColor;

uniform float halfInverseCell;

int VELOCITY = 0;
int OBSTACLE_N = 1;

void main()
{
    ivec2 T = ivec2(gl_FragCoord.xy);

    vec4 oN = texelFetch(sTD2DInputs[OBSTACLE_N], T, 0);

    ivec2 offsetN = ivec2(0,1);
    ivec2 offsetS = ivec2(0,-1);
    ivec2 offsetE = ivec2(1,0);
    ivec2 offsetW = ivec2(-1,0);

    vec2 multN = vec2(1.);
    vec2 multS = vec2(1.);
    vec2 multE = vec2(1.);
    vec2 multW = vec2(1.);

    if (oN.x > 0.) {
        offsetN.y = -1; 
        multN.y = -1.;
    } 
    else if (oN.y > 0.) {
        offsetS.y = 1; 
        multS.y = -1.;
    }
    if (oN.z > 0.) {
        offsetE.x = -1; 
        multE.x = -1.;
    }
    else if (oN.w > 0.) {
        offsetW.x = 1; 
        multW.x = -1.;
    }

    vec2 vN = texelFetch(sTD2DInputs[VELOCITY], T + offsetN, 0).xy * multN;
    vec2 vS = texelFetch(sTD2DInputs[VELOCITY], T + offsetS, 0).xy * multS;
    vec2 vE = texelFetch(sTD2DInputs[VELOCITY], T + offsetE, 0).xy * multE;
    vec2 vW = texelFetch(sTD2DInputs[VELOCITY], T + offsetW, 0).xy * multW;    

    fragColor = vec4( halfInverseCell*halfInverseCell * ((vE.x - vW.x) + (vN.y - vS.y)) );

}
