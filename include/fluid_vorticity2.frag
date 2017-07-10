out vec4 fragColor;

uniform float halfReciprocalGridScale; // 0.5 / gridScale
uniform vec2 vorticityConfinementScale;
uniform float dt;

const int VELOCITY = 0;
const int VORTICITY = 1;

void main()
{
	ivec2 TC = ivec2(gl_FragCoord.xy);
	
	vec2 vortCenter = texelFetch(sTD2DInputs[VORTICITY], TC, 0).xy;
    vec2 vortLeft = texelFetch(sTD2DInputs[VORTICITY], TC-ivec2(-1,0), 0).xy;
    vec2 vortRight = texelFetch(sTD2DInputs[VORTICITY], TC-ivec2(1,0), 0).xy;
    vec2 vortTop = texelFetch(sTD2DInputs[VORTICITY], TC-ivec2(0,1), 0).xy;
    vec2 vortBottom = texelFetch(sTD2DInputs[VORTICITY], TC-ivec2(0,-1), 0).xy;
    
    vec2 force = halfReciprocalGridScale * vec2(abs(vortTop.x) - abs(vortBottom.x), abs(vortRight.x) - abs(vortLeft.x));
    
    // Safe normalize
	float epsilon = 2.4414e-4; // 2^-12
	float magnitudeSquared = max(epsilon, dot(force, force));
	force *= inversesqrt(magnitudeSquared);
	force *= vorticityConfinementScale * vortCenter.xy * vec2(1, -1);

	vec4 velocityNew = texelFetch(sTD2DInputs[VELOCITY], TC, 0);
	fragColor = velocityNew + dt * vec4(force, 0, 0);

}