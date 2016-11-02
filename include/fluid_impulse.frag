out vec4 fragColor;

uniform float dt;

int IN = 0;
int IMPULSE = 1;

const float smallf = 0.0000001f;

void main()
{
  ivec2 TC = ivec2(gl_FragCoord.xy);

  vec4 x = texelFetch(sTD2DInputs[IN], TC, 0);
  vec4 impulse = texelFetch(sTD2DInputs[IMPULSE], TC, 0);
  x += impulse;
  fragColor = x;
  // float d = distance(vec2(1280/2,720/3), gl_FragCoord.xy);
  // float Radius = 100.;
  // if (d < Radius) {
  //   float a = (Radius - d) * 0.5;
  //   a = min(a, 1.0);
  //   fragColor = vec4(vec3(1.), a);
  // } else {
  //     fragColor = x;
  // }
}
