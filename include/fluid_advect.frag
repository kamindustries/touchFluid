out vec4 fragColor;
uniform float dt;
uniform vec4 diffusionRate;
uniform vec2 res;

int SOURCE = 0;
int VELOCITY = 1;
int OBSTACLE = 2;

vec2 invSize = 1./res;

void main()
{

  float obstacle = texture(sTD2DInputs[OBSTACLE], vUV.st, 0).b;
  if (obstacle > 0) {
      fragColor = vec4(0);
      return;
  }

  vec2 fragCoord = gl_FragCoord.xy;
  vec2 v = texture(sTD2DInputs[VELOCITY], vUV.st).xy;
  vec2 advectedCoord = invSize * (fragCoord - dt * v);
  fragColor = texture(sTD2DInputs[SOURCE], advectedCoord) * diffusionRate;

}
