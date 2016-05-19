// Sobel filter
// based on: https://www.shadertoy.com/view/4ss3Dr

uniform float res;
float increment = 1./res;

out vec4 fragColor;

vec3 sample(float x, float y, in vec2 fragCoord)
{
  vec2 uv = fragCoord.xy;
	uv = (uv + vec2(x * increment, y * increment));
	return texture2D(sTD2DInputs[0], uv).xyz;
}

float luminance(vec3 c)
{
	return dot(c, vec3(.2126, .7152, .1));
}

vec3 filter(in vec2 fragCoord)
{
  float sz = 1.;
	vec3 hc =sample(-sz,-sz, fragCoord) *  1. + sample( 0,-sz, fragCoord) *  2.
		 	+sample( sz,-sz, fragCoord) *  1. + sample(-sz, sz, fragCoord) * -2.
		 	+sample( 0, sz, fragCoord) * -2. + sample( sz, sz, fragCoord) * -2.;

    vec3 vc =sample(-sz,-sz, fragCoord) *  1. + sample(-sz, 0, fragCoord) *  2.
		 	+sample(-sz, sz, fragCoord) *  1. + sample( sz,-sz, fragCoord) * -2.
		 	+sample( sz, 0, fragCoord) * -2. + sample( sz, sz, fragCoord) * -2.;

	// return sample(0, 0, fragCoord) * pow(luminance(vc*vc + hc*hc), .2);
  return vec3(pow(luminance(vc*vc + hc*hc), 5.));
}

void main()
{
  vec4 cd = texture2D(sTD2DInputs[0], vUV.st);
	vec3 filter = filter(vUV.st);
  cd.b = filter.b;
  // cd.b = filter.b - cd.b;
  if (cd.b < .5) cd.b = 0.;
  // if (cd.b < .5) cd.b = 0.;
  // cd = clamp(cd, 0., 1.);
  cd.a = 1.;
	fragColor = cd;
}
