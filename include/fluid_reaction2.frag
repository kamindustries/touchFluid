layout(location=0) out vec4 fragColor0;
layout(location=1) out vec4 fragColor1;

uniform vec2 res;
uniform vec2 invRes;
uniform float reset;
uniform float absTime;
uniform int CHEM = 0;
uniform vec3 e = vec3(1, 0, -1); //Used to move to adjoining pixels

// Reaction Diffusion - 2 Pass
// https://www.shadertoy.com/view/XsG3z1
// ---------------------------
//
// Expansive Reaction-Diffusion - Flexi
// https://www.shadertoy.com/view/4dcGW2
//
// Gray-Scott diffusion - knighty
// https://www.shadertoy.com/view/MdVGRh
//
// To begin, sprinkle the buffer with some initial noise on the first few frames (Sometimes, the
// first frame gets skipped, so you do a few more).
//
// During the buffer loop pass, determine the reaction diffusion value using a combination of the
// value stored in the buffer's "X" channel, and a the blurred value - stored in the "Y" channel
// (You can see how that's done in the code below). Blur the value from the "X" channel (the old
// reaction diffusion value) and store it in "Y", then store the new (reaction diffusion) value
// in "X." Display either the "X" value  or "Y" buffer value in the "Image" tab, add some window
// dressing, then repeat the process. Simple... Slightly confusing when I try to explain it, but
// trust me, it's simple. :)

// Cheap vec3 to vec3 hash. Works well enough, but there are other ways.
vec3 hash33(in vec2 p){
    float n = sin(dot(p, vec2(41, 289)));
    return fract(vec3(2097152, 262144, 32768)*n);
}

// Serves no other purpose than to save having to write this out all the time. I could write a
// "define," but I'm pretty sure this'll be inlined.
vec4 tx(in vec2 p){ return texture(sTD2DInputs[CHEM], p); }

// Weighted blur function. Pretty standard.
float blur(in vec2 p)
{
  // Used to move to adjoining pixels. - uv + vec2(-1, 1)*px, uv + vec2(1, 0)*px, etc.
  // vec3 e = vec3(1, 0, -1);
  vec2 px = 1./res;

  // Weighted 3x3 blur, or a cheap and nasty Gaussian blur approximation.
	float dx = 0.0;
  // Four corners. Those receive the least weight.
	dx += tx(p + e.xx*px ).x + tx(p + e.xz*px ).x + tx(p + e.zx*px ).x + tx(p + e.zz*px ).x;
  // Four sides, which are given a little more weight.
  dx += (tx(p + e.xy*px ).x + tx(p + e.yx*px ).x + tx(p + e.yz*px ).x + tx(p + e.zy*px ).x)*2.;
	// The center pixel, which we're giving the most weight to, as you'd expect.
	dx += tx(p + e.yy*px ).x*4.;
  // Normalizing.
  return dx/16.;
}


void main()
{

	vec2 uv = vUV.st; // Screen coordinates. Range: [0, 1]
  vec2 pw = invRes; // Relative pixel width. Used for neighboring pixels, etc.

	float avgReactDiff = blur(uv);

	// The noise value. Because the result is blurred, we can get away with plain old static noise.
  // However, smooth noise, and various kinds of noise textures will work, too.
  vec3 noise = hash33(uv + vec2(53, 43)*absTime*.01)*.6 + .2;

  // Used to move to adjoining pixels. - uv + vec2(-1, 1)*px, uv + vec2(1, 0)*px, etc.
  // vec3 e = vec3(1, 0, -1);

  // Gradient epsilon value. The "1.5" figure was trial and error, but was based on the 3x3 blur radius.
  vec2 pwr = pw*1.5;

  // Use the blurred pixels (stored in the Y-Channel) to obtain the gradient. I haven't put too much
  // thought into this, but the gradient of a pixel on a blurred pixel grid (average neighbors), would
  // be analogous to a Laplacian operator on a 2D discreet grid. Laplacians tend to be used to describe
  // chemical flow, so... Sounds good, anyway. :)
  //
  // Seriously, though, take a look at the formula for the reacion-diffusion process, and you'll see
  // that the following few lines are simply putting it into effect.

  // Gradient of the blurred pixels from the previous frame.
	vec2 lap = vec2(tx(uv + e.xy*pwr).y - tx(uv - e.xy*pwr).y, tx(uv + e.yx*pwr).y - tx(uv - e.yx*pwr).y);

  // Add some diffusive expansion, scaled down to the order of a pixel width.
  uv = uv + lap*pw*3.0;

  // Stochastic decay. Ie: A differention equation, influenced by noise.
  // You need the decay, otherwise things would keep increasing, which in this case means a white screen.
  // float newReactDiff = tx(uv).x + (noise.z - 0.5)*0.0025 - 0.002;
  float newReactDiff = tx(uv).x + (noise.z - 0.5)*0.01;

  // Reaction-diffusion.
  // newReactDiff += dot(tx(uv + (noise.xy-0.5)*pw).xy, vec2(1, -1))*0.145;
	newReactDiff += dot(tx(uv + (noise.xy-0.5)*pw).xy, vec2(1, -1))*0.1;


  // Storing the reaction diffusion value in the X channel, and avgReactDiff (the blurred pixel value)
  // in the Y channel. However, for the first few frames, we add some noise. Normally, one frame would
  // be enough, but for some weird reason, it doesn't always get stored on the very first frame.
  if (reset > 0) fragColor0 = vec4(noise.xy, 0., 0.);
  else fragColor0.xy = clamp(vec2(newReactDiff, avgReactDiff/.98), 0., 1.);

  fragColor1 = 1. - texture(sTD2DInputs[CHEM], vUV.st).wyyw;

}

// void mainImage(out vec4 fragColor, in vec2 fragCoord){
//     // The screen coordinates.
//     vec2 uv = fragCoord/iResolution.xy;
//
//     // Read in the blurred pixel value. There's no rule that says you can't read in the
//     // value in the "X" channel, but blurred stuff is easier to bump, that's all.
//     float c = 1. - texture2D(iChannel0, uv).y;
//     // Reading in the same at a slightly offsetted position. The difference between
//     // "c2" and "c" is used to provide the highlighting.
//     float c2 = 1. - texture2D(iChannel0, uv + .5/iResolution.xy).y;
//
//     // Color the pixel by mixing two colors in a sinusoidal kind of pattern.
//     float pattern = -cos(uv.x*0.75*3.14159-0.9)*cos(uv.y*1.5*3.14159-0.75)*0.5 + 0.5;
//     // Blue and gold
//     vec3 col = vec3(c*1.5, pow(c, 2.25), pow(c, 6.));
//     col = mix(col, col.zyx, clamp(pattern-.2, 0., 1.) );
//
//     // Extra color variations.
//     //vec3 col = mix(vec3(c*1.2, pow(c, 8.), pow(c, 2.)), vec3(c*1.3, pow(c, 2.), pow(c, 10.)), pattern );
// 	  //vec3 col = mix(vec3(c*1.3, c*c, pow(c, 10.)), vec3(c*c*c, c*sqrt(c), c), pattern );
//
//     // Adding the highlighting. Not as nice as bump mapping, but still pretty effective.
//     col += vec3(.6, .85, 1.)*max(c2*c2 - c*c, 0.)*12.;
//
//     // Apply a vignette and increase the brightness for that fake spotlight effect.
//     col *= pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y) , .125)*1.15;
//     fragColor = vec4(min(col, 1.), 1.);
// }
