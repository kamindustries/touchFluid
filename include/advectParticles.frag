layout(location=0) out vec4 final_color0;
layout(location=1) out vec4 final_color1;
layout(location=2) out vec4 final_color2;
layout(location=3) out vec4 final_color3;
layout(location=4) out vec4 final_color4;
layout(location=5) out vec4 final_color5;
layout(location=6) out vec4 final_color6;

uniform float res;
uniform float trigger;
uniform float Nvis;
uniform float frame;
uniform float dt;
uniform float lifespan;
uniform float spawnMode;
uniform float velColor;
uniform vec3 hsv;
uniform float hueRotation;
uniform float colorMix;
float increment = 1.0/res;

int POS_OLD = 0;
int NRML_OLD = 1;
int COLOR_OLD = 2;
int VELOCITY_OLD = 3;
int MASS_OLD = 4;
int MOMENTUM_OLD = 5;
int VELOCITY = 6;
int MASS = 7;
int MOMENTUM = 8;
int NOISE = 9;
int COLOR_NEW = 10;
int BOUNDARY = 11;
int LIFE_OLD = 12;

float bounds = .98;

highp float rand(vec2 co) {
    // http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
    highp float a = 12.9898;
    highp float b = 78.233;
    highp float c = 43758.5453;
    highp float dt= dot(co.xy ,vec2(a,b));
    highp float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}
float luminance(vec3 c)
{
	return dot(c, vec3(.2126, .7152, .0722));
}

float fit(float val, float inMin, float inMax, float outMin, float outMax) {
    return ((outMax - outMin) * (val - inMin) / (inMax - inMin)) + outMin;
}

bool checkBorder(vec2 pos) {
  if (pos.x < 1.-bounds || pos.x > bounds || pos.y < 1.-bounds || pos.y > bounds) {
    return true;
  }
  else return false;
}

bool checkBounds(vec2 pos) {
  if (pos.x < 1.-bounds || pos.x > bounds || pos.y < 1.-bounds || pos.y > bounds ||
      texture2D(sTD2DInputs[BOUNDARY], pos).b > 0.0)
  {
    return true;
  }
  else return false;
}

bool checkBndCustom(vec2 pos) {
  if (pos.x < 1.-bounds || pos.x > bounds || pos.y < 1.-bounds || pos.y > bounds ||
    texture2D(sTD2DInputs[COLOR_NEW], pos).b < .1)
  {
    return true;
  }
  else return false;
}


///////////////////////////////////////////////////////////
// M A I N
///////////////////////////////////////////////////////////
void main() {
  if (vUV.t < Nvis) { // the % of available particles we see
  	bool anim = true;

    vec4 color;
  	vec3 position, normals, velocity, velocity_old;
  	float mass, momentum, life, life_var;

    bool spawn = trigger > 0. ? true : false;
    float lifeThreshold = 0.;

    if (texture2D(sTD2DInputs[LIFE_OLD], vUV.st).r <= lifeThreshold) spawn = true;

    // SPAWN
    if (spawn) {
      vec2 check;
      position = texture2D(sTD2DInputs[NOISE], vUV.st).rgb;

      // Try not to spawn inside a boundary
      // int bnd = checkBounds(position.st) ? 1 : 0;
      int bnd = checkBndCustom(position.st) ? 1 : 0;
      int try = 40;
      int i = 0;
      int b = bnd;
      if (bnd > 0) {
        while (b > 0 && i < try) {
          float i2 = float(i)/float(try);
          check = vec2(rand(vUV.st+i2), rand(vUV.st+i2+.1));
          position.st = check;

          // b = checkBounds(position.st) ? 1 : 0;
          b = checkBndCustom(position.st) ? 1 : 0;
          i++;
        }
      }

      position.z = 0.0;

      vec2 posCoord = position.xy;

      velocity = velocity_old = vec3(0.0);
      velocity = texture2D(sTD2DInputs[VELOCITY_OLD], posCoord).rgb;
      velocity_old = velocity;

      normals = vec3(1.0);

      vec2 r1 = vec2(rand(vec2(vUV.s,vUV.t+.2)), rand(vec2(vUV.t+.4,vUV.s+.7)));
      vec2 r2 = vec2(rand(vec2(vUV.s+.1,vUV.t+.1)), rand(vec2(vUV.t+.3,vUV.s+.8)));
      vec2 r3 = vec2(rand(vec2(vUV.s+.05,vUV.t+.45)), rand(vec2(vUV.t+.48,vUV.s+.77)));
      mass = texture2D(sTD2DInputs[MASS], r1).r;
      momentum = texture2D(sTD2DInputs[MOMENTUM], r2).r;

      //-------------------
      // C O L O R
      //-------------------
      vec3 velCd;
        velCd.x = 1.4;
        velCd.y = hsv.y;
        // velCd.z = atan(velocity.x, velocity.y);
        float hue = hueRotation;
        velCd.z = hueRotation;
      velCd = bch2rgb(velCd);

      color = texture2D(sTD2DInputs[COLOR_NEW], posCoord);
      color.rgb = mix(color.rgb, velCd, colorMix);
      color.a = 1.;
      //-------------------

      life = 1.;
      life_var = texture2D(sTD2DInputs[NOISE], r3).g;
      life_var = fit(life_var, 0., 1., .5, 1.);

      life_var = life_var * ((1.-lifespan + .01) * .01);
      // life_var = life_var * (.01 - (lifespan * .009999));
      // life_var = 0.;
      // life_var = .005;
    } //SPAWN

  	else {
      position = texture2D(sTD2DInputs[POS_OLD], vUV.st).rgb;
      normals = texture2D(sTD2DInputs[NRML_OLD], vUV.st).rgb;
      mass = texture2D(sTD2DInputs[MASS_OLD], vUV.st ).r;
  		momentum = texture2D(sTD2DInputs[MOMENTUM_OLD], vUV.st ).r;
      life = texture2D(sTD2DInputs[LIFE_OLD], vUV.st).r;
      life_var = texture2D(sTD2DInputs[LIFE_OLD], vUV.st).g;

      vec2 posCoord = position.xy;
      velocity_old = texture2D(sTD2DInputs[VELOCITY_OLD], vUV.st ).rgb;
      velocity = texture2D(sTD2DInputs[VELOCITY], posCoord ).rgb;
      velocity = velocity * (mass * 1. ) * dt + velocity_old * momentum; // how is fluid force related to the *.1?

      // Get the magnitude
      float velAmp = velocity.x*velocity.x + velocity.y*velocity.y;
      float mag = sqrt(velAmp);
      float mag_old = sqrt(velocity_old.x*velocity_old.x + velocity_old.y*velocity_old.y);
      float accel = (abs(mag-mag_old))/.1;

      // Set color
      color = texture2D(sTD2DInputs[COLOR_OLD], vUV.st);

      // color.rgb = clamp(color.rgb, .1, 1.);
      // if (luminance(color.rgb) < .5) color.rgb += sqrt(mag * .5) * velColor;

      // mixing velocity
      // if knob is < .5, just offset hue
      // if knob is > .5, mult against magnitude
      float velLumMix = (velColor > .5) ? fit(velColor, .5, 1., 0., 1.) : 0.;
      float velHueMix = (velColor < .5) ? fit(velColor, 0., 1., 0., 1.) : fit(velColor, .5, 1., 1., 0.);

      color.rgb = rgb2bch(color.rgb);
        color.x = mix(1.4 * hsv.z, mag*200., velLumMix);
        color.z = mix(color.z, mag*20., velHueMix);
      color.rgb = bch2rgb(color.rgb);




      position += velocity;

      // Boundary conditions
      // WORKING: borders
      // NOT WORKING: obstacles
      float oInc = increment * 1.;
      vec2 pNew = position.xy;
      vec2 bN = vec2(pNew.x, pNew.y + oInc);
      vec2 bS = vec2(pNew.x, pNew.y - oInc);
      vec2 bE = vec2(pNew.x + oInc, pNew.y);
      vec2 bW = vec2(pNew.x - oInc, pNew.y);
      // float oN = checkBorder(bN) ? 1. : texture2D(sTD2DInputs[BOUNDARY], bN).b;
      // float oS = texture2D(sTD2DInputs[BOUNDARY], bS).b;
      // float oE = texture2D(sTD2DInputs[BOUNDARY], bE).b;
      // float oW = texture2D(sTD2DInputs[BOUNDARY], bW).b;
      // float oC = texture2D(sTD2DInputs[BOUNDARY], pNew).b;
      float oN = checkBorder(bN) ? 1. : 0.;
      float oS = checkBorder(bS) ? 1. : 0.;
      float oE = checkBorder(bE) ? 1. : 0.;
      float oW = checkBorder(bW) ? 1. : 0.;
      float oC = checkBounds(pNew) ? 1. : 0.;

      if (oW > 0) {
  			position.x = 1-bounds+increment;
  			velocity.x *= -1.;
  		}
      if (oS > 0) {
        position.y = 1-bounds+increment;
        velocity.y *= -1.;
      }

  		if (oE > 0) {
  			position.x = bounds-increment;
  			velocity.x *= -1.;
  		}
  		if (oN > 0) {
  			position.y = bounds-increment;
  			velocity.y *= -1.;
  		}
      if (oC > 0) {
        // life = 0.0;
      }
      // Basic boundary conditions
  		// if (position.x < 1-bounds) {
  		// 	position.x = 1-bounds;
  		// 	velocity.x *= -1.;
  		// }
      // if (position.y < 1-bounds) {
      //   position.y = 1-bounds;
      //   velocity.y *= -1.;
      // }
      //
  		// if (position.x > bounds) {
  		// 	position.x = bounds;
  		// 	velocity.x *= -1.;
  		// }
  		// if (position.y > bounds) {
  		// 	position.y = bounds;
  		// 	velocity.y *= -1.;
  		// }

      // Position and normals
      normals = normalize(normals + ((velocity - 0.5) / 3.));

      // life -= .0005;
      life -= life_var;

    } //else


  	final_color0 = vec4(position.rgb,1);
  	final_color1 = vec4(normals,1);
  	final_color2 = vec4(color);
  	final_color3 = vec4(velocity,1);
  	final_color4 = vec4(vec3(mass),1);
  	final_color5 = vec4(vec3(momentum),1);
    final_color6 = vec4(vec3(life, life_var, 0.),1);
  }//Nvis
} //main
