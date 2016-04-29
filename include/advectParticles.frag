layout(location=0) out vec4 final_color0;
layout(location=1) out vec4 final_color1;
layout(location=2) out vec4 final_color2;
layout(location=3) out vec4 final_color3;
layout(location=4) out vec4 final_color4;
layout(location=5) out vec4 final_color5;
layout(location=6) out vec4 final_color6;

uniform float res;
uniform float trigger;
uniform float fluidForce;
uniform float frame;
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

float bounds = .99;

///////////////////////////////////////////////////////////
// Ghetto random
///////////////////////////////////////////////////////////
float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}
float fit(float val, float inMin, float inMax, float outMin, float outMax) {
    return ((outMax - outMin) * (val - inMin) / (inMax - inMin)) + outMin;
}

bool checkBorder(vec2 pos) {
  if (pos.x < 1-bounds || pos.x > bounds || pos.y < 1-bounds || pos.y > bounds) {
    return true;
  }
  else return false;
}

bool checkBounds(vec2 pos) {
  if (pos.x < 1-bounds || pos.x > bounds || pos.y < 1-bounds || pos.y > bounds ||
      texture2D(sTD2DInputs[BOUNDARY], pos).b > 0.0)
  {
    return true;
  }
  else return false;
}


///////////////////////////////////////////////////////////
// M A I N
///////////////////////////////////////////////////////////
void main() {
	bool anim = true;

  vec4 color;
	vec3 position, normals, velocity, velocity_old;
	float mass, momentum, life;

  if (trigger > 0.0 || texture2D(sTD2DInputs[COLOR_OLD], vUV.st).a < 0.01) {
    position = vec3(0.0); //initialize position

    // Try not to spawn inside a boundary
    float bnd = checkBounds(vUV.st) ? 1.0 : 0.0;

    int try = 5;
    int i = 0;
    float b = bnd;
    if (bnd > 0.0) {
      while (b > 0. && i < try) {
        float i2 = float(i)/5.;
        vec2 check = vec2(rand(vUV.st+i2), rand(vUV.st*2.4+i2));
        position = vec3(check.x, check.y, 0);

        b = checkBounds(check) ? 1 : 0;
        i++;
      }
    }
    else {
      // Spawn in place with a little randomness
      float px = vUV.s + ((rand(vec2(vUV.s*.01,vUV.t*.2))*2.0)-1.0)*.2;
      float py = vUV.t + ((rand(vec2(vUV.s*.5, vUV.t*.75))*2.0)-1.0)*.2;
      position = vec3(px, py, 0.0);
    }

    position.z = 0.0;

    // vec2 posCoord = (position.xy + vec2(1.0))/2.0;
    vec2 posCoord = position.xy;

    velocity = velocity_old = vec3(0.0);
    velocity = texture2D(sTD2DInputs[VELOCITY_OLD], posCoord).rgb;
    velocity_old = velocity;

    normals = vec3(1.0);
    mass = texture2D(sTD2DInputs[MASS], posCoord).r;
    momentum = texture2D(sTD2DInputs[MOMENTUM], posCoord).r;

    color.rgb = texture2D(sTD2DInputs[COLOR_NEW], posCoord).rgb;
    color.a = 1.0;
    if (checkBounds(posCoord)) color.a = 0.;

    life = texture2D(sTD2DInputs[NOISE], posCoord).r;
    life = fit(life, 0., 1., .5, 1.5);
  }

	else {
    position = texture2D(sTD2DInputs[POS_OLD], vUV.st).rgb;
    normals = texture2D(sTD2DInputs[NRML_OLD], vUV.st).rgb;
    color = texture2D(sTD2DInputs[COLOR_OLD], vUV.st);
    mass = texture2D(sTD2DInputs[MASS_OLD], vUV.st ).r;
		momentum = texture2D(sTD2DInputs[MOMENTUM_OLD], vUV.st ).r;
    life = texture2D(sTD2DInputs[LIFE_OLD], vUV.st).r;

    // vec2 posCoord = (position.xy+vec2(1.0))/2.0;
    vec2 posCoord = position.xy;
    velocity_old = texture2D(sTD2DInputs[VELOCITY_OLD], vUV.st ).rgb;
    velocity = texture2D(sTD2DInputs[VELOCITY], posCoord ).rgb;
    velocity = velocity * (mass * fluidForce ) * .1 + velocity_old * momentum; // how is fluid force related to the *.1?

    // Get the magnitude
    float mag = sqrt(velocity.x*velocity.x + velocity.y*velocity.y);
    float mag_old = sqrt(velocity_old.x*velocity_old.x + velocity_old.y*velocity_old.y);
    float accel = (abs(mag-mag_old))/.1;

    float cd = sqrt(mag*1.);
    cd *= 20;
    cd = clamp(cd, 0., 1.);

    // Set color
    float hue = atan(velocity.x, velocity.y);
    color.rgb = bch2rgb(vec3(cd, sqrt(1-cd), hue));

    // Set alpha
    color.a = 1.;
    // color.a = cd * 2.;
    // color.a = clamp(cd, .4, 1.);
    // color.a = 1.;

    // if (mag < .005) color.a = 0.0; //kills if its too slow... reconsider this

    // Boundary conditions
    float oInc = increment * 1.;
    vec2 pNew = position.xy;
    vec2 bN = vec2(pNew.x, pNew.y + oInc);
    vec2 bS = vec2(pNew.x, pNew.y - oInc);
    vec2 bE = vec2(pNew.x + oInc, pNew.y);
    vec2 bW = vec2(pNew.x - oInc, pNew.y);
    float oN = texture2D(sTD2DInputs[BOUNDARY], bN).b;
    float oS = texture2D(sTD2DInputs[BOUNDARY], bS).b;
    float oE = texture2D(sTD2DInputs[BOUNDARY], bE).b;
    float oW = texture2D(sTD2DInputs[BOUNDARY], bW).b;
    float oC = texture2D(sTD2DInputs[BOUNDARY], pNew).b;

    if (oN > 0.0 || oS > 0.0) {
      velocity.y *= -1;
    }
    if (oE > 0.0 || oW > 0.0) {
      velocity.x *= -1;
    }
    // if (oC > 0.0) color.a = 0.0;
    // if (checkBorder(position.xy)) color.a = 0.0;

    // Basic boundary conditions
		// if (position.x < 1-bounds) {
		// 	position.x = 1-bounds;
		// 	velocity.x *= -1;
		// }
		// if (position.x > bounds) {
		// 	position.x = bounds;
		// 	velocity.x *= -1;
		// }
		// if (position.y < 1-bounds) {
		// 	position.y = 1-bounds;
		// 	velocity.y *= -1;
		// }
		// if (position.y > bounds) {
		// 	position.y = bounds;
		// 	velocity.y *= -1;
		// }

    // Position and normals
    position += velocity;
    normals = normalize(normals + ((velocity - 0.5) / 3.));
    if (checkBounds(position.xy)) color.a = 0.0;

    life -= .0005;
    if (life <= 0.0) color.a = 0.0;

  } //else


	final_color0 = vec4(position.rgb,1);
	final_color1 = vec4(normals,1);
	final_color2 = vec4(color);
	final_color3 = vec4(velocity,1);
	final_color4 = vec4(vec3(mass),1);
	final_color5 = vec4(vec3(momentum),1);
  final_color6 = vec4(vec3(life),1);

} //main
