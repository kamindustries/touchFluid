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
uniform float dt;
uniform vec2 mousePos;
uniform vec3 mb;
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
    texture2D(sTD2DInputs[COLOR_NEW], pos).b < .01)
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
	float mass, momentum, life, life_var;

  bool spawn = trigger > 0. ? true : false;

  bool mouseTrig = (mb.x > 0. && mb.y > 0.) ? true : false;
  mouseTrig = false; //turn it off for now
  float lifeT = 0.;
  if (mouseTrig) lifeT = .3;
  if (texture2D(sTD2DInputs[LIFE_OLD], vUV.st).r <= lifeT) spawn = true;

  // SPAWN
  if (spawn) {

    vec2 check = vec2(rand(vUV.st+1.2), rand(vUV.st*2.4+1.2));
    position = texture2D(sTD2DInputs[NOISE], vUV.st).rgb;
    if (mouseTrig) {
      position.xy = mousePos.xy;
      vec2 rnd = vec2(rand(vec2(vUV.s,vUV.t+.2)), rand(vec2(vUV.s+.4,vUV.t+.7)));
      position.xy += ((rnd * vec2(2.)) * .5) * .01;
    }

    // Try not to spawn inside a boundary
    int bnd = checkBounds(position.st) ? 1 : 0;
    // int bnd = checkBndCustom(position.st) ? 1.0 : 0.0;
    int try = 10;
    int i = 0;
    int b = bnd;
    if (bnd > 0) {
      while (b > 0 && i < try) {
        float i2 = float(i)/5.;
        float fn = frame;
        check = vec2(rand(vec2(fn,fn+.2)), rand(vec2(fn+.4,fn+.7)));
        position = texture2D(sTD2DInputs[NOISE], check).xyz;
        b = checkBounds(position.xy) ? 1 : 0;
        // b = checkBndCustom(position.st) ? 1.0 : 0.0;
        i++;
      }
    }

    position.z = 0.0;

    vec2 posCoord = position.xy;
    if (mouseTrig) posCoord = texture2D(sTD2DInputs[POS_OLD], vUV.st).rg;

    velocity = velocity_old = vec3(0.0);
    velocity = texture2D(sTD2DInputs[VELOCITY_OLD], posCoord).rgb;
    velocity_old = velocity;

    normals = vec3(1.0);
    mass = texture2D(sTD2DInputs[MASS], posCoord).r;
    momentum = texture2D(sTD2DInputs[MOMENTUM], posCoord).r;

    // if (mmb > 0) color = texture2D(sTD2DInputs[COLOR_NEW], posCoord);
    // else color = texture2D(sTD2DInputs[COLOR_OLD], vUV.st);
    color = texture2D(sTD2DInputs[COLOR_NEW], posCoord);
    color.a = 1.;

    life = 1.;
    life_var = texture2D(sTD2DInputs[NOISE], posCoord).g;
    life_var = fit(life_var, 0., 1., .5, 1.);
    life_var *= .001;
    // life_var = 0.;
    // life_var = .01;
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
    float mag = sqrt(velocity.x*velocity.x + velocity.y*velocity.y);
    float mag_old = sqrt(velocity_old.x*velocity_old.x + velocity_old.y*velocity_old.y);
    float accel = (abs(mag-mag_old))/.1;

    float cd = sqrt(mag*1.);
    cd *= 20;
    cd = clamp(cd, 0., 1.);

    // Set color
    if (mb.z > 0) color = texture2D(sTD2DInputs[COLOR_NEW], posCoord);
    else color = texture2D(sTD2DInputs[COLOR_OLD], vUV.st);

    color.rgb = vec3(1.);
    // color.rgb = vec3(1.);
    // float hue = atan(velocity.x, velocity.y);
    // color.rgb = bch2rgb(vec3(cd, sqrt(1-cd), hue));

    // Set alpha
    // color.a = 1.;
    // color.a = cd * 2.;
    // color.a = clamp(cd, .4, 1.);
    // color.a = 1.;

    // if (mag < .005) color.a = 0.0; //kills if its too slow... reconsider this
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

} //main
