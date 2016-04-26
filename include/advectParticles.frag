layout(location=0) out vec4 final_color0;
layout(location=1) out vec4 final_color1;
layout(location=2) out vec4 final_color2;
layout(location=3) out vec4 final_color3;
layout(location=4) out vec4 final_color4;
layout(location=5) out vec4 final_color5;

uniform float res;
uniform float trigger;
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

///////////////////////////////////////////////////////////
// Ghetto random
///////////////////////////////////////////////////////////
float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}


///////////////////////////////////////////////////////////
// M A I N
///////////////////////////////////////////////////////////
void main() {
	bool anim = true;

  vec4 color;
	vec3 position, normals, velocity, velocity_old;
	float mass, momentum;


  if (trigger > 0.0 || texture2D(sTD2DInputs[COLOR_OLD], vUV.st).a < 0.01) {
    // vec2 cx = (vUV.st * 2.0) - 1.0;
    // cx *= 10.0;
    // position = vec3(cx, 0.0);

    // Try not to spawn inside a boundary
    float bnd = (texture2D(sTD2DInputs[BOUNDARY], vUV.st)).b;
    int try = 5;
    int i = 0;
    float b = bnd;
    if (bnd > 0.0) {
      while (b > 0. && i < try) {
        float i2 = float(i)/5.;
        vec2 check = vec2(rand(vUV.st+i2), rand(vUV.st*4.+i2));
        b = texture2D(sTD2DInputs[BOUNDARY], check).b;
        position = (vec3(check.x, check.y, 0) * 2.) - 1.;
        i++;
      }
    }
    else {
      position = (vec3(vUV.s + (rand(vUV.st)*.01), vUV.t + (rand(vUV.st*2.2)*.01), 0) * 2.) - 1.;
    }

    // position = (texture2D(sTD2DInputs[NOISE], vUV.st ).rgb * 2.0) - 1.0;
    position.z = 0.0;
    position *= 10.;

    vec2 texCoord3 = (position.xy + vec2(10.0))/20.0; //Find out why it's -10 to 10 and not -1 to 1

    velocity = velocity_old = vec3(0.0);
    normals = vec3(1.0);
    mass = texture2D(sTD2DInputs[MASS], texCoord3 ).r;
    momentum = texture2D(sTD2DInputs[MOMENTUM], texCoord3 ).r;

    // float mag = (sqrt(abs(velocity_old.x*velocity_old.x) + abs(velocity_old.y*velocity_old.y)));
    // color.rgb = vec3(mag);
    // color.rgb *= 20.;

    color.rgb = texture2D(sTD2DInputs[COLOR_NEW], vUV.st).rgb;
    color.a = mass;
  }

	else {
    position = texture2D(sTD2DInputs[POS_OLD], vUV.st).rgb;
    normals = texture2D(sTD2DInputs[NRML_OLD], vUV.st).rgb;
    color = texture2D(sTD2DInputs[COLOR_OLD], vUV.st);
    mass = texture2D(sTD2DInputs[MASS_OLD], vUV.st ).r;
		momentum = texture2D(sTD2DInputs[MOMENTUM_OLD], vUV.st ).r;

    vec2 posCoord = (position.xy+vec2(10.0))/20.0;
    velocity_old = texture2D(sTD2DInputs[VELOCITY_OLD], vUV.st ).rgb;
    velocity = texture2D(sTD2DInputs[VELOCITY], posCoord ).rgb;
    velocity = velocity * (mass * 1.0) + velocity_old * momentum;

    // Get the magnitude
    float mag = (sqrt(abs(velocity.x*velocity.x) + abs(velocity.y*velocity.y)));
    float cd = mag*20.;

    // Set color
    float hue = atan(velocity.x, velocity.y);
    color.rgb = bch2rgb(vec3(clamp(cd, .5, 1.), cd*.7,  hue));

    // Set alpha
    // color.a -= .005;
    color.a = clamp(cd, .1, 1.);
    if (mag < .001) color.a = 0.0; //kills if its too slow... reconsider this

    // Boundary conditions
    float oInc = increment * 10.;
    vec2 pNew = position.xy;
    vec2 bN = (vec2(pNew.x, pNew.y + oInc) + 10.0) / 20.0;
    vec2 bS = (vec2(pNew.x, pNew.y - oInc) + 10.0) / 20.0;
    vec2 bE = (vec2(pNew.x + oInc, pNew.y) + 10.0) / 20.0;
    vec2 bW = (vec2(pNew.x - oInc, pNew.y) + 10.0) / 20.0;
    float oN = texture2D(sTD2DInputs[BOUNDARY], bN).b;
    float oS = texture2D(sTD2DInputs[BOUNDARY], bS).b;
    float oE = texture2D(sTD2DInputs[BOUNDARY], bE).b;
    float oW = texture2D(sTD2DInputs[BOUNDARY], bW).b;
    float oC = texture2D(sTD2DInputs[BOUNDARY], (pNew+10.)/20.).b;

    if (oN > 0.0 || oS > 0.0) {
      velocity.y *= -1;
    }
    if (oE > 0.0 || oW > 0.0) {
      velocity.x *= -1;
    }
    if (oC > 0.0) color.a = 0.0;

    // Position and normals
    position += velocity;
    normals = normalize(normals + ((velocity - 0.5) / 3.));

    // Basic boundary conditions
    // float bounds = 9.95;
		// if (position.x < -1*bounds) {
		// 	position.x = -1*bounds;
		// 	velocity.x *= -1;
		// }
		// if (position.x > bounds) {
		// 	position.x = bounds;
		// 	velocity.x *= -1;
		// }
		// if (position.y < -1*bounds) {
		// 	position.y = -1*bounds;
		// 	velocity.y *= -1;
		// }
		// if (position.y > bounds) {
		// 	position.y = bounds;
		// 	velocity.y *= -1;
		// }
    // position += velocity;

  } //else


	final_color0 = vec4(position.rgb,1);
	final_color1 = vec4(normals,1);
	final_color2 = vec4(color);
	final_color3 = vec4(velocity,1);
	final_color4 = vec4(vec3(mass),1);
	final_color5 = vec4(vec3(momentum),1);

} //main
