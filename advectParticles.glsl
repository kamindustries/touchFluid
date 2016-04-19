layout(location=0) out vec4 final_color0;
layout(location=1) out vec4 final_color1;
layout(location=2) out vec4 final_color2;
layout(location=3) out vec4 final_color3;
layout(location=4) out vec4 final_color4;
layout(location=5) out vec4 final_color5;

uniform vec3 resolution;
uniform vec4 colz;
uniform vec4 trigger;
float res = 500.0;
float increment = 1.0/500.0;


/////////////////////////////////////////////////////
// important bit! Play with this shiz
float speed = .1;

//this increases the frequency of the curling effect.
uniform float tmult = 1.0;

//this increases the turbulence effect
////////////////////////////////////////////////////

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

uniform float turbulencemult = 2.0;
void main()
{
	bool anim = true;

	vec3 position, normals, color, velocity, velocity_old;
	float mass, momentum;


	float limits = 1-(increment * res );

  if (trigger.x > 0.0) {
    vec2 cx = (vUV.st * 2.0) - 1.0;
    cx *= 10.0;
    position = vec3(cx, 0.0);

    float rx = (rand(vUV.st) * 2.) - 1.;
    float ry = (rand(vUV.st+.3*.3) * 2.) - 1.;
    position = vec3(rx*10., ry*10., 0.0);

    // color = vec3(1.0);
    color = texture2D(sTD2DInputs[13], vUV.st ).rgb;
    vec2 texCoord3 = (position.xy+vec2(10.0))/20.0;
    color = texture2D(sTD2DInputs[13], texCoord3 ).rgb;

    velocity = vec3(0.0);

    // velocity = texture2D(sTD2DInputs[7], vUV.st ).rgb;

    normals = vec3(1.0);
    mass = rand(vUV.st);
    momentum = rand(vUV.st+.4*.333);
  }


	// if(vUV.t > 1-increment && anim) {
	// 	//get the incoming data
	// 	position = texture2D(sTD2DInputs[4],vec2(vUV.s,0.0)).rgb; //lookup our position
	// 	normals = texture2D(sTD2DInputs[6],vec2(vUV.s,0.0)).rgb;
	// 	color = clamp(colz.rgb, 0.3,1.0) /1.0  + (vec3(1,1,0) * trigger.x /1);
	// 	velocity = vec3(0.0);
  //
	// 	mass = texture2D(sTD2DInputs[9],vec2(vUV.s,0.0)).r;
	// 	momentum = texture2D(sTD2DInputs[10],vec2(vUV.s,0.0)).r;
	// }

	else {

    position = texture2D(sTD2DInputs[1], vUV.st).rgb;
    normals = texture2D(sTD2DInputs[2], vUV.st).rgb;
    color = texture2D(sTD2DInputs[5], vUV.st).rgb;
    mass = texture2D(sTD2DInputs[11], vUV.st ).r;
		momentum = texture2D(sTD2DInputs[12], vUV.st ).r;

    vec2 posCoord = (position.xy+vec2(10.0))/20.0;

    velocity_old = texture2D(sTD2DInputs[8], vUV.st ).rgb;
    velocity = texture2D(sTD2DInputs[7], posCoord ).rgb;
    velocity = velocity * (mass * 1.0) * (500./500.) + velocity_old * momentum;

    position += velocity;

		float bounds = 9.0;
		if (position.x < -1*bounds) {
			position.x = -1*bounds;
			velocity.x *= -1;
		}
		if (position.x > bounds) {
			position.x = bounds;
			velocity.x *= -1;
		}
		if (position.y < -1*bounds) {
			position.y = -1*bounds;
			velocity.y *= -1;
		}
		if (position.y > bounds) {
			position.y = bounds;
			velocity.y *= -1;
		}


		normals = normalize( normals +((velocity - 0.5) / 3));
		// color -= increment;

		normals = vec3(1.0);
		// color = vec3(1.0);
    // color = vec3(.5, .1, .7);


	}



	final_color0 = vec4(position.rgb,1);
	final_color1 = vec4(normals,1);
	final_color2 = vec4(color,1);
	final_color3 = vec4(velocity,1);
	final_color4 = vec4(vec3(mass),1);
	final_color5 = vec4(vec3(momentum),1);

}
