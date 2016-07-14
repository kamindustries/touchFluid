layout(location = 0) out vec4 toDensTemp;
layout(location = 1) out vec4 toParticles;

uniform float densToggle;
uniform float tempToggle;
uniform float nReact;
uniform float F;
uniform float dt;
uniform int rdEq;

out vec4 fragColor;
void main()
{
	vec4 pt = texture(sTD2DInputs[0], vUV.st);
	vec4 shapes = texture(sTD2DInputs[1], vUV.st);
	vec4 images = texture(sTD2DInputs[2], vUV.st);
	vec4 words = texture(sTD2DInputs[3], vUV.st);
	vec4 multitouch = texture(sTD2DInputs[4], vUV.st);

	float velocityMag = sqrt(pt.r*pt.r + pt.g*pt.g);
	velocityMag = clamp(velocityMag*5., 0., 1.);

	float dtMult = dt * 20.;

	float mixed = pt.b * dtMult;
	float shapeMult = shapes.r * dtMult;
	float imgMult = rgb2bch(images.rgb).x * dtMult;
	float wordMult = rgb2bch(words.rgb).x * dtMult;
	float multitouchMult = multitouch.b * 2.2;

	mixed = mixed + shapeMult + imgMult + wordMult;
	mixed = clamp(mixed, 0., 1.);

	float densA = 0.;
	float densB = mixed * densToggle;
	float temp = mixed * tempToggle;

	// !! Always add multitouch density !!
	densB += multitouchMult;

	if (rdEq > 0){
		if (rdEq > 1) densA = clamp(densB, 0., 1.);
		densA = densA * densToggle;
		densB = clamp(densB, 0., 1.) * F * .5;
	}

	// TO DENSITY AND TEMPERATURE
	vec4 cd = vec4(densA, densB, temp, 1.);
	toDensTemp = cd;

	// TO PARTICLE COLOR
	cd = ((pt*velocityMag*10.) + shapes + images + words + multitouch);
	toParticles = cd;
}
