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

	float velocityMag = sqrt(pt.r*pt.r + pt.g*pt.g);
	velocityMag = clamp(velocityMag*5., 0., 1.);

	float mixed = velocityMag;
	float shapeMult = shapes.r * (dt * 10.);
	float imgMult = rgb2bch(images.rgb).x * (dt * 5.);
	float wordMult = rgb2bch(words.rgb).x * (dt * 5.);

	mixed = mixed + shapeMult + imgMult + wordMult;


	mixed = clamp(mixed, 0., 1.);

	// float densA = mixed * densToggle;
	//densA = .02 * (nReact+1.);
	float densA = 0.;
	float densB = mixed * densToggle;
	float temp = mixed * tempToggle;

	if (rdEq > 0){
		densA = mixed * densToggle;
		densB = mixed * F * .5;
	}

	vec4 cd = vec4(densA, densB, temp, 1.);

	fragColor = cd;
}
