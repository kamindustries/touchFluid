uniform float spare1;
out vec4 fragColor;

void main() {
	vec4 cd = texture(sTD2DInputs[0], vUV.st);
	vec4 cd2 = texture(sTD2DInputs[1], vUV.st);
	vec4 cd3 = texture(sTD2DInputs[2], vUV.st);

	cd.r *= 1.;
	cd.g *= 1.;
	cd.b *= 15.;
	cd.a *= 15.;

	float chemA = cd.r;
	float chemB = cd.g;
	float velU = cd.b;
	float velV = cd.a;

	chemA *= .4;
	chemA += (chemB * chemB) * 1;
	cd.rgb = vec3(chemA);

	float speed = (abs(velU)+abs(velV)) * 1.;
	float mag = sqrt(velU*velU + velV*velV);

	// float b = (mag * 0.5) + (chemB * 1.);
	chemB = clamp(chemB, 0., 1.);
	chemB = 1.-chemB;
	float chemFract = smoothstep(0.0, 1.0, fract(chemB * 5.0)) * smoothstep(0.0, 1.0, chemB);

	float b = chemFract;
	b += (cd2.b * 10000.);

	// float b = smoothstep(0., 1., chemFract);
	// b = pow(b, 2.);
	// float hue = atan(velV, velU);
	// float hue = smoothstep(0., 1., chemB*.5) * 3.1459*2.;
	// hue += (mag*2.);
	float hue = .5;
	// hue += mag * 4.;

	// float sat = mag + ((chemA + chemB)*0.5);
	float sat = chemB;
	if (sat > 1.) sat = 1.;
	if (sat < 0.) sat = 0.;

	// sat = 1.;
	// b = chemB*chemB*chemB;

	vec3 bch = bch2rgb(vec3( b, sat, hue) );

	cd.rgb = bch;
	cd.a = 1.0;

	fragColor = cd;
}
