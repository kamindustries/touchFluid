uniform float spare1;
out vec4 fragColor;

void main() {
	vec4 cd = texture(sTD2DInputs[0], vUV.st);
	vec4 cd2 = texture(sTD2DInputs[1], vUV.st);

	// cd.r *= 1.;
	// cd.g *= 1.;
	// cd.b *= 15.;
	// cd.a *= 15.;

	float chemA = cd.r;
	float chemB = cd.g;
	float velU = cd.b;
	float velV = cd.a;

	float temperature = cd2.r;
	float pressure = cd2.g;
	float divergence = cd2.b;

	float speed = (abs(velU)+abs(velV)) * 1.;
	float mag = sqrt(velU*velU + velV*velV);

	vec3 bch;
		bch.r = (chemA*chemA) + (chemB*chemB);
		bch.r = chemA + chemB;

		bch.g = 1.-clamp(chemB*chemB, 0., 1.);
		bch.g *= (mag*10.);
		// float sat = mag*10.;
		// sat = sqrt(sat);
		// bch.g = clamp(sat, 0., 1.);

		bch.b = atan(velU, velV);
		// bch.b = cd2.r;
	bch = bch2rgb(bch);

	cd = vec4(bch, 1.);

	fragColor = cd;
}
