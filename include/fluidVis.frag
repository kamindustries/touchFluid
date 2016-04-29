uniform float spare1;
out vec4 fragColor;

void main() {
	vec4 cd = texture(sTD2DInputs[0], vUV.st);

	// cd.r *= 1.;
	// cd.g *= 1.;
	// cd.b *= 15.;
	// cd.a *= 15.;

	float chemA = cd.r;
	float chemB = cd.g;
	float velU = cd.b;
	float velV = cd.a;

	if (abs(velU) < 0.004) velU = 0.004;
	if (abs(velV) < 0.004) velV = 0.004;

	float speed = (abs(velU)+abs(velV)) * 1.;
	float mag = sqrt(velU*velU + velV*velV);

	vec3 bch;
		bch.x = (chemA*chemA) + (chemB*chemB);
		bch.x = clamp(bch.x, 0., 1.);

		bch.y = clamp(chemB*chemB, 0., 1.);

		bch.z = atan(velU, velV);
	bch = bch2rgb(bch);

	cd = vec4(bch, 1.);

	fragColor = cd;
}
