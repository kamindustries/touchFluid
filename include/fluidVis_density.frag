uniform float spare1;
out vec4 fragColor;

void main() {
	vec4 cd = texture(sTD2DInputs[0], vUV.st);

	float chemA = cd.r;
	float chemB = cd.g;
	float velU = cd.b;
	float velV = cd.a;

	float dens = (chemB) * 10.;
	dens = floor(dens);
 	dens /= 10.;
	dens = clamp(dens, 0., 2.);
	dens = dens*dens;

	// vec3 bch;
	// 	bch.r = dens*1.4;
	// 	// bch.g = dens;
	// 	bch.g = 0.;
	// 	bch.b = 2.+dens*3.145;
	// bch = bch2rgb(bch);
	vec3 bch;
		bch.r = chemB*1.4;
		// bch.g = dens;
		bch.g = 0.;
		bch.b = 2.+chemB*3.145;
	bch = bch2rgb(bch);

	cd = vec4(bch, 1.);

	fragColor = cd;
}
