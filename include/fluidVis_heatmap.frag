uniform float spare1;
out vec4 fragColor;

void main() {
	vec4 cd = texture(sTD2DInputs[0], vUV.st);

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

	float b = (mag * 0.5) + (chemB * 1.);
	float hue = atan(velV, velU);
	hue = (chemA + 3.);
	hue += (chemB + 2.2);

	float sat = mag + ((chemA + chemB)*0.5);
	if (sat > 1.) sat = 1.;
	if (sat < 0.) sat = 0.;

	vec3 bch = bch2rgb(vec3( b, sat, hue) );

	cd.rgb = bch;
	cd.a = 1.0;

	fragColor = cd;
}
