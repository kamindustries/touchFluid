uniform float trailsLength;

out vec4 fragColor;

void main()
{
	vec4 cdNew = texture(sTD2DInputs[0], vUV.st);
	vec4 cdOld = texture(sTD2DInputs[1], vUV.st);
	vec4 cdOut;

	// Slow fade-in trails
	cdNew = clamp(cdNew, 0.,1.);
	cdOld = clamp(cdOld, 0.,1.);

	// cdOld.rgb += cdNew.rgb;
	// cdOld.a = cdNew.a;

	// cdOut = cdOld;
// vec4 result = vec4(gl_FragColor.a) * gl_FragColor + vec4(1.0 - gl_FragColor.a) * pixel_color;

	float len = .95 + (trailsLength * .05);
	if (trailsLength < .001) len = 0.;

	vec4 cd = vec4(cdNew.a) * cdNew + vec4(1. - cdNew.a) * (cdOld * len);

	cdOut = cd;

	// vec3 bch = rgb2bch(cdOld.rgb);
	// 	float len = (trailsLength * .1);
	// 	// bch.x *= .995;
	// 	bch.x *= (.99);
	// 	bch.y *= (.9 + len);
	// 	bch.z += .02;
	// bch = bch2rgb(bch);

	// cdOut.rgb = bch;
	// cdOut.rgb = bch;


	// cdOld = clamp(cdOld, 0.,1.);
	// cdNew = clamp(cdNew, 0.,1.);
	//
	// cdNew += (cdOld);
	// vec3 bch = rgb2bch(cdNew.rgb);
	// 	bch.x *= .99;
	// 	bch.y *= .95;
	// 	bch.z += .01;
	//
	// cdNew.rgb = bch2rgb(bch.rgb) *1;

	// cdOut = cdNew;

	fragColor = cdOut;
}
