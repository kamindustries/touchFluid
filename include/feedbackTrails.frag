out vec4 fragColor;
void main()
{
	vec4 cdNew = texture(sTD2DInputs[0], vUV.st);
	vec4 cdOld = texture(sTD2DInputs[1], vUV.st);
	vec4 cdOut = cdNew;

	// Slow fade-in trails
	cdNew = clamp(cdNew, 0.,1.);
	cdOld = clamp(cdOld, 0.,1.);
	cdOld += cdNew*.6;

	vec3 bch = rgb2bch(cdOld.rgb);
		bch.x *= .94;
		bch.y *= .96;
		bch.z += .02;
	bch = bch2rgb(bch);

	cdOut.rgb = bch;

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
