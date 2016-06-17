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

	float len = sqrt(smoothstep(0., 1., sqrt(trailsLength)));
	// len = .95 + (len * .05);
	// len = .9;
	// len = 0.6;

	// len = .95 + (trailsLength * .05);
	// if (trailsLength < .001) len = 0.;

	// Over
	// f = A.a * A + (1 - A.a) * B
	vec4 cd = vec4(cdNew.a) * cdNew + vec4(1. - cdNew.a) * (cdOld * len);

	cdOut = cd;

	fragColor = cdOut;
}
