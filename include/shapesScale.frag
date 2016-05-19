
uniform float numCopies;
uniform float copiesOffset;
uniform float rotate;


out vec4 fragColor;
void main()
{
	vec4 cd = texture(sTD2DInputs[0], vUV.st);

	for (int i = 0; i < numCopies; i++) {
		float i_f = float(i+2);

		vec2 offset = (vUV.st * 2.) - 1.;

		//rotate
		float rot = rotate * i_f * .75;
		vec2 scaled = offset;
		offset.x = cos(rot)*scaled.x + sin(rot)*scaled.y;
		offset.y = -sin(rot)*scaled.x + cos(rot)*scaled.y;

		offset = offset * ( i_f * (.3/copiesOffset));

		offset = (offset + 1.) * .5;

		vec4 cp = texture(sTD2DInputs[0], offset);
		cd.rgb += cp.rgb;
	}

	fragColor = cd;
}
