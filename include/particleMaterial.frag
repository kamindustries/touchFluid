in Vertex {
	vec4 color;
	vec3 camSpaceVert;
	vec3 camVector;
	vec3 norm;
}vVert;

// Output variable for the color
layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

uniform float softEdge;

float luminance(vec3 c) {
	return dot(c, vec3(.2126, .7152, .0722));
}

void main() {
	vec2 coord = (gl_PointCoord.xy * vec2(2.)) - vec2(1.);
	float dist =  sqrt(dot(coord, coord)); //make circle sprites

	vec4 cd = vVert.color;

	float soft = 1.-softEdge;
	soft *= soft*soft;
	soft *= 10;
	soft += .02; //min softness
	dist = pow(dist, soft);
	if (dist > 1.) cd = vec4(0.0);

	// cd.a *= luminance(cd.rgb); // !!!watch out for this!!!
	cd.a *= (1.-clamp(dist,0.,1.));

	fragColor[0] = cd;

}
