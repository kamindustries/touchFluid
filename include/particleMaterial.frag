in Vertex {
	vec4 color;
	vec3 camSpaceVert;
	vec3 camVector;
	vec3 norm;
}vVert;

// Output variable for the color
layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

void main()
{
	TDCheckOrderIndTrans();
	// fragColor[0] = vVert.color;
	vec2 coord = (gl_PointCoord.xy * vec2(2.)) - vec2(1.);
	float dist =  sqrt(dot(coord, coord)); //make circle sprites

	vec4 cd = vVert.color;
	if (dist > 1.) cd = vec4(0.0);
	fragColor[0] = cd;

}
