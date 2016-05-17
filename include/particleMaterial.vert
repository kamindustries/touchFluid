uniform sampler2D positions;
uniform sampler2D normals;
uniform sampler2D colors;
uniform sampler2D life;
uniform float res;

float increment = 1/res;
in float pID;

out Vertex {
	vec4 color;
	vec3 camSpaceVert;
	vec3 camVector;
	vec3 norm;
}vVert;

// Tukey Windowing
// sig expected to be in 0-1 range
float tukeyWindow(float sig, float alpha) {
	if (alpha == 0.) // rectangular window
		return 1.;
	else if (sig < alpha / 2.)
		return 0.5 * (1. + cos(2.*3.1459265/alpha * (sig-alpha/2.)));
	else if (sig > 1.-alpha/2.)
		return 0.5 * (1. + cos(2.*3.1459265/alpha * (sig-1.+alpha/2.)));
	else
		return 1.;
}

void main()
{
  float h = 0.5/res; //needs to be 1/2 / res for center of pixel

	float id = gl_VertexID;
	float off = mod(id,res);
	id = floor(id/res);

  float u = id/res;
  float v = off/res;
  u += h;
  v += h;

	// int id_i = gl_VertexID;
  // float row = floor(id_i % int(res)) + h;
 	//u = (row * h) + (h * 0.5);
	//v = ((pID/500.) * h) + (h * 0.5);

	vec2 tcoord = vec2(u,v);
	vec4 positions = texture(positions, tcoord);

	vec4 worldSpaceVert =TDDeform(positions);
	vec4 camSpaceVert = uTDMat.cam * worldSpaceVert;
	gl_Position = TDWorldToProj(worldSpaceVert);

	vec4 cd = texture2D(colors, tcoord);
	vVert.color = cd;

	// Tukey window
	float ptSize = tukeyWindow(1.-texture2D(life, tcoord).r, 0.01);
	ptSize *= 1.5;
	if (ptSize < 1.) ptSize = 0.;

	gl_PointSize = ptSize;

	#ifndef TD_PICKING_ACTIVE


#else // TD_PICKING_ACTIVE

	// This will automatically write out the nessesarily values
	// for this shader to work with picking.
	// See the documentation if you want to write custom values for picking.
	TDWritePickingValues();

#endif // TD_PICKING_ACTIVE
}
