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

	// Setting point size to value so black dots dont show up
	float age = texture2D(life, tcoord).r * 2.;
	float ptSize = age;
	if (ptSize < .5) ptSize = .5;

	gl_PointSize = ptSize;

	#ifndef TD_PICKING_ACTIVE


#else // TD_PICKING_ACTIVE

	// This will automatically write out the nessesarily values
	// for this shader to work with picking.
	// See the documentation if you want to write custom values for picking.
	TDWritePickingValues();

#endif // TD_PICKING_ACTIVE
}
