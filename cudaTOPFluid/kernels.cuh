/* 
	CUDA kernels and functions
	Kurt Kaminski 2016
*/

#include "private/TCUDA_Types.h"
#include "defines.h"
#include <math.h>

//#include <stdio.h>

__device__ int
clamp(int i)
{
	if (i < 0) i = 0;
	if (i > 255) i = 255;
	return i;
}

__device__ float
clamp(float i, float min, float max)
{
	if (i < min) i = min;
	if (i > max) i = max;
	return i;
}

// Get 1d index from 2d coords
__device__ int 
IX(int x, int y) 
{
	return x + (y * blockDim.x * gridDim.x);
}

__device__ int 
getX(int w) 
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	if (x >= w) x = 0; if (x < 0) x = w-1;
	return x;
}

__device__ int 
getY(int h) 
{
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	if (y >= h) y = 0; if (y < 0) y = h-1;
	return y;
}


// Returns true if within the bounds of both the container edges and a user-defined boundary
__device__ bool
checkBounds(int *_boundary, int x, int y, int w, int h)
{
	if (x > 1 && x < w-1 && y > 1 && y < h-1 && _boundary[IX(x,y)] < 1 ){
		return true;
	}
	else {
		return false;
	}
}

// Functions for converting to/from a int (4 bytes, 1 byte per RGBA, which are in the range 0-255)
// to 4 floats in the range 0.0-1.0 
// Note how the data is stored in BGRA format due to how its stored on the GPU.
__device__ int 
rgbaToInt(float r, float g, float b, float a)
{
    return
		(clamp((int)(a * 255.0f)) << 24) |
		(clamp((int)(r * 255.0f)) << 16) |
		(clamp((int)(g * 255.0f)) <<  8) |
		(clamp((int)(b * 255.0f)) <<  0);
}

__device__ void 
intToRgba(int pixel, float &r, float &g, float &b, float &a)
{
	b = float(pixel&0xff) / 255.0f;
	g = float((pixel>>8)&0xff) / 255.0f;
	r = float((pixel>>16)&0xff) / 255.0f;
	a = float((pixel>>24)&0xff) / 255.0f;
}

// Set boundary conditions
__device__ void set_bnd( int b, int x, int y, float *field, int *boundary, int w, int h) {
	int sz = w*h;
	int id = IX(x,y);
	
	bool outOfBnd = false;
	if (boundary[id] > 0) outOfBnd = true;

	//if (x==0)	field[id] = b==1 ? -1*field[IX(1,y)] : field[IX(1,y)];
	//if (x==w-1) field[id] = b==1 ? -1*field[IX(w-2,y)] : field[IX(w-2,y)];
	//if (y==0)   field[id] = b==2 ? -1*field[IX(x,1)] : field[IX(x,1)];
	//if (y==h-1) field[id] = b==2 ? -1*field[IX(x,h-2)] : field[IX(x,h-2)];
	
	if (x==0)	field[id] = b==1 ? -1*field[IX(1,y)] : -1 * field[IX(1,y)];
	if (x==w-1) field[id] = b==1 ? -1*field[IX(w-2,y)] : -1 * field[IX(w-2,y)];
	if (y==0)   field[id] = b==2 ? -1*field[IX(x,1)] : -1 * field[IX(x,1)];
	if (y==h-1) field[id] = b==2 ? -1*field[IX(x,h-2)] : -1 * field[IX(x,h-2)];

	if (outOfBnd){
		field[id] = -1*field[id];
		field[IX(x+1,y)] = -1*field[IX(x+1,y)];
		field[IX(x-1,y)] = -1*field[IX(x-1,y)];
		field[IX(x,y+1)] = -1*field[IX(x,y+1)];
		field[IX(x,y-1)] = -1*field[IX(x,y-1)];
	}

	if (id == 0)      field[id] = 0.5*(field[IX(1,0)]+field[IX(0,1)]);  // southwest
	if (id == sz-w) field[id] = 0.5*(field[IX(1,h-1)]+field[IX(0, h-2)]); // northwest
	if (id == w-1)  field[id] = 0.5*(field[IX(w-2,0)]+field[IX(w-1,1)]); // southeast
	if (id == sz-1)   field[id] = 0.5*(field[IX(w-2,h-1)]+field[IX(w-1,h-2)]); // northeast
}

__global__ void 
DrawSquare( float *field, float value, int w, int h ) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	float posX = (float)x/w;
	float posY = (float)y/h;
	if ( posX < .92 && posX > .45 && posY < .51 && posY > .495 ) {
		field[id] = value;
	}
}

__global__ void 
DrawBnd( int *boundary, int w, int h ) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	float posX = (float)x/w;
	float posY = (float)y/h;
	if ( posX < .82 && posX > .70 && posY < .33 && posY > .21 ) {
		boundary[id] = 1;
	}
	else boundary[id] = 0;
}

__global__ void SetBoundary( int b, float *field, int *boundary, int w, int h ) {
	int x = getX(w);
	int y = getY(h);

	set_bnd(b, x, y, field, boundary, w, h);
}

__global__ void getSum( float *_data, float _sum, int w, int h ) {
  int x = getX(w);
  int y = getY(h);

  _sum += _data[IX(x,y)];
}

__global__ void 
ClearArray(float *field, float value, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	field[id] = value;
}

__global__ void 
ClearArray(int *field, float value, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	field[id] = value;
}

__global__ void GetFromUI ( float * field, float value, int x_coord, int y_coord, int w, int h ) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (x>x_coord-2 && x<x_coord+2 && y>y_coord-2 && y<y_coord+2){
		// if (x == x_coord && y==y_coord){
		field[id] = value;
	}
	else return;
}

__global__ void
MakeSource(int *src, float *dest, int w, int h)
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	int pixel = src[id];
	float r,g,b,a;
	intToRgba(pixel, r, g, b, a);
	
	dest[id] = r;
}

__global__ void
MakeSource(int *src, int *dest, int w, int h)
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	int pixel = src[id];
	float r,g,b,a;
	intToRgba(pixel, r, g, b, a);
	
	dest[id] = src[id]&0xff/255;
}

__global__ void 
AddSource(float *field, float *source, float dt, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	field[id] += (dt * source[id]);
}

__global__ void
MakeColor(float *src, int *dest, int w, int h)
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	dest[id] = rgbaToInt(src[id], src[id], src[id], 1.0);
	//dest[id] = rgbaToInt(1.0, src[id], src[id], 1.0);
}

__global__ void LinSolve( float *field, float *field0, float a, float c, int w, int h) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	field[id] = (float)(field0[id] + ((float)a*(field[IX(x-1,y)] + field[IX(x+1,y)] + field[IX(x,y-1)] + field[IX(x,y+1)]))) / c;
}

/**
 * Calculate the buoyancy force as part of the velocity solver.
 * Fbuoy = -a*d*Y + b*(T-Tamb)*Y where Y = (0,1). The constants
 * a and b are positive with appropriate (physically meaningful)
 * units. T is the temperature at the current cell, Tamb is the
 * average temperature of the fluid grid. The density d provides
 * a mass that counteracts the buoyancy force.
 *
 * In this simplified implementation, we say that the tempterature
 * is synonymous with density (since smoke is *hot*) and because
 * there are no other heat sources we can just use the density
 * field instead of a new, seperate temperature field.
 *
 * @param Fbuoy Array to store buoyancy force for each cell.
 **/

__global__ void buoyancy(float *Fbuoy, float *dens, float _Tamb, float Y, int w, int h)
{
  int x = getX(w);
  int y = getY(h);
  int id = IX(x,y);
  float a = 0.000625f;
  float b = 0.025f;
  Fbuoy[id] = a * dens[id] + -b * (dens[id] - _Tamb) * Y;
  //Fbuoy[id] = a * dens[id] + -b * (dens[id] - _Tamb);
}


/**
 * Calculate the curl at position (i, j) in the fluid grid.
 * Physically this represents the vortex strength at the
 * cell. Computed as follows: w = (del x U) where U is the
 * velocity vector at (i, j).
 *
 * @param i The x index of the cell.
 * @param j The y index of the cell.
 **/

__device__ float curl(int i, int j, float *u, float *v)
{
  float du_dy = (u[IX(i, j+1)] - u[IX(i, j-1)]) * 0.5f;
  float dv_dx = (v[IX(i+1, j)] - v[IX(i-1, j)]) * 0.5f;

  // return du_dy - dv_dx;
  return du_dy - dv_dx;
}


/**
 * Calculate the vorticity confinement force for each cell
 * in the fluid grid. At a point (i,j), Fvc = N x w where
 * w is the curl at (i,j) and N = del |w| / |del |w||.
 * N is the vector pointing to the vortex center, hence we
 * add force perpendicular to N.
 *
 * @param Fvc_x The array to store the x component of the
 *        vorticity confinement force for each cell.
 * @param Fvc_y The array to store the y component of the
 *        vorticity confinement force for each cell.
 **/

__global__ void vorticityConfinement(float *Fvc_x, float *Fvc_y, float *u, float *v, int w, int h)
{
  int x = getX(w);
  int y = getY(h);
  int id = IX(x,y);

  float dw_dx, dw_dy;
  float length;
  float vel;

    if (x>0 && x<w-1 && y>0 && y<h-1){
    // Calculate magnitude of curl(u,v) for each cell. (|w|)
    // curl[I(i, j)] = Math.abs(curl(i, j));

      // Find derivative of the magnitude (n = del |w|)
      dw_dx = ( abs(curl(x+1,y, u, v)) - abs(curl(x-1,y, u, v)) ) * 0.5f;
      dw_dy = ( abs(curl(x,y+1, u, v)) - abs(curl(x,y-1, u, v)) ) * 0.5f;

      // Calculate vector length. (|n|)
      // Add small factor to prevent divide by zeros.
      length = sqrt(dw_dx * dw_dx + dw_dy * dw_dy);
      if (length == 0.0) length -= 0.000001f;
      // N = ( n/|n| )
      dw_dx /= length;
      dw_dy /= length;

      vel = curl(x, y, u, v);

      // N x w
      Fvc_x[id] = dw_dy * -vel;
      Fvc_y[id] = dw_dx *  vel;
    }

}


__global__ void Advect ( float *field, float *field0, float *u, float *v, float dt, int w, int h ) {
  int i = getX(w);
  int j = getY(h);
  int id = IX(i,j);

  int i0, j0, i1, j1;
  float x, y, s0, t0, s1, t1, dt0;

  dt0 = (float)dt*float(w-2);

  // if (x>0 && x<DIM-1 && y>0 && y<DIM-1){
    x = (float)i - dt0 * u[id];
    y = (float)j - dt0 * v[id];

    if (x < 0.5f) x = 0.5f;
    if (x > (float)(w-2.0)+0.5f) x = (float)(w-2.0)+0.5f;
    i0 = (int)x;
    i1 = i0+1;

    if (y < 0.5f) y = 0.5f;
    if (y > (float)(h-2.0)+0.5f) y = (float)(h-2.0)+0.5f;
    j0 = (int)y;
    j1 = j0+1;

    s1 = (float)x-i0;
    s0 = (float)1-s1;
    t1 = (float)y-j0;
    t0 = (float)1-t1;

    field[id] = (float)s0*(t0*field0[IX(i0,j0)] + t1*field0[IX(i0,j1)])+
			 				         s1*(t0*field0[IX(i1,j0)] + t1*field0[IX(i1,j1)]);
  // }
}

__global__ void Project ( float *u, float *v, float *p, float *div, int w, int h ) {
  int x = getX(w);
  int y = getY(h);
  int id = IX(x,y);

  if (x>0 && x<w-1 && y>0 && y<h-1){
    div[id] = -0.5 *(u[IX(x+1,y)] - u[IX(x-1,y)] + v[IX(x,y+1)] - v[IX(x,y-1)]) / float(w-2);
    p[id] = 0;
  }
}

__global__ void ProjectFinish ( float *u, float *v, float *p, float *div, int w, int h ) {
  int x = getX(w);
  int y = getY(h);
  int id = IX(x,y);

  if (x>0 && x<w-1 && y>0 && y<h-1){
    u[id] -= (0.5 * float(w-2) * (p[IX(x+1,y)] - p[IX(x-1,y)]));
    v[id] -= (0.5 * float(h-2) * (p[IX(x,y+1)] - p[IX(x,y-1)]));
  }
}

__global__ void 
Diffusion(float *_chem, float *_lap, int *_boundary, float _difConst, float dt, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	//if (checkBounds(_boundary, x, y, w, h)) {
	//	int n1 = id + 1;
	//	int n2 = id - 1;
	//	int n3 = id + w;
	//	int n4 = id - w;

		// constants
		float xLength = (float)x/100.0;
		float dx = (float)xLength/(float)x;
		float alpha = (float)(_difConst * dt / (float)(dx*dx));

		_lap[id] = (float)(-4.0f * _chem[id]) + (float)(_chem[IX(x+1,y)] + _chem[IX(x-1,y)] + _chem[IX(x,y+1)] + _chem[IX(x,y-1)]);
		_lap[id] = (float)_lap[id]*alpha;
	//}
}

__global__ void 
AddLaplacian( float *_chem, float *_lap, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	_chem[id] += _lap[id];
}

__global__ void React( float *_chemA, float *_chemB, int *_boundary, float dt, int w, int h) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	//if (checkBounds(_boundary, x, y, w, h)) {
		float F = 0.05;
		float k = 0.0675;
		float A = _chemA[id];
		float B = _chemB[id];

		float reactionA = -A * (B*B) + (F * (1.0-A));
		float reactionB = A * (B*B) - (F+k)*B;
		_chemA[id] += (dt * reactionA);
		_chemB[id] += (dt * reactionB);
	//}
	//else {
		//_chemA[id] = 0.0;
		//_chemB[id] = 0.0;
	//}
}

__global__ void
sampleKernel( int* src, int inw, int inh, int *dest, int w, int h )
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	
	// If the resolution isn't a multiple of the grid/thread size, or the resolutions don't match
	// we need to make sure we arn't reading or writting beyond the bounds of the data 
	if (x >= inw || y >= inh)
	{
		if (x < w && y < h)
			dest[y * w + x] = rgbaToInt(0.0f, 0.0f, 0.0f, 1.0);
		return;
	}
	else if (x >= w || y >= h)
	{
		return;
	}
	else
	{
		int pixel = src[y * inw + x];
		float r,g,b,a;
		intToRgba(pixel, r, g, b, a);
		
		// Simple monochrome operation
		float v = r*0.3f + g*0.6f + b*0.1f;
		dest[y * w + x] = rgbaToInt(v, v, v, a);
	}
}


