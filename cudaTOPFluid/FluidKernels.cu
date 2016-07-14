/* 
	CUDA kernels and functions
	Kurt Kaminski 2016
*/

#ifndef __FLUID_KERNELS__
#define __FLUID_KERNELS__

#include <cuda_runtime.h>


/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
// Get 1d index from 2d coords
//__device__ int 
//IX(int x, int y) 
//{
//	return x + (y * blockDim.x * gridDim.x);
//}
//
//__device__ int 
//getX(int w) 
//{
//	int x = threadIdx.x + (blockIdx.x * blockDim.x);
//	//if (x >= w) x = 0; if (x < 0) x = w-1;
//	if (x >= w) return 0;
//	else return x;
//}
//
//__device__ int 
//getY(int h) 
//{
//	int y = threadIdx.y + (blockIdx.y * blockDim.y);
//	//if (y >= h) y = 0; if (y < 0) y = h-1;
//	if (y >= h) return 0;
//	else return y;
//}
//
//__global__ void 
//ClearArray(float *field, float value, int w, int h) 
//{
//	int x = getX(w);
//	int y = getY(h);
//	int id = IX(x,y);
//	
//	field[id] = value;
//}
//
//__global__ void 
//ClearArray(int *field, float value, int w, int h) 
//{
//	int x = getX(w);
//	int y = getY(h);
//	int id = IX(x,y);
//
//	field[id] = value;
//}
//
//__device__ int
//clamp(int i)
//{
//	if (i < 0)
//		i = 0;
//	if (i > 255)
//		i = 255;
//		
//	return i;
//}
//
//// Functions for converting to/from a int (4 bytes, 1 byte per RGBA, which are in the range 0-255)
//// to 4 floats in the range 0.0-1.0 
//// Note how the data is stored in BGRA format due to how its stored on the GPU.
//__device__ int 
//rgbaToInt(float r, float g, float b, float a)
//{
//    return
//        (clamp((int)(a * 255.0f)) << 24) |
//        (clamp((int)(r * 255.0f)) << 16) |
//        (clamp((int)(g * 255.0f)) <<  8) |
//        (clamp((int)(b * 255.0f)) <<  0);
//}
//
//__device__ 
//void intToRgba(int pixel, float &r, float &g, float &b, float &a)
//{
//	b = float(pixel&0xff) / 255.0f;
//	g = float((pixel>>8)&0xff) / 255.0f;
//	r = float((pixel>>16)&0xff) / 255.0f;
//	a = float((pixel>>24)&0xff) / 255.0f;
//}
//
//
//__global__ void
//sampleKernel( int* src, int inw, int inh, int *dest, int w, int h )
//{
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
//	int bw = blockDim.x;
//	int bh = blockDim.y;
//	int x = blockIdx.x*bw + tx;
//	int y = blockIdx.y*bh + ty;
//	
//	// If the resolution isn't a multiple of the grid/thread size, or the resolutions don't match
//	// we need to make sure we arn't reading or writting beyond the bounds of the data 
//	if (x >= inw || y >= inh)
//	{
//		if (x < w && y < h)
//			dest[y * w + x] = rgbaToInt(0.0f, 0.0f, 0.0f, 1.0);
//		return;
//	}
//	else if (x >= w || y >= h)
//	{
//		return;
//	}
//	else
//	{
//		int pixel = src[y * inw + x];
//		float r,g,b,a;
//		intToRgba(pixel, r, g, b, a);
//		
//		// Simple monochrome operation
//		float v = r*0.3f + g*0.6f + b*0.1f;
//		dest[y * w + x] = rgbaToInt(v, v, v, a);
//	}
//}
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

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

__device__ float 
fitRange(float valueIn, float baseMin, float baseMax, float limitMin, float limitMax) 
{
	return ((limitMax - limitMin) * (valueIn - baseMin) / (baseMax - baseMin)) + limitMin;
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
	//if (x >= w) x = 0; if (x < 0) x = w-1;
	if (x >= w) return 0;
	else return x;
}

__device__ int 
getY(int h) 
{
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	//if (y >= h) y = 0; if (y < 0) y = h-1;
	if (y >= h) return 0;
	else return y;
}

// Returns true if within the bounds of both the container edges and a user-defined boundary
__device__ bool
checkBounds(float *_boundary, int x, int y, int w, int h)
{
	if (x > 1 && x < w-2 && y > 1 && y < h-2 && _boundary[4*IX(x,y)+0] < 1 ){
		return true;
	}
	else {
		return false;
	}
}
__device__ bool
checkBounds(int x, int y, int w, int h)
{
	if (x > 1 && x < w-2 && y > 1 && y < h-2){
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

__device__ void
rgbaToColor(float *dest, int id, float r, float g, float b, float a)
{
	dest[4*id+0] = b;
	dest[4*id+1] = g;
	dest[4*id+2] = r;
	dest[4*id+3] = a;
}

// Set boundary conditions
__device__ void 
set_bnd( int b, int x, int y, float *field, float *boundary, int w, int h) {
	int sz = w*h;
	int id = IX(x,y);
	
	bool outOfBnd = boundary[4*id+0] > 0.0 ? true : false;
	//if (boundary[4*id+0] > 0.0) outOfBnd = true;

	//if (x==0)	field[id] = b==1 ? -1*field[IX(1,y)] : field[IX(1,y)];
	//if (x==w-1) field[id] = b==1 ? -1*field[IX(w-2,y)] : field[IX(w-2,y)];
	//if (y==0)   field[id] = b==2 ? -1*field[IX(x,1)] : field[IX(x,1)];
	//if (y==h-1) field[id] = b==2 ? -1*field[IX(x,h-2)] : field[IX(x,h-2)];
	
	if (x==0 || outOfBnd)	field[id] = b==1 ? -1*field[IX(1,y)] : -1 * field[IX(1,y)];
	if (x==w-1 || outOfBnd) field[id] = b==1 ? -1*field[IX(w-2,y)] : -1 * field[IX(w-2,y)];
	if (y==0 || outOfBnd)   field[id] = b==2 ? -1*field[IX(x,1)] : -1 * field[IX(x,1)];
	if (y==h-1 || outOfBnd) field[id] = b==2 ? -1*field[IX(x,h-2)] : -1 * field[IX(x,h-2)];

	//if (outOfBnd){
	//	field[id] = -1*field[id];
	//	field[IX(x+1,y)] = -1*field[IX(x+1,y)];
	//	field[IX(x-1,y)] = -1*field[IX(x-1,y)];
	//	field[IX(x,y+1)] = -1*field[IX(x,y+1)];
	//	field[IX(x,y-1)] = -1*field[IX(x,y-1)];
	//}

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

//__global__ void 
//DrawBnd( int *boundary, int w, int h ) {
//	int x = getX(w);
//	int y = getY(h);
//	int id = IX(x,y);
//
//	float posX = (float)x/w;
//	float posY = (float)y/h;
//	if ( posX < .82 && posX > .70 && posY < .33 && posY > .21 ) {
//		boundary[id] = 1;
//	}
//	else boundary[id] = 0;
//}

__global__ void 
SetBoundary( int b, float *field, float *boundary, int w, int h ) {
	int x = getX(w);
	int y = getY(h);

	set_bnd(b, x, y, field, boundary, w, h);
}

__global__ void 
getSum( float *_data, float _sum, int w, int h ) {
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

__global__ void 
MapArray(float *field, float value, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	field[id] = float(x) * value;
}

// How can I template these?
__global__ void 
AddFromUI ( float *field, float value, float dt, int x_coord, int y_coord, int w, int h ) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (x>x_coord-5 && x<x_coord+5 && y>y_coord-5 && y<y_coord+5){
		// if (x == x_coord && y==y_coord){
		field[id] += value * dt;
	}
	else return;
}

__global__ void 
AddFromUI ( float *field, float *valueUI, int index, float dt, int w, int h ) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	field[id] += valueUI[4*id+index] * dt;
}

__global__ void 
AddObstacleVelocity ( float *u, float *v, float *obstacle, float dt, int w, int h ) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	u[id] += obstacle[4*id+2] * dt; //red
	v[id] += obstacle[4*id+1] * dt; //green
}

__global__ void 
SetFromUI ( float *A, float *B, float *valueUI, int w, int h ) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	float v1 = valueUI[4*id+2]; //red
	float v2 = valueUI[4*id+1]; //green

	if (v1 > 0.0) A[id] = v1;
	if (v2 > 0.0) B[id] = v2;
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

// *!* This is currently only grabbing the red channel *!*
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

__global__ void
MakeColor(float *src0, float *src1, float *src2, float *dest, int w, int h)
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	rgbaToColor(dest, id, src0[id], src1[id], src2[id], 1.0);
}


__global__ void
MakeColor(float *src0, float *src1, float *src2, float *src3, float *dest, int w, int h)
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	rgbaToColor(dest, id, src0[id], src1[id], src2[id], src3[id]);
}

__global__ void
MakeColorLong(	float *r1, float *g1, float *b1, float *a1, 
				float *r2, float *g2, float *b2, float *a2,
				float *dest, int w, int h, int stride)
{
	int idIn = getY(h) * w + (getX(w));
	int idOut = getY(h) * (w*stride) + (getX(w*stride));

	//Left image
	dest[4*idOut+0] = b1[idIn];
	dest[4*idOut+1] = g1[idIn];
	dest[4*idOut+2] = r1[idIn];
	dest[4*idOut+3] = a1[idIn];

	//Right image
	idOut += w;
	dest[4*idOut+0] = b2[idIn];
	dest[4*idOut+1] = g2[idIn];
	dest[4*idOut+2] = r2[idIn];
	dest[4*idOut+3] = 1.0;
}

__device__ float
bilerp(float *src, float _i, float _j, int w, int h)
{
	int i0, j0, i1, j1;
	float s0, t0, s1, t1;
	float i;
	float j;

	// fit bounds
	i = (_i < 0.5f) ? 0.5f : _i;
	i = (_i > float(w)-2.0+0.5f) ? float(w)-2.0+0.5f : _i;

	j = (_j > float(h)-2.0+0.5f) ? float(h)-2.0+0.5f : _j;
	j = (_j < 0.5) ? 0.5 : _j;

	//if (_i < 0.5f) i = 0.5f;
	//else if (_i > float(w)-2.0+0.5f) i = float(w)-2.0+0.5f;
	//if (j < 0.5f) j = 0.5f;
	//else if (j > float(h)-2.0+0.5f) j = float(h)-2.0+0.5f;
		
	// bilinear interpolation
	i0 = int(i);
	i1 = i0+1;		
	j0 = int(j);
	j1 = j0+1;
		
	s1 = (float)i-i0;
	s0 = (float)1-s1;
	t1 = (float)j-j0;
	t0 = (float)1-t1;

	return (float)	s0*(t0*src[IX(i0,j0)] + t1*src[IX(i0,j1)])+
			 		s1*(t0*src[IX(i1,j0)] + t1*src[IX(i1,j1)]);
}

__global__ void 
Advect (float *vel_u, float *vel_v, float *src_u, float *src_v,
						float *boundary, float *dest_u, float *dest_v,
						float timeStep, float diff, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	//if (x > 1 && x < w-1 && y > 1 && y < h-1){
	if (checkBounds(boundary, x, y, w, h)) {
		float dt0 = (float)timeStep * float(w-2);
		float i = float(x) - dt0 * vel_u[id];
		float j = float(y) - dt0 * vel_v[id];

		dest_u[id] = diff * bilerp(src_u, i, j, w, h);
		dest_v[id] = diff * bilerp(src_v, i, j, w, h);
	}

	else {
		dest_u[id] = 0.0;
		dest_v[id] = 0.0;
	}

}

__global__ void 
Advect (float *vel_u, float *vel_v, float *src, float *boundary, float *dest,
						float timeStep, float diff, bool skipBilerp, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (checkBounds(boundary, x, y, w, h)) {
	//if (x > 1 && x < w-1 && y > 1 && y < h-1){
		float dt0 = (float)timeStep * float(w-2);
		float i = float(x) - dt0 * vel_u[id];
		float j = float(y) - dt0 * vel_v[id];

		dest[id] = diff * bilerp(src, i, j, w, h);
	}

	else {
		dest[id] = 0.0;
	}
}

__device__ float 
curl(int i, int j, float *u, float *v)
{
	float du_dy = (u[IX(i, j+1)] - u[IX(i, j-1)]) * 0.5f;
	float dv_dx = (v[IX(i+1, j)] - v[IX(i-1, j)]) * 0.5f;

	return du_dy - dv_dx;
}

__global__ void 
vorticityConfinement(float *u, float *v, float *Fvc_x, float *Fvc_y, float *_boundary, 
								     float curlAmt, float dt, int w, int h)
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);


	//if (x>1 && x<w-2 && y>1 && y<h-2){
	if (checkBounds(_boundary, x, y, w, h)) {

		// Calculate magnitude of curl(u,v) for each cell. (|w|)
		// curl[I(i, j)] = Math.abs(curl(i, j));

		// Find derivative of the magnitude (n = del |w|)
		float dw_dx = ( abs(curl(x+1,y, u, v)) - abs(curl(x-1,y, u, v)) ) * 0.5f;
		float dw_dy = ( abs(curl(x,y+1, u, v)) - abs(curl(x,y-1, u, v)) ) * 0.5f;

		// Calculate vector length. (|n|)
		// Add small factor to prevent divide by zeros.
		float length = sqrt(dw_dx * dw_dx + dw_dy * dw_dy);
		length = length + 0.000001f;
		//if (length == 0.0) length -= 0.000001f;
		// N = ( n/|n| )

		float vel = curl(x, y, u, v);
		
		// N x w
		// 0.5 = curl amount
		Fvc_y[id] = Fvc_y[id] + ((dw_dx/length) * vel * dt * curlAmt);
		Fvc_x[id] = Fvc_x[id] + ((dw_dy/length) * -vel * dt * curlAmt);
	}
	else {
		Fvc_x[id] = 0.0;
		Fvc_y[id] = 0.0;
	}
}


__global__ void 
ApplyBuoyancy( float *vel_u, float *vel_v, float *temp, float *dens, 
							   float *dest_u, float *dest_v, float ambientTemp, float buoy, float weight, 
							   float dt, int w, int h)
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);
	
	if (checkBounds(x, y, w, h)) {
		dest_u[id] = vel_u[id]; 
		dest_v[id] = vel_v[id]; 
	
		float T = temp[id];
		float Sigma = buoy;
		float Kappa = weight;
		if (T > ambientTemp) {
			float D = dens[id];

			dest_u[id] += (dt * (T - ambientTemp) * Sigma - D * Kappa) * 0;
			dest_v[id] += (dt * (T - ambientTemp) * Sigma - D * Kappa) * .1;
		}
		else {
			return;
		}
	}

}

__global__ void 
ComputeDivergence( float *u, float *v, float *boundary, float *dest, int w, int h )
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	//if (x > 2 && x < w-2 && y > 2 && y < h-2){
	if (checkBounds(x, y, w, h)){
		float vN, vS, vE, vW;

		// Find neighboring obstacles:
		float oN = boundary[4 * IX(x, y+1) + 0];
		float oS = boundary[4 * IX(x, y-1) + 0];
		float oE = boundary[4 * IX(x+1, y) + 0];
		float oW = boundary[4 * IX(x-1, y) + 0];

		// Find neighboring velocities, use center pressure for solid cells:
		vN = (oN > 0.0) ? boundary[4 * IX(x, y+1) + 1] : v[IX(x, y+1)];
		vS = (oS > 0.0) ? boundary[4 * IX(x, y-1) + 1] : v[IX(x, y-1)];
		vE = (oE > 0.0) ? boundary[4 * IX(x+1, y) + 2] : u[IX(x+1, y)];
		vW = (oW > 0.0) ? boundary[4 * IX(x-1, y) + 2] : u[IX(x-1, y)];

		//float cellSize = 1.0;
		//dest[id] = (0.5 / cellSize) * ( u[IX(x+1, y)] - u[IX(x-1, y)] + v[IX(x, y+1)] - v[IX(x, y-1)] );
		//dest[id] = 0.5 * ( (u[IX(x+1, y)] - u[IX(x-1, y)]) + (v[IX(x, y+1)] - v[IX(x, y-1)]) ) ;
		dest[id] = 0.5 * ( vE - vW + vN - vS ) / float(w-2);
	}
	else {
		return;
	}
}

__global__ void 
Jacobi( float *p, float *divergence, float *boundary, float *dest, int w, int h )
//Jacobi( float *p, float *divergence, cudaTextureObject_t txBoundary, float *dest, int w, int h )
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (checkBounds(x, y, w, h)){

		// Find neighboring obstacles:
		float oN = boundary[4 * IX(x, y+1) + 0];
		float oS = boundary[4 * IX(x, y-1) + 0];
		float oE = boundary[4 * IX(x+1, y) + 0];
		float oW = boundary[4 * IX(x-1, y) + 0];

		//float oN = tex1Dfetch<float>(txBoundary, 4 * IX(x, y+1) + 0);
		//float oS = tex1Dfetch<float>(txBoundary, 4 * IX(x, y-1) + 0);
		//float oE = tex1Dfetch<float>(txBoundary, 4 * IX(x+1, y) + 0);
		//float oW = tex1Dfetch<float>(txBoundary, 4 * IX(x-1, y) + 0);
		
		// Find neighboring pressure, use center pressure for solid cells:
		//float pC = p[id];
		float pN = (oN > 0.0) ? p[id] : p[IX(x, y+1)];
		float pS = (oS > 0.0) ? p[id] : p[IX(x, y-1)];
		float pE = (oE > 0.0) ? p[id] : p[IX(x+1, y)];
		float pW = (oW > 0.0) ? p[id] : p[IX(x-1, y)];

		//float cellSize = 1.0;
		//float Alpha = -cellSize * cellSize;
		float Alpha = -1.0;
		float bC = divergence[id];
		float InverseBeta = .25;
		dest[id] = (pW + pE + pS + pN + Alpha * bC) * InverseBeta;
	}
	else {
		return;
	}
}

__global__ void 
SubtractGradient( float *vel_u, float *vel_v, float *p, float *boundary, 
								  float *dest_u, float *dest_v, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (checkBounds(x, y, w, h)){
		// Find neighboring obstacles:
		float oN = boundary[4 * IX(x, y+1) + 0];
		float oS = boundary[4 * IX(x, y-1) + 0];
		float oE = boundary[4 * IX(x+1, y) + 0];
		float oW = boundary[4 * IX(x-1, y) + 0];

		// Use center pressure for solid cells:
		//float obstU = 0.0;
		//float obstV = 0.0;
		//float vMask = 1.0;

		// Find neighboring pressure:
		//float pN = p[IX(x, y+1)];
		//float pS = p[IX(x, y-1)];
		//float pE = p[IX(x+1, y)];
		//float pW = p[IX(x-1, y)];
		//float pC = p[id];
		
		//if (oN > 0.0) {pN = p[id]; obstV = boundary[4 * IX(x, y+1) + 1]; vMask = 0.0; }
		//if (oS > 0.0) {pS = p[id]; obstV = boundary[4 * IX(x, y-1) + 1]; vMask = 0.0; }
		//if (oE > 0.0) {pE = p[id]; obstU = boundary[4 * IX(x+1, y) + 2]; vMask = 0.0; }
		//if (oW > 0.0) {pW = p[id]; obstU = boundary[4 * IX(x+1, y) + 2]; vMask = 0.0; }

		float pN = (oN > 0.0) ? p[id] : p[IX(x, y+1)]; 
		float pS = (oS > 0.0) ? p[id] : p[IX(x, y-1)];
		float pE = (oE > 0.0) ? p[id] : p[IX(x+1, y)];
		float pW = (oW > 0.0) ? p[id] : p[IX(x-1, y)];

		float obstV = (oN > 0.0) ? boundary[4 * IX(x, y+1) + 1] : 
					  (oS > 0.0) ? boundary[4 * IX(x, y-1) + 1] : 0.0; 
		float obstU = (oE > 0.0) ? boundary[4 * IX(x+1, y) + 2] : 
					  (oW > 0.0) ? boundary[4 * IX(x+1, y) + 2] : 0.0; 
		float vMask = (oN > 0.0 || oS > 0.0 || oE > 0.0 || oW > 0.0) ? 0.0 : 1.0;

		// Enforce the free-slip boundary condition:
		float old_u = vel_u[id];
		float old_v = vel_v[id];

		//float cellSize = 1.0;
		//float GradientScale = 1.125 / cellSize;
		float GradientScale = 0.5 * float(w-2);
		float grad_u = (pE - pW) * GradientScale;
		float grad_v = (pN - pS) * GradientScale;
		
		float new_u = old_u - grad_u;
		float new_v = old_v - grad_v;

		obstU = 0;
		obstV = 0;
		dest_u[id] = (vMask * new_u) + obstU;
		dest_v[id] = (vMask * new_v) + obstV;
	}
	else {
		dest_u[id] = 0.0;
		dest_v[id] = 0.0;
	}
}


__global__ void 
Diffusion(float *_chem, float *_lap, float *_boundary, float _difConst, float xLen, float yLen, float dt, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	// have to do this check for non-powers of 2 to work...?
	//if (checkBounds(_boundary, x, y, w, h)) {
	if (checkBounds(x, y, w, h)) {

		// constants
		//float xLength = (float)x/100.0; //gray-scott
		//float xLength = (float)x/1.21212; //barkley model
		float xLength = (float)x/xLen;
		float yLength = (float)y/yLen;
		float dx = (float)xLength/(float)x;
		float dy = (float)yLength/(float)y;
		float alpha = (float)(_difConst * dt / (float)(dx*dy));

		_lap[id] = (float)(-4.0f * _chem[id]) + (float)(_chem[IX(x+1,y)] + _chem[IX(x-1,y)] + _chem[IX(x,y+1)] + _chem[IX(x,y-1)]);
		_lap[id] = (float)_lap[id]*alpha;
	}
	else {
		return;
	}
}

__global__ void 
AddLaplacian( float *_chem, float *_lap, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	_chem[id] += _lap[id];
}

__global__ void 
React( float *_chemA, float *_chemB, float F, float k, float e, int rdEquation, float *_boundary, float dt, int w, int h) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (checkBounds(_boundary, x, y, w, h)) {
		//float A = _chemA[id];
		//float B = _chemB[id];

		// Gray-Scott
		//float F = 0.05;
		//float k = 0.0675;
		//float F = 0.0140;
		//float k = 0.0490;
		//float F = 0.0545;
		//float k = 0.062;
		//float F = F_input[id]&0xff/255;
		//F = fitRange(F, 0.0, 1.0, 0.014, 0.066);
		//
		//float k = 1.0 - (F_input[id]&0xff/255);
		//k = fitRange(k, 0.0, 1.0, 0.05, 0.068); 

		//if (A > 1.) A = 1.; if (B > 1.) B = 1.;

		float A = (_chemA[id] > 1.0) ? 1.0 : _chemA[id];
		float B = (_chemB[id] > 1.0) ? 1.0 : _chemB[id];

		// Barkley Model
		if (rdEquation == 1) {
			float reactionA = A * (1-A) * ((A- (B+k)/F) / e) * dt;
			float reactionB = (A - B) * dt;
			_chemA[id] += (dt * reactionA);
			_chemB[id] += (dt * reactionB);
			return;
		}
		//// Barkley Turbulence
		//else if (rdEquation == 2) {
		//	float reactionA = A * (1-A) * ((A- (B+k)/F) / e) * dt;
		//	float reactionB = (A*A*A - B) * dt;
		//	_chemA[id] += (dt * reactionA);
		//	_chemB[id] += (dt * reactionB);
		//	return;
		//}
		// Gray-Scott
		else {
			float reactionA = -A * (B*B) + (F * (1.0-A));
			float reactionB = A * (B*B) - (F+k)*B;
			_chemA[id] += (dt * reactionA);
			_chemB[id] += (dt * reactionB);
			return;
		}
	}
	else {
		_chemA[id] *= -1.0;
		_chemB[id] *= -1.0;
	}
}


////////////////////////////////////////////////////////////////////////
//// Wavelet Turbulence Kernels
////////////////////////////////////////////////////////////////////////
//__device__ float
//minDx(int x, int y, float *input, int w, int h) {
//	//const int index = x + y * res[0] + z * res[0] * res[1];
//	const int maxx = w-2;
//	const float FLT_MAX = 1.70141e+38;
//
//	// get grid values
//	float center = input[IX(x,y)];
//	float left  = (x <= 1)    ? FLT_MAX : input[IX(x-1,y)];
//	float right = (x >= maxx) ? FLT_MAX : input[IX(x+1,y)];
//
//	const float dx = w;
//
//	// get all the derivative estimates
//	float dLeft   = (x <= 1)     ? FLT_MAX : (center - left) * dx;
//	float dRight  = (x >= maxx)  ? FLT_MAX : (right - center) * dx;
//	float dCenter = (x <= 1 || x >= maxx) ? FLT_MAX : (right - left) * dx * 0.5f;
//
//	// if it's on a boundary, only one estimate is valid
//	if (x <= 1) return dRight;
//	if (x >= maxx) return dLeft;
//
//	// if it's not on a boundary, get the smallest one
//	float finalD;
//	finalD = (abs(dCenter) < abs(dRight)) ? dCenter : dRight;
//	finalD = (abs(finalD)  < abs(dLeft))  ? finalD  : dLeft;
//
//	return finalD;
//}
//
//__device__ float
//minDy(int x, int y, float *input, int w, int h) {
//	//const int index = x + y * res[0] + z * res[0] * res[1];
//	const int maxy = h-2;
//	const float FLT_MAX = 1.70141e+38;
//
//	// get grid values
//	float center = input[IX(x,y)];
//	float down  = (y <= 1) ? FLT_MAX : input[IX(x,y-1)];
//	float up = (y >= maxy) ? FLT_MAX : input[IX(x,y+1)];
//
//	const float dx = h; // only for square domains
//
//	// get all the derivative estimates
//	float dDown   = (y <= 1)  ? FLT_MAX : (center - down) * dx;
//	float dUp  = (y >= maxy)  ? FLT_MAX : (up - center) * dx;
//	float dCenter = (y <= 1 || y >= maxy) ? FLT_MAX : (up - down) * dx * 0.5f;
//
//	// if it's on a boundary, only one estimate is valid
//	if (y <= 1) return dUp;
//	if (y >= maxy) return dDown;
//
//	// if it's not on a boundary, get the smallest one
//	float finalD = (abs(dCenter) < abs(dUp)) ? dCenter : dUp;
//	finalD = (abs(finalD) < abs(dDown)) ? finalD : dDown;
//
//	return finalD;
//}
//
//
//__global__ void
//ComputeEigenvalues(float *_tcU, float *_tcV, float *_eigMin, float *_eigMax, int w, int h) {
//	int x = getX(w);
//	int y = getY(h);
//	int id = IX(x,y);
//
//	float A[2][2] = {
//		{ minDx(x, y, _tcU, w, h), minDx(x, y, _tcV, w, h) },
//		{ minDy(x, y, _tcU, w, h), minDy(x, y, _tcV, w, h) }
//	};
//
//	const float T = A[0][0] + A[1][1];
//	const float D = (A[0][0]*A[1][1]) - (A[0][1]*A[1][0]);
//
//	_eigMin[id] = (T*0.5) - pow((T*T)/(4.0-D), 0.5);
//	_eigMax[id] = (T*0.5) + pow((T*T)/(4.0-D), 0.5);
//
//}
//
//__global__ void
//ComputeEnergy(float *_energy, float *_xvel, float *_yvel, int w, int h) {
//	int x = getX(w);
//	int y = getY(h);
//	int id = IX(x,y);
//
//	_energy[id] = 0.5 * (_xvel[id]*_xvel[id] + _yvel[id]*_yvel[id]);
//
//}
//
//__global__ void
//CopyObstacles(float *inObstacles, unsigned char *_obstacles, int w, int h) {
//	int x = getX(w);
//	int y = getY(h);
//	int id = IX(x,y);
//
//	if (inObstacles[4 * id + 0] > 0) {
//		_obstacles[id] = 1;
//	}
//	else {
//		_obstacles[id] = 0;
//	}
//}
//
//__global__ void
//IterateEnergy(float *_energy, unsigned char *_obstacles, int w, int h){ 
//	int x = getX(w);
//	int y = getY(h);
//	int id = IX(x,y);
//
//	const int MARCHED = 2;
//	const int RETIRED = 4;
//	const int slabSizeSm = w * h;
//
//	if (checkBounds(x, y, w, h)) {
//		if (_obstacles[id] && _obstacles[id] != RETIRED) {
//            float sum = 0.0f;
//            int valid = 0;
//
//            if (!_obstacles[id + 1] || _obstacles[id + 1] == RETIRED) {
//              sum += _energy[id + 1];
//              valid++;
//            }
//            if (!_obstacles[id - 1] || _obstacles[id - 1] == RETIRED) {
//              sum += _energy[id - 1];
//              valid++;
//            }
//            if (!_obstacles[id + w] || _obstacles[id + w] == RETIRED) {
//              sum += _energy[id + w];
//              valid++;
//            }
//            if (!_obstacles[id - w] || _obstacles[id - w] == RETIRED) {
//              sum += _energy[id - w];
//              valid++;
//            }
//            if (!_obstacles[id + slabSizeSm] || _obstacles[id + slabSizeSm] == RETIRED) {
//              sum += _energy[id + slabSizeSm];
//              valid++;
//            }
//            if (!_obstacles[id - slabSizeSm] || _obstacles[id - slabSizeSm] == RETIRED) {
//              sum += _energy[id - slabSizeSm];
//              valid++;
//            }
//            if (valid > 0) {
//              _energy[id] = sum / float(valid);
//              _obstacles[id] = MARCHED;
//            }
//          }
//	}
//}
//
//__global__ void
//CheckMarchedObstacles(unsigned char *_obstacles, int w, int h) {
//	int x = getX(w);
//	int y = getY(h);
//	int id = IX(x,y);
//
//	const int MARCHED = 2;
//	const int RETIRED = 4;
//
//	if (checkBounds(x, y, w, h)) {
//		if (_obstacles[id] == MARCHED) {
//			_obstacles[id] = RETIRED;
//		}
//	}
//
//}
//
//__global__ void
//DownsampleNeumann(float *from, float *to, int n, int stride, int w, int h) {
//	int x = getX(w);
//	int y = getY(h);
//	int id = IX(x,y);
//
//	// if these values are not local incorrect results are generated
//	float downCoeffs[32] = { 0.000334f,-0.001528f, 0.000410f, 0.003545f,-0.000938f,-0.008233f, 0.002172f, 0.019120f,
//							-0.005040f,-0.044412f, 0.011655f, 0.103311f,-0.025936f,-0.243780f, 0.033979f, 0.655340f,
//							0.655340f, 0.033979f,-0.243780f,-0.025936f, 0.103311f, 0.011655f,-0.044412f,-0.005040f,
//							0.019120f, 0.002172f,-0.008233f,-0.000938f, 0.003546f, 0.000410f,-0.001528f, 0.000334f };
//
//	const float *const aCoCenter= &downCoeffs[16];
//	for (int i = 0; i < n / 2; i++) {
//		to[i * stride] = 0;
//		for (int k = 2 * i - 16; k < 2 * i + 16; k++) { 
//			// handle boundary
//			float fromval; 
//			if (k < 0) {
//				fromval = from[0];
//			} else if(k > n - 1) {
//				fromval = from[(n - 1) * stride];
//			} else {
//				fromval = from[k * stride]; 
//			} 
//			to[i * stride] += aCoCenter[k - 2 * i] * fromval; 
//		}
//	}
//}

#endif