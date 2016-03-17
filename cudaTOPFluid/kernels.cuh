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
	return x;
}

__device__ int 
getY(int h) 
{
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	//if (y >= h) y = 0; if (y < 0) y = h-1;
	return y;
}


// Returns true if within the bounds of both the container edges and a user-defined boundary
__device__ bool
checkBounds(int *_boundary, int x, int y, int w, int h)
{
	if (x > 1 && x < w-2 && y > 1 && y < h-2 && _boundary[IX(x,y)] < 1 ){
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
	dest[4*id]=b;
	dest[4*id+1] = g;
	dest[4*id+2] = r;
	dest[4*id+3] = a;

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

	if (x>x_coord-5 && x<x_coord+5 && y>y_coord-5 && y<y_coord+5){
		// if (x == x_coord && y==y_coord){
		field[id] += value;
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


__global__ void TEST (float *test, int w, int h)
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	//test[0][id] = .5;
	//test[1][id] = .5;
	test[id] = .5;
}

__device__ float
bilerp(float *src, float i, float j, int w, int h)
{
	int i0, j0, i1, j1;
	float s0, t0, s1, t1;

	// fit bounds
	if (i < 0.5f) i = 0.5f;
	if (i > float(w)-2.0+0.5f) i = float(w)-2.0+0.5f;
	if (j < 0.5f) j = 0.5f;
	if (j > float(h)-2.0+0.5f) j = float(h)-2.0+0.5f;
		
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

__global__ void Advect (float *vel_u, float *vel_v, float *src_u, float *src_v,
						int *boundary, float *dest_u, float *dest_v,
						float timeStep, float diff, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (x > 1 && x < w-1 && y > 1 && y < h-1){
		float dt0 = (float)timeStep * float(w-2);
		float i = float(x) - dt0 * vel_u[id];
		float j = float(y) - dt0 * vel_v[id];

		dest_u[id] = diff * bilerp(src_u, i, j, w, h);
		dest_v[id] = diff * bilerp(src_v, i, j, w, h);
	}

	if (!checkBounds(boundary, x, y, w, h)) {
		dest_u[id] = 0.0;
		dest_v[id] = 0.0;
	}

}

__global__ void Advect (float *vel_u, float *vel_v, float *src, int *boundary, float *dest,
						float timeStep, float diff, bool skipBilerp, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (x > 1 && x < w-1 && y > 1 && y < h-1){
		float dt0 = (float)timeStep * float(w-2);
		float i = float(x) - dt0 * vel_u[id];
		float j = float(y) - dt0 * vel_v[id];

		dest[id] = diff * bilerp(src, i, j, w, h);

		//if (skipBilerp) {
		//	int c_x = x - timeStep * vel_u[id];
		//	int c_y = y - timeStep * vel_v[id];
		//	dest[id] = src[IX(c_x, c_y)];
		//}
	}

	if (!checkBounds(boundary, x, y, w, h)) {
		dest[id] = 0.0;
	}
}

__device__ float curl(int i, int j, float *u, float *v)
{
	float du_dy = (u[IX(i, j+1)] - u[IX(i, j-1)]) * 0.5f;
	float dv_dx = (v[IX(i+1, j)] - v[IX(i-1, j)]) * 0.5f;

	return du_dy - dv_dx;
}

__global__ void vorticityConfinement(float *u, float *v, float *Fvc_x, float *Fvc_y, int *_boundary, 
								     float dt, int w, int h)
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	float dw_dx, dw_dy;
	float length;
	float vel;

	//if (x>1 && x<w-2 && y>1 && y<h-2){
	if (checkBounds(_boundary, x, y, w, h)) {

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
		// 0.5 = curl amount
		Fvc_x[id] += (dw_dy * -vel * dt * 0.5);
		Fvc_y[id] += (dw_dx *  vel * dt * 0.5);
	}
}


__global__ void ApplyBuoyancy( float *vel_u, float *vel_v, float *temp, float *dens, 
							   float *dest_u, float *dest_v, float ambientTemp, float dt, int w, int h)
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);
	
	if (checkBounds(x, y, w, h)) {
		dest_u[id] = vel_u[id]; 
		dest_v[id] = vel_v[id]; 
	
		float T = temp[id];
		float Sigma = 1.0;
		float Kappa = 0.05;
		if (T > ambientTemp) {
			float D = dens[id];
			float dt0 = (float)dt;

			dest_u[id] += (dt0 * (T - ambientTemp) * Sigma - D * Kappa) * 0;
			dest_v[id] += (dt0 * (T - ambientTemp) * Sigma - D * Kappa) * .01;
		}
	}

}

__global__ void ComputeDivergence( float *u, float *v, int *boundary, float *dest, int w, int h )
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (x > 2 && x < w-2 && y > 2 && y < h-2){
		float cellSize = 1.0;
		//dest[id] = (0.5 / cellSize) * ( u[IX(x+1, y)] - u[IX(x-1, y)] + v[IX(x, y+1)] - v[IX(x, y-1)] );
		//dest[id] = 0.5 * ( (u[IX(x+1, y)] - u[IX(x-1, y)]) + (v[IX(x, y+1)] - v[IX(x, y-1)]) ) ;
		dest[id] = 0.5 * ( u[IX(x+1, y)] - u[IX(x-1, y)] + v[IX(x, y+1)] - v[IX(x, y-1)] ) / float(w-2);
	}
}

__global__ void Jacobi( float *p, float *divergence, int *boundary, float *dest, int w, int h )
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (x > 1 && x < w-1 && y > 1 && y < h-1){
		// Find neighboring pressure:
		float pN = p[IX(x, y+1)];
		float pS = p[IX(x, y-1)];
		float pE = p[IX(x+1, y)];
		float pW = p[IX(x-1, y)];
		float pC = p[id];

		// Find neighboring obstacles:
		int oN = boundary[IX(x, y+1)];
		int oS = boundary[IX(x, y-1)];
		int oE = boundary[IX(x+1, y)];
		int oW = boundary[IX(x-1, y)];

		// Use center pressure for solid cells:
		if (oN > 0) pN = pC;
		if (oS > 0) pS = pC;
		if (oE > 0) pE = pC;
		if (oW > 0) pW = pC;

		float cellSize = 1.0;
		//float Alpha = -cellSize * cellSize;
		float Alpha = -1.0;
		float bC = divergence[id];
		float InverseBeta = .25;
		dest[id] = (pW + pE + pS + pN + Alpha * bC) * InverseBeta;	
		//dest[id] = (divergence[id] + (1.0*(p[IX(x+1,y)] + p[IX(x-1,y)] + p[IX(x,y+1)] + p[IX(x,y-1)]))) * .25;
	}
}

__global__ void SubtractGradient( float *vel_u, float *vel_v, float *p, int *boundary, 
								  float *dest_u, float *dest_v, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (x > 1 && x < w-1 && y > 1 && y < h-1){
		// Find neighboring pressure:
		float pN = p[IX(x, y+1)];
		float pS = p[IX(x, y-1)];
		float pE = p[IX(x+1, y)];
		float pW = p[IX(x-1, y)];
		float pC = p[id];
		
		// Find neighboring obstacles:
		int oN = boundary[IX(x, y+1)];
		int oS = boundary[IX(x, y-1)];
		int oE = boundary[IX(x+1, y)];
		int oW = boundary[IX(x-1, y)];

		// Use center pressure for solid cells:
		float obstV = 0.0;
		float vMask = 1.0;
		
		if (oN > 0) { pN = pC; vMask = 0.0; }
		if (oS > 0) {pS = pC; vMask = 0.0; }
		if (oE > 0) {pE = pC; vMask = 0.0; }
		if (oW > 0) {pW = pC; vMask = 0.0; }
	    
		// Enforce the free-slip boundary condition:
		float old_u = vel_u[id];
		float old_v = vel_v[id];

		// GradientScale is 1.125 / CellSize
		float cellSize = 1.0;
		//float GradientScale = 1.125 / cellSize;
		float GradientScale = 0.5 * float(w-2);
		float grad_u = (pE - pW) * GradientScale;
		float grad_v = (pN - pS) * GradientScale;
		
		float new_u = old_u - grad_u;
		float new_v = old_v - grad_v;

		dest_u[id] = (vMask * new_u) + obstV;
		dest_v[id] = (vMask * new_v) + obstV;
	
	}
}


__global__ void 
Diffusion(float *_chem, float *_lap, int *_boundary, float _difConst, float dt, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	// have to do this check for non-powers of 2 to work...?
	if (checkBounds(_boundary, x, y, w, h)) {

		// constants
		float xLength = (float)x/100.0;
		float dx = (float)xLength/(float)x;
		float alpha = (float)(_difConst * dt / (float)(dx*dx));

		int n1 = getX(x-1);
		int n2 = getX(x+1);
		int n3 = getY(y-1);
		int n4 = getY(y+1);
		_lap[id] = (float)(-4.0f * _chem[id]) + (float)(_chem[IX(x+1,y)] + _chem[IX(x-1,y)] + _chem[IX(x,y+1)] + _chem[IX(x,y-1)]);
		_lap[id] = (float)_lap[id]*alpha;
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

__global__ void React( float *_chemA, float *_chemB, int *F_input, float *rd, int *_boundary, float dt, int w, int h) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	if (checkBounds(_boundary, x, y, w, h)) {
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
		float F = rd[0];
		float k = rd[1];
		
		float A = _chemA[id];
		float B = _chemB[id];

		float reactionA = -A * (B*B) + (F * (1.0-A));
		float reactionB = A * (B*B) - (F+k)*B;
		_chemA[id] += (dt * reactionA);
		_chemB[id] += (dt * reactionB);
	}
	else {
		_chemA[id] *= -1.0;
		_chemB[id] *= -1.0;
	}
}