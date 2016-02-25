/* Shared Use License: This file is owned by Derivative Inc. (Derivative) and
 * can only be used, and/or modified for use, in conjunction with 
 * Derivative's TouchDesigner software, and only if you are a licensee who has
 * accepted Derivative's TouchDesigner license or assignment agreement (which
 * also govern the use of this file).  You may share a modified version of this
 * file with another authorized licensee of Derivative's TouchDesigner software.
 * Otherwise, no redistribution or sharing of this file, with or without
 * modification, is permitted.
 */

#include "TCUDA_Types.h"
#include "defines.h"
#include <stdio.h>

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
ClearArray(float *field, float value, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	field[id] = value;
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
LinSolve(float *field, float *field0, float a, float c, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

    field[id] = (float)(field0[id] + ((float)a*(field[IX(x-1,y)] + field[IX(x+1,y)] + field[IX(x,y-1)] + field[IX(x,y+1)]))) / c;
}

__global__ void 
Diffusion(float *_chem, float *_lap, float _difConst, float dt, int w, int h) 
{
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);
	int size = (w*h)-2;

	if (x>1 && x<size && y>1 && y<size){
		int n1 = id + 1;
		int n2 = id - 1;
		int n3 = id + w;
		int n4 = id - w;

		// constants
		// float xLength = (float)DIM/100.0;
		float xLength = 2.56f;
		// float dx = (float)xLength/DIM;
		float dx = 0.01f;
		float alpha = (float)(_difConst * dt / (float)(dx*dx));

		_lap[id] = (float)(-4.0f * _chem[id]) + (float)(_chem[n1] + _chem[n2] + _chem[n3] + _chem[n4]);
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

__global__ void React( float *_chemA, float *_chemB, float dt, int w, int h) {
	int x = getX(w);
	int y = getY(h);
	int id = IX(x,y);

	int size = (w*h)-2;
	if (x>1 && x<w-1 && y>1 && y<h-1){
		float F = 0.05;
		float k = 0.0675;
		float A = _chemA[id];
		float B = _chemB[id];

		float reactionA = -A * (B*B) + (F * (1.0-A));
		float reactionB = A * (B*B) - (F+k)*B;
		_chemA[id] += (dt * reactionA);
		_chemB[id] += (dt * reactionB);
	}
	else {
		_chemA[id] = 0.0;
		_chemB[id] = 0.0;
	}
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


