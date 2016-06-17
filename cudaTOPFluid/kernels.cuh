/* 
	CUDA kernels and functions
	Kurt Kaminski 2016
*/

#ifndef __KERNELS__
#define __KERNELS__


#include <cuda_runtime_api.h>
//#include <math.h>

__device__ int 
clamp(int i);

__device__ float 
clamp(float i, float min, float max);

__device__ float
fitRange(float valueIn, float baseMin, float baseMax, float limitMin, float limitMax);

// Get 1d index from 2d coords
__device__ int 
IX(int x, int y);

__device__ int 
getX(int w);

__device__ int 
getY(int h);


// Returns true if within the bounds of both the container edges and a user-defined boundary
__device__ bool
checkBounds(float *_boundary, int x, int y, int w, int h);

__device__ bool
checkBounds(int x, int y, int w, int h);

// Functions for converting to/from a int (4 bytes, 1 byte per RGBA, which are in the range 0-255)
// to 4 floats in the range 0.0-1.0 
// Note how the data is stored in BGRA format due to how its stored on the GPU.
__device__ int 
rgbaToInt(float r, float g, float b, float a);

__device__ void 
intToRgba(int pixel, float &r, float &g, float &b, float &a);

__device__ void
rgbaToColor(float *dest, int id, float r, float g, float b, float a);

// Set boundary conditions
__device__ void 
set_bnd( int b, int x, int y, float *field, float *boundary, int w, int h);

__global__ void 
DrawSquare( float *field, float value, int w, int h );

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
SetBoundary( int b, float *field, float *boundary, int w, int h );


__global__ void 
getSum( float *_data, float _sum, int w, int h );

__global__ void 
ClearArray(float *field, float value, int w, int h);

__global__ void 
ClearArray(int *field, float value, int w, int h);

// How can I template these?
__global__ void 
AddFromUI ( float *field, float value, float dt, int x_coord, int y_coord, int w, int h );

__global__ void 
AddFromUI ( float *field, float *valueUI, int index, float dt, int w, int h );

__global__ void 
AddObstacleVelocity ( float *u, float *v, float *obstacle, float dt, int w, int h );

__global__ void 
SetFromUI ( float *A, float *B, float *valueUI, int w, int h );

__global__ void
MakeSource(int *src, float *dest, int w, int h);

__global__ void
MakeSource(int *src, int *dest, int w, int h);

__global__ void 
AddSource(float *field, float *source, float dt, int w, int h);

__global__ void
MakeColor(float *src, int *dest, int w, int h);

__global__ void
MakeColor(float *src0, float *src1, float *src2, float *dest, int w, int h);

__global__ void
MakeColor(float *src0, float *src1, float *src2, float *src3, float *dest, int w, int h);

__global__ void
MakeColorLong(	float *r1, float *g1, float *b1, float *a1, 
				float *r2, float *g2, float *b2, float *a2,
				float *dest, int w, int h, int stride);

__global__ void 
TEST (float *test, int w, int h);

__device__ float
bilerp(float *src, float i, float j, int w, int h);

__global__ void Advect (float *vel_u, float *vel_v, float *src_u, float *src_v,
						float *boundary, float *dest_u, float *dest_v,
						float timeStep, float diff, int w, int h);

__global__ void Advect (float *vel_u, float *vel_v, float *src, float *boundary, float *dest,
						float timeStep, float diff, bool skipBilerp, int w, int h);

__device__ float 
curl(int i, int j, float *u, float *v);

__global__ void 
vorticityConfinement(float *u, float *v, float *Fvc_x, float *Fvc_y, float *_boundary, 
								     float curlAmt, float dt, int w, int h);


__global__ void 
ApplyBuoyancy( float *vel_u, float *vel_v, float *temp, float *dens, 
							   float *dest_u, float *dest_v, float ambientTemp, float buoy, float weight, 
							   float dt, int w, int h);

__global__ void 
ComputeDivergence( float *u, float *v, float *boundary, float *dest, int w, int h );

__global__ void 
Jacobi( float *p, float *divergence, float *boundary, float *dest, int w, int h );

__global__ void 
SubtractGradient( float *vel_u, float *vel_v, float *p, float *boundary, 
								  float *dest_u, float *dest_v, int w, int h);

__global__ void 
Diffusion(float *_chem, float *_lap, float *_boundary, float _difConst, float xLen, float yLen, float dt, int w, int h);

__global__ void 
AddLaplacian( float *_chem, float *_lap, int w, int h);

__global__ void 
React( float *_chemA, float *_chemB, float F, float k, float e, int rdEquation, float *_boundary, float dt, int w, int h);

#endif