/* 
	CUDA kernels and functions
	Kurt Kaminski 2016
*/


#ifndef __FLUID_SIM__
#define __FLUID_SIM__

#include <iostream>
#include <map>
#include <string>

#include "FluidKernels.cu"

//float *testArray;
//
//void testing(int w, int h, int* in, int* out)
//{
//	cudaMalloc((void**)&testArray, sizeof(float)*w*h);
//
//	dim3 block(16, 16, 1);
//	int extraX = 0;
//	int extraY = 0;
//	if (w % block.x > 0)
//		extraX = 1;
//	if (h % block.y > 0)
//		extraY = 1;
//	dim3 grid((w / block.x) + extraX , (h / block.y) + extraY , 1);
//	sampleKernel <<< grid, block >>> (in, w, h,
//										out, w, h);
//}

// Data
float *chemA, *chemA_prev, *chemB, *chemB_prev, *laplacian;
float *vel[2], *vel_prev[2];
float *pressure, *pressure_prev;
float *temperature, *temperature_prev;
float *divergence;
float *boundary;
float *newObstacle; //incoming obstacles and velocities
float *mouse, *mouse_old;
float *res, *globals, *advectionConstants, *rdConstants;

// Global constants
dim3 grid, threads;
int dimX, dimY;
float dt;
int nDiff;
int nReact;
int nJacobi;

// Advection constants
float velDiff;
float tempDiff;
float densDiff;
float curlAmt;
float buoy;
float weight;
float diff;
float visc;
float force;
float source_density;
float source_temp;

// RD constants
float F;
float k;
float dA;
float dB;
float xLen;
float yLen;
int rdEquation;
float e;

///////////////////////////////////////////////////////////////////////////////
// Set constants from UI
///////////////////////////////////////////////////////////////////////////////
void setFluidConstants()
{
	// globals[] = {dt, nDiff, nReact, nJacobi}
	dt = globals[0];
	nDiff = (int)globals[1];
	nReact = (int)globals[2];
	nJacobi = (int)globals[3];

	// advectionConstants[] = {velDiff, tempDiff, densDiff, curl, buoyancy, weight}
	velDiff = advectionConstants[0];
	tempDiff = advectionConstants[1];
	densDiff = advectionConstants[2];
	curlAmt = advectionConstants[3];
	buoy = advectionConstants[4];
	weight = advectionConstants[5];

	// rdConstants[]
	F = rdConstants[0];
	k = rdConstants[1];
	dA = rdConstants[2];
	dB = rdConstants[3];
	xLen = (int)rdConstants[4];
	yLen = (int)rdConstants[5];
	rdEquation = (int)rdConstants[6];
	e = rdConstants[7];
}

///////////////////////////////////////////////////////////////////////////////
// Clear arrays
///////////////////////////////////////////////////////////////////////////////
void clearArrays()
{
	cudaSetDevice(0);

	for (int i=0; i<2; i++){
	  ClearArray<<<grid,threads>>>(vel[i], 0.0, dimX, dimY);
	  ClearArray<<<grid,threads>>>(vel_prev[i], 0.0, dimX, dimY);
	}

	ClearArray<<<grid,threads>>>(chemA, 1.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(chemA_prev, 1.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(chemB, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(chemB_prev, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(boundary, 0.0, dimX, dimY);

	ClearArray<<<grid,threads>>>(pressure, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(pressure_prev, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(temperature, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(temperature_prev, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(divergence, 0.0, dimX, dimY);

	printf("Fluid::clearArrays(): Cleared GPU arrays.\n");
}

///////////////////////////////////////////////////////////////////////////////
// Set up global variables
///////////////////////////////////////////////////////////////////////////////
void initGlobals(std::map<std::string, const TCUDA_ParamInfo*> &nodes)
{
	// Resolution
	res = (float*)malloc(sizeof(float)*nodes["fluidRes"]->chop.numChannels);
	res = (float*)nodes["fluidRes"]->data;
	
	dimX = res[0];
	dimY = res[1];

	threads = dim3(16,16);
	grid.x = (dimX + threads.x - 1) / threads.x;
	grid.y = (dimY + threads.y - 1) / threads.y;

	printf("Threads dim: %d x %d\n", threads.x, threads.y);
	printf("Grid dim: %d x %d\n", grid.x, grid.y);

	// Allocate mouse array
	mouse = (float*)malloc(sizeof(float)*nodes["mouse"]->chop.numChannels);
	mouse_old = (float*)malloc(sizeof(float)*nodes["mouse"]->chop.numChannels);

	// Local mouse pointer points to CHOP node
	mouse = (float*)nodes["mouse"]->data;
	for (int i = 0; i < nodes["mouse"]->chop.numChannels; i++){
		mouse_old[0]=mouse[1];
	}

	// Allocate arrays for local constants
	globals = (float*)malloc(sizeof(float)*nodes["globals"]->chop.numChannels);
	advectionConstants = (float*)malloc(sizeof(float)*nodes["advection"]->chop.numChannels);
	rdConstants = (float*)malloc(sizeof(float)*nodes["rd"]->chop.numChannels);

	// Local constants pointers points to CHOP nodes
	globals = (float*)nodes["globals"]->data;
	advectionConstants = (float*)nodes["advection"]->data;
	rdConstants = (float*)nodes["rd"]->data;

	// Set fluid constants
	setFluidConstants();

	printf("initParameters(): done.\n");
}

///////////////////////////////////////////////////////////////////////////////
// Initialize memory
///////////////////////////////////////////////////////////////////////////////
void initMemory()
{
	int size = dimX * dimY;

	// Allocate GPU memory
	cudaMalloc((void**)&chemA, sizeof(float)*size);
	cudaMalloc((void**)&chemA_prev, sizeof(float)*size);
	cudaMalloc((void**)&chemB, sizeof(float)*size);
	cudaMalloc((void**)&chemB_prev, sizeof(float)*size);
	cudaMalloc((void**)&laplacian, sizeof(float)*size);
	cudaMalloc((void**)&boundary, sizeof(float)*size * 4);

	for (int i=0; i<2; i++){
		cudaMalloc((void**)&vel[i], sizeof(float)*size);
		cudaMalloc((void**)&vel_prev[i], sizeof(float)*size);
	}

	cudaMalloc((void**)&pressure, sizeof(float)*size );
	cudaMalloc((void**)&pressure_prev, sizeof(float)*size );
	cudaMalloc((void**)&temperature, sizeof(float)*size );
	cudaMalloc((void**)&temperature_prev, sizeof(float)*size );
	cudaMalloc((void**)&divergence, sizeof(float)*size );

	clearArrays();
}

///////////////////////////////////////////////////////////////////////////////
// Initialize
///////////////////////////////////////////////////////////////////////////////
void init(const TCUDA_ParamInfo **_params, const TCUDA_ParamInfo *_output, std::map<std::string, const TCUDA_ParamInfo*> &nodes)
{
	initGlobals(nodes);
	initMemory();

	printf("Fluid::init(): Dimensions = %d x %d --\n", dimX, dimY);
	printf("Fluid::init(): Allocated GPU memory.\n");
}


void getFromUI(float* inDensity, float* inObstacle)
{

	//ClearArray<<<grid,threads>>>(chemA_prev, 0.0, dimX, dimY);
	//ClearArray<<<grid,threads>>>(chemB_prev, 0.0, dimX, dimY);

	// Get pointer for incoming obstacle and velocity data
	boundary = inObstacle;

	// Apply incoming density and temperature
	// bgra == 0,1,2,3 == temperature, chemA, chemB == index
	AddFromUI<<<grid,threads>>>(temperature_prev, inDensity, 0, dt, dimX, dimY);
	AddFromUI<<<grid,threads>>>(chemB_prev, inDensity, 1, dt, dimX, dimY);
	AddFromUI<<<grid,threads>>>(chemA_prev, inDensity, 2, dt, dimX, dimY);
	//SetFromUI<<<grid,threads>>>(chemA, chemB, (float*)nodes["density"]->data, dimX, dimY);
	
	// Apply obstacle velocity
	AddObstacleVelocity<<<grid,threads>>>(vel_prev[0], vel_prev[1], inObstacle, dt, dimX, dimY);

}

///////////////////////////////////////////////////////////////////////////////
// Advection reaction-diffusion
///////////////////////////////////////////////////////////////////////////////
void reactDiffAdvect(float* inObstacle)
{

	// Naive ARD-----------------------
	AddSource<<<grid,threads>>>(chemB, chemB_prev, dt, dimX, dimY);
	//AddSource<<<grid,threads>>>(_chemA, _chemA0, dt, dimX, dimY);
	//chemA_prev = chemA;
	//chemB_prev = chemB;

	for (int i = 0; i < nDiff; i++){
		Diffusion<<<grid,threads>>>(chemA_prev, laplacian, inObstacle, dA, xLen, yLen, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(chemA_prev, laplacian, dimX, dimY);
		//SetBoundary<<<grid,threads>>>(0, chemA_prev, inObstacle, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		Diffusion<<<grid,threads>>>(chemB_prev, laplacian, inObstacle, dB, xLen, yLen, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(chemB_prev, laplacian, dimX, dimY);
		//SetBoundary<<<grid,threads>>>(0, chemB_prev, inObstacle, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		for (int j = 0; j < nReact; j++){
			React<<<grid,threads>>>( chemA_prev, chemB_prev, F, k, e, rdEquation, inObstacle, dt, dimX, dimY );
		}
	}

	// Density advection: chemA
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], chemA_prev, inObstacle, chemA,
							dt, densDiff, true, dimX, dimY);

	// Density advection: chemB
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], chemB_prev, inObstacle, chemB,
							dt, densDiff, true, dimX, dimY);

	SWAP ( chemA_prev, chemA );
	SWAP ( chemB_prev, chemB );
}

///////////////////////////////////////////////////////////////////////////////
// Simulate
///////////////////////////////////////////////////////////////////////////////
void step(float* inDensity, float* inObstacle)
{
	//cudaSetDevice(0);

	setFluidConstants();

	// Velocity advection
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], vel_prev[0], vel_prev[1],
								inObstacle, vel[0], vel[1], 
								dt, velDiff, dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);

	// Temperature advection
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], temperature_prev, inObstacle, temperature,
							dt, tempDiff, false, dimX, dimY);
	SWAP(temperature_prev, temperature);

	// Vorticity Confinement
	vorticityConfinement<<<grid,threads>>>( vel[0], vel[1], vel_prev[0], vel_prev[1], inObstacle, 
											curlAmt, dt, dimX, dimY);
		
	float Tamb = 0.0;
	//getSum<<<grid,threads>>>(temperature_prev, Tamb, dimX, dimY);
	//Tamb /= (float(dimX) * float(dimY));
	ApplyBuoyancy<<<grid,threads>>>(vel_prev[0], vel_prev[1], temperature_prev, chemB,
									vel[0], vel[1], Tamb, buoy, weight, dt, dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);

	// Apply impulses
	getFromUI(inDensity, inObstacle);

	// Reaction-Diffusion and Density advection
	reactDiffAdvect(inObstacle);

	// Compute divergence
	ComputeDivergence<<<grid,threads>>>( vel_prev[0], vel_prev[1], inObstacle, divergence, dimX, dimY );

	// Pressure solve
	ClearArray<<<grid,threads>>>(pressure_prev, 0.0, dimX, dimY);
	for (int i=0; i<nJacobi; i++){
		
		Jacobi<<<grid,threads>>>(pressure_prev, divergence, inObstacle, pressure, dimX, dimY);
		//Jacobi<<<grid,threads>>>(pressure_prev, divergence, txObstacle, pressure, dimX, dimY);
		SWAP(pressure_prev, pressure);

	}

	// Subtract pressure gradient from velocity
	SubtractGradient<<<grid,threads>>>( vel_prev[0], vel_prev[1], pressure_prev, inObstacle, 
										vel[0], vel[1], dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);


	//if (_wTurb) {
	//	float dx = 1.0/float(dimX);
	//	_wTurb->step(dt/dx, vel[0], vel[1], inObstacle);
	//}

}

void makeColor(float* output)
{
	MakeColor<<<grid,threads>>>(chemA, chemB, vel[0], vel[1], 
								output, dimX, dimY);
}

#endif