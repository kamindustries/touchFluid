/* 
	CUDA kernels and functions
	Kurt Kaminski 2016
*/

#include <cuda_runtime_api.h>
#include "Fluid.cuh"
#include "kernels.cuh"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Fluid::Fluid(){
	// Set initial values
	// Global constants
	dt = 0.1f;
	nDiff = 2;
	nReact = 1;
	nJacobi = 30;

	// Advection constants
	velDiff = .9999f;
	tempDiff = .99f;
	densDiff = 1.0f;
	curlAmt = 1.0f;
	buoy = 1.0f;
	weight = 0.05f;
	diff = 0.00001f;
	visc = 0.000001f;
	force = 300.;
	source_density = 10.0;
	source_temp = 2.0;

	// RD constants
	F = 0.05f;
	k = 0.0675f;
	dA = 0.0002f; // gray-scott
	dB = 0.00001f;
	xLen = 100.0f;
	yLen = 100.0f;
	rdEquation = 0;
	e = .02;
}

Fluid::~Fluid()
{
	cudaFree(chemA);
	cudaFree(chemA_prev);
	cudaFree(chemB);
	cudaFree(chemB_prev);
	cudaFree(laplacian);
	cudaFree(pressure);
	cudaFree(pressure_prev);
	cudaFree(temperature);
	cudaFree(temperature_prev);
	cudaFree(divergence);
	cudaFree(boundary);
	for (int i=0; i<2; i++){
		cudaFree(vel[i]);
		cudaFree(vel_prev[i]);
	}
}

void Fluid::init(int xRes, int yRes)
{
	dimX = xRes;
	dimY = yRes;
	int size = dimX * dimY;

	threads = dim3(16,16);
	grid.x = (dimX + threads.x - 1) / threads.x;
	grid.y = (dimY + threads.y - 1) / threads.y;

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

	printf("Fluid::init(): Dimensions = %d x %d --\n", dimX, dimY);
	printf("Fluid::init(): Allocated GPU memory.\n");

}

void Fluid::clearArrays()
{
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


void Fluid::getFromUI(float* inDensity, float* inObstacle)
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
void Fluid::reactDiffAdvect(float* inObstacle)
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
void Fluid::step(float* inDensity, float* inObstacle)
{
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
		SWAP(pressure_prev, pressure);
	}

	// Subtract pressure gradient from velocity
	SubtractGradient<<<grid,threads>>>( vel_prev[0], vel_prev[1], pressure_prev, inObstacle, 
										vel[0], vel[1], dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);

}

///////////////////////////////////////////////////////////////////////////////
// Send data to TD texture (2x wide)
///////////////////////////////////////////////////////////////////////////////
void Fluid::makeColorLong(float* output)
{
	MakeColorLong<<<grid,threads>>>(chemA, chemB, vel[0], vel[1], 
									temperature, pressure, divergence, temperature,
									output, dimX, dimY, 2);
}