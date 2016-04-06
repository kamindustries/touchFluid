/* 
	Simulation functions that kick off CUDA kernels
	Kurt Kaminski 2016
*/

#include <iostream>
#include <map>

#include "common.cuh"
#include "kernels.cuh"
#include "util_functions.hpp"
#include "private/TCUDA_Types.h"

using namespace std;

dim3 grid, threads;

bool runOnce = true;
bool initialized = false;

int dimX, dimY, size;
float *chemA, *chemA_prev, *chemB, *chemB_prev, *laplacian;
float *vel[2], *vel_prev[2];
float *pressure, *pressure_prev;
float *temperature, *temperature_prev;
float *density, *density_prev;
float *divergence;
float *boundary;

// incoming data
map<string, const TCUDA_ParamInfo*> nodes;
float *mouse, *mouse_old;
float *constants;
const TCUDA_ParamInfo *mouseCHOP;
const TCUDA_ParamInfo *densityTOP;
const TCUDA_ParamInfo *boundaryTOP;
const TCUDA_ParamInfo *rdCHOP;
const TCUDA_ParamInfo *constCHOP;
const TCUDA_ParamInfo *resetCHOP;

float dt = 0.1f;
float dA = 0.0002; // gray-scott
float dB = 0.00001;
float xLen = 100.0f;
float yLen = 100.0f;
int nDiff = 2;
int nReact = 1;
int nJacobi = 30;

//float dt = .02f;
//float dA = 0.75; // barkley model
//float dB = 0.0;

float diff = 0.00001f;
float visc = 0.000001f;
float force = 30.;
float buoy = 0.0;
float source_density = 2.0;
float source_temp = .25;

// ffmpeg -i [input] -c:v libvpx -b:v 1M [output].webm
// ffmpeg -i [input] -c:v libx264 -b:v 1M [output].webm

///////////////////////////////////////////////////////////////////////////////
// Find connected nodes for easier reference in a map
///////////////////////////////////////////////////////////////////////////////
bool findNodes(const int nparams, const TCUDA_ParamInfo **params){

	// fill nodes<> with key/value pairs
	nodes["mouse"] = mouseCHOP;
	nodes["density"] = densityTOP;
	nodes["boundary"] = boundaryTOP;
	nodes["rdCHOP"] = rdCHOP;
	nodes["constants"] = rdCHOP;
	nodes["reset"] = resetCHOP;


	// search incoming params[] for matching name and assigne nodes<> value to it
	bool missingNodes = false;
	typedef map<string, const TCUDA_ParamInfo*>::iterator iterator;
	for (iterator it = nodes.begin(); it != nodes.end(); it++) {
		for (int i = 0; i < nparams; i++){
			if (hasBeginning(params[i]->name, it->first.c_str())) {
				it->second = params[i];
				printf("findNodes(): found %s: %s\n", it->first.c_str(), it->second->name);
				break;
			}
			if (i == nparams-1) {
				printf("findNodes(): error: could not find %s!\n", it->first.c_str());
				missingNodes = true;
			}
		}
	}

	if (missingNodes) {
		printf("findNodes(): couldn't find required nodes. To continue, attach required nodes.\n");
		return false;
	}
	else {
		printf("findNodes(): done finding nodes.\n");
		return true;
	}

}

void getConstants() {
	// constants[] = {dt, xLen, yLen, nDiff, nReact, nJacobi}
	dt = constants[0];
	xLen = constants[1];
	yLen = constants[2];
	nDiff = (int)constants[3];
	nReact = (int)constants[4];
	nJacobi = (int)constants[5];
}

///////////////////////////////////////////////////////////////////////////////
// Set up global variables
///////////////////////////////////////////////////////////////////////////////
void initVariables(const TCUDA_ParamInfo **_params, const TCUDA_ParamInfo *_output)
{
	// Set container dimensions to whatever the incoming TOP is set to
	dimX = _output->top.width;
	dimY = _output->top.height;
	size = dimX * dimY;

	threads = dim3(16,16);
	grid.x = (dimX + threads.x - 1) / threads.x;
	grid.y = (dimY + threads.y - 1) / threads.y;
	
	printf("-- DIMENSIONS: %d x %d --\n", dimX, dimY);
	
	// Allocate mouse array
	mouse = (float*)malloc(sizeof(float)*nodes["mouse"]->chop.numChannels);
	mouse_old = (float*)malloc(sizeof(float)*nodes["mouse"]->chop.numChannels);
	
	// Local mouse pointer points to CHOP node
	mouse = (float*)nodes["mouse"]->data;
	for (int i = 0; i < nodes["mouse"]->chop.numChannels; i++){
		mouse_old[0]=mouse[1];
	}

	// Allocate constants array
	constants = (float*)malloc(sizeof(float)*nodes["constants"]->chop.numChannels);
	
	// Local constants pointer points to CHOP node
	constants = (float*)nodes["constants"]->data;
	getConstants();

	printf("initVariables(): done.\n");
}

///////////////////////////////////////////////////////////////////////////////
// Allocate GPU memory
///////////////////////////////////////////////////////////////////////////////
void initCUDA() 
{
	cudaMalloc((void**)&chemA, sizeof(float)*size);
	cudaMalloc((void**)&chemA_prev, sizeof(float)*size);
	cudaMalloc((void**)&chemB, sizeof(float)*size);
	cudaMalloc((void**)&chemB_prev, sizeof(float)*size);
	cudaMalloc((void**)&laplacian, sizeof(float)*size);
	cudaMalloc((void**)&boundary, sizeof(float)*size * 4);

	for (int i=0; i<2; i++){
		cudaMalloc((void**)&vel[i], sizeof(int)*size);
		cudaMalloc((void**)&vel_prev[i], sizeof(int)*size);
	}

	cudaMalloc((void**)&pressure, sizeof(float)*size );
	cudaMalloc((void**)&pressure_prev, sizeof(float)*size );
	cudaMalloc((void**)&temperature, sizeof(float)*size );
	cudaMalloc((void**)&temperature_prev, sizeof(float)*size );
	cudaMalloc((void**)&density, sizeof(float)*size );
	cudaMalloc((void**)&density_prev, sizeof(float)*size );
	cudaMalloc((void**)&divergence, sizeof(float)*size );

	printf("initCUDA(): Allocated GPU memory.\n");
}

///////////////////////////////////////////////////////////////////////////////
// Initialize GPU arrays
///////////////////////////////////////////////////////////////////////////////
void initArrays() 
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
  ClearArray<<<grid,threads>>>(density, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(density_prev, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(divergence, 0.0, dimX, dimY);

  printf("initArrays(): Initialized GPU arrays.\n");
}

///////////////////////////////////////////////////////////////////////////////
// Initialize
///////////////////////////////////////////////////////////////////////////////
bool init(const int _nparams, const TCUDA_ParamInfo **_params, const TCUDA_ParamInfo *_output)
{
	printNodeInfo(_nparams, _params);

	if ( findNodes(_nparams, _params) ) {
		initVariables(_params, _output);
		initCUDA();
		initArrays();
		printf("init(): done.\n");
		return true;
	}
	else {
		printf("init(): could not initalize. Not simulating.\n");
		return false;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Get from UI
///////////////////////////////////////////////////////////////////////////////
void get_from_UI(const TCUDA_ParamInfo **params, float *_temp, float *_dens, float *_u, float *_v) 
{
	//ClearArray<<<grid,threads>>>(chemA_prev, 1.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(chemA_prev, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(chemB_prev, 0.0, dimX, dimY);
	//ClearArray<<<grid,threads>>>(_u, 0.0, dimX, dimY);
	//ClearArray<<<grid,threads>>>(_v, 0.0, dimX, dimY);

	//DrawSquare<<<grid,threads>>>(chemA, 1.0, dimX, dimY);
	//DrawSquare<<<grid,threads>>>(chemB, 0.75 * .5, dimX, dimY);
	
	// Apply incoming density
	SetFromUI<<<grid,threads>>>(chemA, chemB, (float*)nodes["density"]->data, dimX, dimY);
	
	// Apply obstacle velocity
	AddFromUI<<<grid,threads>>>(_u, _v, (float*)nodes["boundary"]->data, dimX, dimY);
	
	// Update mouse and constants info
	// mouse[] = {x, y, LMB, RMB, MMB, wheel}
	//mouse = (float*)nodes["mouse"]->data;
	getConstants();

	if ( mouse[2] < 1.0 && mouse[3] < 1.0 ) return;

	int i, j = dimX * dimY;
	i = (int)(mouse[0]*dimX-1);
	j = (int)(mouse[1]*dimY-1);

	float x_diff = mouse[0]-mouse_old[0];
	float y_diff = mouse[1]-mouse_old[1];
	//printf("%f, %f\n", x_diff, y_diff);

	if (i<1 || i>dimX || j<1 || j>dimY ) return;

	if (mouse[2] > 0.0 && mouse[3] > 0.0) {
		AddFromUI<<<grid,threads>>>(_u, x_diff * force, i, j, dimX, dimY);
		AddFromUI<<<grid,threads>>>(_v, y_diff * force, i, j, dimX, dimY);
	}

	if (mouse[3] > 0.0) {
		AddFromUI<<<grid,threads>>>(_dens, source_density, i, j, dimX, dimY);
		AddFromUI<<<grid,threads>>>(_temp, source_temp, i, j, dimX, dimY);
		//GetFromUI<<<grid,threads>>>(_chemB0, source_density, i, j, dimX, dimY);
		//particleSystem.addParticles(mouse[0], mouse[1], 100, .04);
	}

	if (mouse[4] > 0.0) printf("mouse[4] is down!\n");

	for (int i=0; i<6; i++){
		mouse_old[i]=mouse[i];
	}

	return;
}

///////////////////////////////////////////////////////////////////////////////
// Density step
///////////////////////////////////////////////////////////////////////////////
void dens_step (  float *_chemA, float *_chemA0, float *_chemB, float *_chemB0,
				  float *u, float *v, float *bounds, float dt )
{

	// Naive ARD-----------------------
	//AddSource<<<grid,threads>>>(_chemB, _chemB0, dt, dimX, dimY);
	//AddSource<<<grid,threads>>>(_chemA, _chemA0, dt, dimX, dimY);
	_chemA0 = _chemA;
	_chemB0 = _chemB;
	for (int i = 0; i < nDiff; i++){
		Diffusion<<<grid,threads>>>(_chemA, laplacian, bounds, dA, xLen, yLen, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemA, laplacian, dimX, dimY);
		SetBoundary<<<grid,threads>>>(0, _chemA, bounds, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		Diffusion<<<grid,threads>>>(_chemB, laplacian, bounds, dB, xLen, yLen, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemB, laplacian, dimX, dimY);
		SetBoundary<<<grid,threads>>>(0, chemB, bounds, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		for (int j = 0; j < nReact; j++){
			React<<<grid,threads>>>( _chemA, _chemB, (float*)nodes["rdCHOP"]->data, bounds, dt, dimX, dimY );
		}
	}

	SWAP ( _chemA0, _chemA );
	SWAP ( _chemB0, _chemB );

	// Density advection: chemB
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], _chemA0, bounds, _chemA,
							dt, 1.0, true, dimX, dimY);

	// Density advection: chemB
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], _chemB0, bounds, _chemB,
							dt, 1.0, true, dimX, dimY);
}


///////////////////////////////////////////////////////////////////////////////
// Simulate
///////////////////////////////////////////////////////////////////////////////
static void simulate(const TCUDA_ParamInfo **params, const TCUDA_ParamInfo *output){


	// Velocity advection
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], vel_prev[0], vel_prev[1],
								(float*)nodes["boundary"]->data, vel[0], vel[1], 
								dt, .9995, dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);

	// Temperature advection
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], temperature_prev, (float*)nodes["boundary"]->data, temperature,
							dt, .99, false, dimX, dimY);
	SWAP(temperature_prev, temperature);

	// Vorticity Confinement
	vorticityConfinement<<<grid,threads>>>( vel[0], vel[1], vel_prev[0], vel_prev[1], 
											(float*)nodes["boundary"]->data, dt, dimX, dimY);
		
	float Tamb = 0.0;
	getSum<<<grid,threads>>>(temperature_prev, Tamb, dimX, dimY);
	Tamb /= float(dimX * dimY);
	ApplyBuoyancy<<<grid,threads>>>(vel_prev[0], vel_prev[1], temperature_prev, chemB_prev,
									vel[0], vel[1], Tamb, dt, dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);

	// Apply impulses
	get_from_UI(params, temperature_prev, chemB_prev, vel_prev[0], vel_prev[1]);

	// Reaction-Diffusion and Density advection
	dens_step( chemA, chemA_prev, chemB, chemB_prev, vel_prev[0], vel_prev[1], (float*)nodes["boundary"]->data, dt );

	// Compute divergence
	ComputeDivergence<<<grid,threads>>>( vel_prev[0], vel_prev[1], (float*)nodes["boundary"]->data, divergence, dimX, dimY );

	// Pressure solve
	ClearArray<<<grid,threads>>>(pressure_prev, 0.0, dimX, dimY);
	for (int i=0; i<nJacobi; i++){
		Jacobi<<<grid,threads>>>(pressure_prev, divergence, (float*)nodes["boundary"]->data, pressure, dimX, dimY);
		SWAP(pressure_prev, pressure);
	}

	// Subtract pressure gradient from velocity
	SubtractGradient<<<grid,threads>>>( vel_prev[0], vel_prev[1], pressure_prev, (float*)nodes["boundary"]->data, 
										vel[0], vel[1], dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);


	MakeColor<<<grid,threads>>>(chemA, chemB, vel[0], vel[1], (float*)output->data, dimX, dimY);
	//MakeColor<<<grid,threads>>>(chemB, (float*)nodes["boundary"]->data, chemB, (float*)output->data, dimX, dimY);

}

extern "C"
{
	// The main function to execute CUDA kernel(s).
	// nparams is the number of parameters passed into this function
	// and params is the array of those parameters
	// output contains information about the output you need to write
	// output.data is the array to write out to (this will be turned into a TOP by Touch)

	// 3d texture idea: output different Z slices with each frame #, compiling them into a Texture3d TOP
	//					would have to change framerate to compensate for #of slices/fps 
	DLLEXPORT bool
	tCudaExecuteKernel(const TCUDA_NodeInfo *info, const int nparams, const TCUDA_ParamInfo **params, const TCUDA_ParamInfo *output)
	{
		if (runOnce) {
			initialized = init(nparams, params, output);
			runOnce = false;
		}

		if (initialized) {
			if (*(float*)nodes["reset"]->data > 0.0f) {
				initArrays();
			}

			simulate(params, output);
			return true;
		}

		else return false;

	}

}