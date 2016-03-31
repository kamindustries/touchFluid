/* 
	Simulation functions that kick off CUDA kernels
	Kurt Kaminski 2016
*/

#include "kernels.cuh"
#include "common.cuh"

dim3 grid, threads;

bool runOnce = true;
int dimX, dimY, size;
float *chemA, *chemA_prev, *chemB, *chemB_prev, *laplacian;
float *vel[2], *vel_prev[2];
float *pressure, *pressure_prev;
float *temperature, *temperature_prev;
float *density, *density_prev;
float *divergence;
float *boundary;

// incoming data
float *mouse, *mouse_old;
const TCUDA_ParamInfo *mouseCHOP;
const TCUDA_ParamInfo *boundaryTOP;
const TCUDA_ParamInfo *F_TOP;
const TCUDA_ParamInfo *rdCHOP;

float dt = .1;
float diff = 0.00001f;
float visc = 0.000001f;
float force = 30.;
float buoy = 0.0;
float source_density = 2.0;
float source_temp = .25;
float dA = 0.0002; // diffusion constants
float dB = 0.00001;

char* TCUDA_DataType_enum[];
char* TCUDA_ProjectionType_enum[];
char* TCUDA_ObjSubType_enum[];
char* TCUDA_DataFormat_enum[];
char* TCUDA_ParamDataFormat_enum[];
char* TCUDA_DataLocation_enum[];
char* TCUDA_ChanOrder_enum[];
char* TCUDA_OutputType_enum[];
char* TCUDA_PixelFormat_enum[];
char* TCUDA_ObjParamType_enum[];
char* TCUDA_FogType_enum[];
char* TCUDA_MemType_enum[];

// ffmpeg -i [input] -c:v libvpx -b:v 1M [output].webm
bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

bool hasBeginning (std::string const &fullString, std::string const &beginning) {
	if (fullString.find(beginning) != std::string::npos )
		return true;
	else 
		return false;
}

///////////////////////////////////////////////////////////////////////////////
// Print connected node information
///////////////////////////////////////////////////////////////////////////////
void printNodeInfo(const int nparams, const TCUDA_ParamInfo **params){
	printf("\n----------\nINCOMING PARAMETERS:\n");
	printf("%d nodes connected\n\n", nparams);

	for (int i = 0; i < nparams; i++) {
		printf("Node %d: %s\n", params[i]->inputNumber, params[i]->name);
		printf("%d values\n", params[i]->param.numValues);
		if (params[i]->dataType == TCUDA_DATA_TYPE_TOP){
			printf("  TOP INFO:\n");
			printf("  w: %d, h: %d\n", params[i]->top.width, params[i]->top.height);
			printf("  %s\n", TCUDA_ChanOrder_enum[params[i]->top.chanOrder]); 
		}
		if (params[i]->dataType == TCUDA_DATA_TYPE_CHOP){
			printf("  CHOP INFO:\n");
			printf("  Num channels: %d\n", params[i]->chop.numChannels); 
			printf("  Length: %d\n", params[i]->chop.length);
			printf("  Sample rate: %f\n", params[i]->chop.sampleRate); 
		}
		printf("\n");
	}
	printf("----------\n\n");
}

///////////////////////////////////////////////////////////////////////////////
// Find CHOPS
///////////////////////////////////////////////////////////////////////////////
void findCHOPS(const int nparams, const TCUDA_ParamInfo **params){
	for (int i = 0; i < nparams; i++){
		if (hasBeginning(params[i]->name, "OUT_mouse")){
			mouseCHOP = params[i];
			printf("findCHOPS(): found mouse: %s\n", mouseCHOP->name);
		}
		if (hasBeginning(params[i]->name, "OUT_boundary")){
			boundaryTOP = params[i];
			printf("findCHOPS(): found Boundary TOP: %s\n", boundaryTOP->name);
		}
		if (hasBeginning(params[i]->name, "OUT_rdF")){
			F_TOP = params[i];
			printf("findCHOPS(): found F TOP: %s\n", F_TOP->name);
		}
		if (hasBeginning(params[i]->name, "OUT_rdCHOP")){
			rdCHOP = params[i];
			printf("findCHOPS(): found rd CHOP: %s\n", rdCHOP->name);
		}
	}

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
	
	// Get mouse info
	int num_mouse_chans = mouseCHOP->chop.numChannels;
	mouse = (float*)malloc(sizeof(float)*num_mouse_chans);
	mouse_old = (float*)malloc(sizeof(float)*num_mouse_chans);
	cudaMemcpy(mouse, (float*)mouseCHOP->data, sizeof(float)*num_mouse_chans, cudaMemcpyDeviceToHost);
	for (int i=0; i<num_mouse_chans; i++){
		mouse_old[i]=mouse[i];
	}
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
void initialize(const int _nparams, const TCUDA_ParamInfo **_params, const TCUDA_ParamInfo *_output)
{
	printNodeInfo(_nparams, _params);
	findCHOPS(_nparams, _params);
	initVariables(_params, _output);
	initCUDA();
	initArrays();
	printf("initialize(): done.\n");
}

///////////////////////////////////////////////////////////////////////////////
// Get from UI
///////////////////////////////////////////////////////////////////////////////
void get_from_UI(const TCUDA_ParamInfo **params, float *_temp, float *_dens, float *_u, float *_v) 
{
	ClearArray<<<grid,threads>>>(chemA_prev, 1.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(chemB_prev, 0.0, dimX, dimY);
	//ClearArray<<<grid,threads>>>(_u, 0.0, dimX, dimY);
	//ClearArray<<<grid,threads>>>(_v, 0.0, dimX, dimY);

	//DrawSquare<<<grid,threads>>>(_chemB0, 1.0, dimX, dimY);

	// Use first input as material for chemB
	//MakeSource<<<grid,threads>>>((int*)params[0]->data, _chemB0, dimX, dimY);
	
	// Use second input as boundary conditions
	//boundary = (float*)boundaryTOP->data;
	//MakeSource<<<grid,threads>>>((float*)boundaryTOP->data, boundary, dimX, dimY);
	
	// Apply obstacle velocity
	GetFromUI<<<grid,threads>>>(_u, _v, (float*)boundaryTOP->data, dimX, dimY);
	
	// Update mouse info
	cudaMemcpy(mouse, (float*)mouseCHOP->data, sizeof(float)*mouseCHOP->chop.numChannels, cudaMemcpyDeviceToHost);
	
	if ( mouse[2] < 1.0 && mouse[3] < 1.0 ) return;

	// map mouse position to window size
	//float mouse[0] = (float)(mouse_x)/(float)win_x;
	//float mouse[1] = (float)(win_y-mouse_y)/(float)win_y;
	int i, j = dimX * dimY;
	i = (int)(mouse[0]*dimX-1);
	j = (int)(mouse[1]*dimY-1);

	float x_diff = mouse[0]-mouse_old[0];
	float y_diff = mouse[1]-mouse_old[1];
	//printf("%f, %f\n", x_diff, y_diff);

	if ( i<1 || i>dimX || j<1 || j>dimY ) return;

	if ( mouse[2] > 0.0 && mouse[3] > 0.0) {
		GetFromUI<<<grid,threads>>>(_u, x_diff * force, i, j, dimX, dimY);
		GetFromUI<<<grid,threads>>>(_v, y_diff * force, i, j, dimX, dimY);
	}

	if ( mouse[3] > 0.0) {
		GetFromUI<<<grid,threads>>>(_dens, source_density, i, j, dimX, dimY);
		GetFromUI<<<grid,threads>>>(_temp, source_temp, i, j, dimX, dimY);
		//GetFromUI<<<grid,threads>>>(_chemB0, source_density, i, j, dimX, dimY);
//		particleSystem.addParticles(mouse[0], mouse[1], 100, .04);
	}

	if ( mouse[4] > 0.0 || mouse[5] > 0.0) {
		printf("Mouse wheel is down!\n");
	}

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
	AddSource<<<grid,threads>>>(_chemB, _chemB0, dt, dimX, dimY);
	//AddSource<<<grid,threads>>>(_chemA, _chemA0, dt, dimX, dimY);
	_chemA0 = _chemA;
	_chemB0 = _chemB;
	for (int i = 0; i < 2; i++){
		Diffusion<<<grid,threads>>>(_chemA, laplacian, bounds, dA, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemA, laplacian, dimX, dimY);
		SetBoundary<<<grid,threads>>>(0, _chemA, bounds, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		Diffusion<<<grid,threads>>>(_chemB, laplacian, bounds, dB, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemB, laplacian, dimX, dimY);
		SetBoundary<<<grid,threads>>>(0, chemB, bounds, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		for (int j = 0; j < 2; j++){
		React<<<grid,threads>>>( _chemA, _chemB, (float*)F_TOP->data, (float*)rdCHOP->data, bounds, dt, dimX, dimY );
		
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
								(float*)boundaryTOP->data, vel[0], vel[1], 
								dt, .9995, dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);

	// Temperature advection
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], temperature_prev, (float*)boundaryTOP->data, temperature,
							dt, .99, false, dimX, dimY);
	SWAP(temperature_prev, temperature);

	// Vorticity Confinement
	vorticityConfinement<<<grid,threads>>>( vel[0], vel[1], vel_prev[0], vel_prev[1], 
											(float*)boundaryTOP->data, dt, dimX, dimY);
		
	float Tamb = 0.0;
	getSum<<<grid,threads>>>(temperature_prev, Tamb, dimX, dimY);
	Tamb /= float(dimX * dimY);
	ApplyBuoyancy<<<grid,threads>>>(vel_prev[0], vel_prev[1], temperature_prev, chemB_prev,
									vel[0], vel[1], Tamb, dt, dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);

	// Apply impulses
	get_from_UI(params, temperature_prev, chemB_prev, vel_prev[0], vel_prev[1]);

	// Density advection
	dens_step( chemA, chemA_prev, chemB, chemB_prev, vel_prev[0], vel_prev[1], (float*)boundaryTOP->data, dt );

	// Compute divergence
	ComputeDivergence<<<grid,threads>>>( vel_prev[0], vel_prev[1], (float*)boundaryTOP->data, divergence, dimX, dimY );

	// Pressure solve
	ClearArray<<<grid,threads>>>(pressure_prev, 0.0, dimX, dimY);
	for (int i=0; i<30; i++){
		Jacobi<<<grid,threads>>>(pressure_prev, divergence, (float*)boundaryTOP->data, pressure, dimX, dimY);
		SWAP(pressure_prev, pressure);
	}

	// Subtract pressure gradient from velocity
	SubtractGradient<<<grid,threads>>>( vel_prev[0], vel_prev[1], pressure_prev, (float*)boundaryTOP->data, 
										vel[0], vel[1], dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);


	MakeColor<<<grid,threads>>>(chemA, chemB, vel[0], vel[1], (float*)output->data, dimX, dimY);
	//MakeColor<<<grid,threads>>>(chemB, (float*)boundaryTOP->data, chemB, (float*)output->data, dimX, dimY);
	//(float*)boundaryTOP->data

}

extern "C"
{
	// The main function where you should execute your CUDA kernel(s).
	// nparams is the number of parameters passed into this function
	// and params is the array of those parameters
	// output contains information about the output you need to write
	// output.data is the array that you write out to (this will be turned into a TOP by Touch)

	// 3d texture idea: output different Z slices with each frame #, compiling them into a Texture3d TOP
	//					would have to change framerate to compensate for #of slices/fps 
	DLLEXPORT bool
	tCudaExecuteKernel(const TCUDA_NodeInfo *info, const int nparams, const TCUDA_ParamInfo **params, const TCUDA_ParamInfo *output)
	{
		if (runOnce) {
			initialize(nparams, params, output);
			runOnce = false;
		}

		simulate(params, output);

		return true;
	}

}