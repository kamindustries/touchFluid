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
float *u, *u_prev, *v, *v_prev;

// incoming data
float *mouse, *mouse_old;
int *boundary;
const TCUDA_ParamInfo *mouseCHOP;

float dt = 0.1;
float diff = 0.00001f;
float visc = 0.000001f;
float force = 500.0;
float buoy = 0.0;
float source_density = 5.0;
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


bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

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

void findCHOPS(const int nparams, const TCUDA_ParamInfo **params){
	for (int i = 0; i < nparams; i++){
		if (hasEnding(params[i]->name, "mouseOUT")){
			mouseCHOP = params[i];
			printf("findCHOPS(): found mouse: %s\n", mouseCHOP->name);
		}
	}

}

void initVariables(const TCUDA_ParamInfo **_params, const TCUDA_ParamInfo *_output)
{
	// Set container dimensions to whatever the incoming TOP is set to
	dimX = _output->top.width;
	dimY = _output->top.height;
	size = dimX * dimY;

	threads = dim3(16,16);
	grid.x = (_output->top.width + threads.x - 1) / threads.x; //replace with dimX, dimY
	grid.y = (_output->top.height + threads.y - 1) / threads.y;
	
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

void initCUDA() 
{
	cudaMalloc((void**)&u, sizeof(float)*size );
	cudaMalloc((void**)&u_prev, sizeof(float)*size );
	cudaMalloc((void**)&v, sizeof(float)*size );
	cudaMalloc((void**)&v_prev, sizeof(float)*size );

	cudaMalloc((void**)&chemA, sizeof(float)*size);
	cudaMalloc((void**)&chemA_prev, sizeof(float)*size);
	cudaMalloc((void**)&chemB, sizeof(float)*size);
	cudaMalloc((void**)&chemB_prev, sizeof(float)*size);
	cudaMalloc((void**)&laplacian, sizeof(float)*size);
	cudaMalloc((void**)&boundary, sizeof(int)*size);

	printf("initCUDA(): Allocated GPU memory.\n");
}


void initArrays() 
{
  ClearArray<<<grid,threads>>>(u, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(u_prev, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(v, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(v_prev, 0.0, dimX, dimY);

  ClearArray<<<grid,threads>>>(chemA, 1.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(chemA_prev, 1.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(chemB, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(chemB_prev, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(boundary, 0.0, dimX, dimY);

  printf("initArrays(): Initialized GPU arrays.\n");
}

void initialize(const int _nparams, const TCUDA_ParamInfo **_params, const TCUDA_ParamInfo *_output)
{
	printNodeInfo(_nparams, _params);
	findCHOPS(_nparams, _params);
	initVariables(_params, _output);
	initCUDA();
	initArrays();
	printf("initialize(): done.\n");
}

void get_from_UI(const TCUDA_ParamInfo **params, float *_chemA0, float *_chemB0, float *_u, float *_v) 
{
	ClearArray<<<grid,threads>>>(_chemA0, 1.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(_chemB0, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(_u, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(_v, 0.0, dimX, dimY);

	//DrawSquare<<<grid,threads>>>(_chemB0, 1.0, dimX, dimY);

	// Use first input as material for chemB
	MakeSource<<<grid,threads>>>((int*)params[0]->data, _chemB0, dimX, dimY);
	
	// Use second input as boundary conditions
	MakeSource<<<grid,threads>>>((int*)params[1]->data, boundary, dimX, dimY);

	// Update mouse info
	cudaMemcpy(mouse, (float*)params[2]->data, sizeof(float)*params[2]->chop.numChannels, cudaMemcpyDeviceToHost);

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
		GetFromUI<<<grid,threads>>>(_chemB0, source_density, i, j, dimX, dimY);
//		particleSystem.addParticles(mouse[0], mouse[1], 100, .04);
	}

	for (int i=0; i<6; i++){
		mouse_old[i]=mouse[i];
	}

	return;
}

void diffuse_step(int b, float *field, float *field0, int *bounds, float diff, float dt){
	float a=dt*diff*float(dimX-2)*float(dimY-2); // needed to float(N) to get it to work...
	for (int k = 0; k < 20; k++) {
		LinSolve<<<grid,threads>>>( field, field0, a, (float)1.0+(4.0*a), dimX, dimY );
		SetBoundary<<<grid,threads>>>( b, field, bounds, dimX, dimY );
	}
}

void advect_step ( int b, float *field, float *field0, float *u, float *v, int *bounds, float dt ){
	Advect<<<grid,threads>>>( field, field0, u, v, dt, dimX, dimY );
	SetBoundary<<<grid,threads>>>( b, field, bounds, dimX, dimY );
}

void proj_step( float *u, float *v, float *p, float *div, int *bounds) {
	Project<<<grid,threads>>>( u, v, p, div, dimX, dimY);
	SetBoundary<<<grid,threads>>>(0, div, bounds, dimX, dimY);
	SetBoundary<<<grid,threads>>>(0, p, bounds, dimX, dimY);
	for (int k = 0; k < 20; k++) {
		LinSolve<<<grid,threads>>>( p, div, 1.0, 4.0, dimX, dimY );
		SetBoundary<<<grid,threads>>>(0, p, bounds, dimX, dimY);
	}
	ProjectFinish<<<grid,threads>>>( u, v, p, div, dimX, dimY );
	SetBoundary<<<grid,threads>>>(1, u, bounds, dimX, dimY);
	SetBoundary<<<grid,threads>>>(2, v, bounds, dimX, dimY);
}

void vel_step ( float *u, float *v, float *u0, float *v0, float *dens, int *bounds, float visc, float dt ) {
  AddSource<<<grid,threads>>>( u, u0, dt, dimX, dimY );
  AddSource<<<grid,threads>>>( v, v0, dt, dimX, dimY );

  // add in vorticity confinement force
  vorticityConfinement<<<grid,threads>>>(u0, v0, u, v, dimX, dimY);
  AddSource<<<grid,threads>>>(u, u0, dt, dimX, dimY);
  AddSource<<<grid,threads>>>(v, v0, dt, dimX, dimY);

  // add in buoyancy force
  // get average temperature
  float Tamb = 0.0;
    getSum<<<grid,threads>>>(v0, Tamb, dimX, dimY);
    Tamb /= (dimX * dimY);
  buoyancy<<<grid,threads>>>(v0, dens, Tamb, buoy, dimX, dimY);
  AddSource<<<grid,threads>>>(v, v0, dt, dimX, dimY);

  SWAP ( u0, u ); 
  diffuse_step( 1, u, u0, bounds, visc, dt);
  SWAP ( v0, v ); 
  diffuse_step( 2, v, v0, bounds, visc, dt);

  proj_step( u, v, u0, v0, bounds);

  SWAP ( u0, u );
  SWAP ( v0, v );
  advect_step(1, u, u0, u0, v0, bounds, dt);
  advect_step(2, v, v0, u0, v0, bounds, dt);

  proj_step( u, v, u0, v0, bounds);
}

void dens_step (  float *_chemA, float *_chemA0, float *_chemB, float *_chemB0,
				  float *u, float *v, int *bounds, float dt )
{
	// Naive ARD-----------------------
	AddSource<<<grid,threads>>>(_chemB, _chemB0, dt, dimX, dimY);
	_chemA0 = _chemA;
	_chemB0 = _chemB;
	for (int i = 0; i < 10; i++){
		Diffusion<<<grid,threads>>>(_chemA, laplacian, bounds, dA, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemA, laplacian, dimX, dimY);
		SetBoundary<<<grid,threads>>>(0, _chemA, bounds, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		Diffusion<<<grid,threads>>>(_chemB, laplacian, bounds, dB, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemB, laplacian, dimX, dimY);
		SetBoundary<<<grid,threads>>>(0, chemB, bounds, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		React<<<grid,threads>>>( _chemA, _chemB, bounds, dt, dimX, dimY );
		//SetBoundary<<<grid,threads>>>(0, _chemA, bounds, dimX, dimY);
		//SetBoundary<<<grid,threads>>>(0, _chemB, bounds, dimX, dimY);
	}

	SWAP ( _chemA0, _chemA );
	advect_step(0, _chemA, _chemA0, u, v, bounds, dt);
	SWAP ( _chemB0, _chemB );
	advect_step(0, _chemB, _chemB0, u, v, bounds, dt);
}


///////////////////////////////////////////////////////////////////////////////
// Simulate
///////////////////////////////////////////////////////////////////////////////
static void simulate(const TCUDA_ParamInfo **params, const TCUDA_ParamInfo *output){

	//MakeSource<<<grid,threads>>>((int*)params[0]->data, chemA, dimX, dimY);
	//MakeColor<<<grid,threads>>>(chemA, (int*)output->data, dimX, dimY);

	//if (frameNum > 0 && togSimulate) {
		get_from_UI(params, chemA_prev, chemB_prev, u_prev, v_prev);
		vel_step( u, v, u_prev, v_prev, chemB, boundary, visc, dt );
		dens_step( chemA, chemA_prev, chemB, chemB_prev, u, v, boundary, dt );
		MakeColor<<<grid,threads>>>(chemB, (int*)output->data, dimX, dimY);	
		//MakeVerticesKernel<<<grid,threads>>>(displayVertPtr, u, v);
	//}

	//size_t  sizeT;
	//cudaGraphicsMapResources( 1, &cgrTxData, 0 );
	//checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&displayPtr, &sizeT, cgrTxData));
	//checkCudaErrors(cudaGraphicsUnmapResources( 1, &cgrTxData, 0 ));

	//cudaGraphicsMapResources( 1, &cgrVertData, 0 );
	//checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&displayVertPtr, &sizeT, cgrVertData));
	//checkCudaErrors(cudaGraphicsUnmapResources( 1, &cgrVertData, 0 ));

	//sdkStopTimer(&timer);
	//computeFPS();
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

		//// Get mouse values
		//// currently naively set to 3rd input
		//printf("\n----------\nMouse values:\n");
		//const TCUDA_ParamInfo *td_mouse = params[2];
		//for (int i = 0; i < num_mouse_chans; i++) {
		//	printf("%s: %d, %f\n", td_mouse->name, mouse[i]);
		//}
		//printf("----------\n");

		simulate(params, output);


		return true;
	}

}