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
int *boundary;

float dt = 0.1;
float diff = 0.00001f;
float visc = 0.000f;
float force = 5.0;
float buoy;
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

void initVariables(const TCUDA_ParamInfo *output)
{
	// Set container dimensions to whatever the incoming TOP is set to
	dimX = output->top.width;
	dimY = output->top.height;
	size = dimX * dimY;

	threads = dim3(16,16);
	grid.x = (output->top.width + threads.x - 1) / threads.x; //replace with dimX, dimY
	grid.y = (output->top.height + threads.y - 1) / threads.y;
	
	printf("-- DIMENSIONS: %d x %d --\n", dimX, dimY);
	printf("initVariables(): done.\n");
	//buoy = 0.0;

}

void initCUDA() 
{
	//cudaMalloc((void**)&u, sizeof(float)*size );
	//cudaMalloc((void**)&u_prev, sizeof(float)*size );
	//cudaMalloc((void**)&v, sizeof(float)*size );
	//cudaMalloc((void**)&v_prev, sizeof(float)*size );
	//cudaMalloc((void**)&dens, sizeof(float)*size );
	//cudaMalloc((void**)&dens_prev, sizeof(float)*size );

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
  //ClearArray<<<grid,threads>>>(u, 0.0);
  //ClearArray<<<grid,threads>>>(u_prev, 0.0);
  //ClearArray<<<grid,threads>>>(v, 0.0);
  //ClearArray<<<grid,threads>>>(v_prev, 0.0);
  //ClearArray<<<grid,threads>>>(dens, 0.0);
  //ClearArray<<<grid,threads>>>(dens_prev, 0.0);

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
	initVariables(_output);
	initCUDA();
	initArrays();
	printf("initialize(): done.");
}

void get_from_UI(const TCUDA_ParamInfo **field, float *_chemA0, float *_chemB0) 
{
//	int i, j = (N+2)*(N+2);

	ClearArray<<<grid,threads>>>(_chemA0, 1.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(_chemB0, 0.0, dimX, dimY);
//	ClearArray<<<grid,threads>>>(_u, 0.0);
//	ClearArray<<<grid,threads>>>(_v, 0.0);

	//DrawSquare<<<grid,threads>>>(_chemB0, 1.0, dimX, dimY);

	// Use first input as material for chemB
	MakeSource<<<grid,threads>>>((int*)field[0]->data, _chemB0, dimX, dimY);
	
	// Use second input as boundary conditions
	MakeSource<<<grid,threads>>>((int*)field[1]->data, boundary, dimX, dimY);

//	if ( !mouse_down[0] && !mouse_down[2] ) return;

//	// map mouse position to window size
//	float mx_f = (float)(mouse_x)/(float)win_x;
//	float my_f = (float)(win_y-mouse_y)/(float)win_y;
//	i = (int)(mx_f*N+1);
//	j = (int)(my_f*N+1);

//	float x_diff = mouse_x-mouse_x_old;
//	float y_diff = mouse_y_old-mouse_y;
//	if (frameNum % 50 == 0) printf("%f, %f\n", x_diff, y_diff);

//	if ( i<1 || i>N || j<1 || j>N ) return;

//	if ( mouse_down[0] ) {
//	GetFromUI<<<grid,threads>>>(_u, i, j, x_diff * force);
//	GetFromUI<<<grid,threads>>>(_v, i, j, y_diff * force);
//	}

//	if ( mouse_down[2]) {
//	GetFromUI<<<grid,threads>>>(_chemB, i, j, source_density);
//	particleSystem.addParticles(mx_f, my_f, 100, .04);
//	}

//	mouse_x_old = mouse_x;
//	mouse_y_old = mouse_y;

//	return;
}

//void dens_step (  float *_chemA, float *_chemA0, float *_chemB, float *_chemB0,
//				  float *u, float *v, float diff, float dt )
void dens_step (  float *_chemA, float *_chemA0, float *_chemB, float *_chemB0,
				  float dt )
{
	// Naive ARD-----------------------
	AddSource<<<grid,threads>>>(_chemB, _chemB0, dt, dimX, dimY );
	_chemA0 = _chemA;
	_chemB0 = _chemB;
	for (int i = 0; i < 10; i++){
		Diffusion<<<grid,threads>>>(_chemA, laplacian, boundary, dA, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemA, laplacian, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		Diffusion<<<grid,threads>>>(_chemB, laplacian, boundary, dB, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemB, laplacian, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		React<<<grid,threads>>>( _chemA, _chemB, boundary, dt, dimX, dimY );
	}

	//SWAP ( _chemA0, _chemA );
	//advect_step(0, _chemA, _chemA0, u, v, dt);
	//SWAP ( _chemB0, _chemB );
	//advect_step(0, _chemB, _chemB0, u, v, dt);
}


///////////////////////////////////////////////////////////////////////////////
// Simulate
///////////////////////////////////////////////////////////////////////////////
static void simulate(const TCUDA_ParamInfo **field, const TCUDA_ParamInfo *output){

	//MakeSource<<<grid,threads>>>((int*)field[0]->data, chemA, dimX, dimY);
	//MakeColor<<<grid,threads>>>(chemA, (int*)output->data, dimX, dimY);

	//if (frameNum > 0 && togSimulate) {
		//get_from_UI(chemA_prev, chemB_prev, u_prev, v_prev);
		get_from_UI(field, chemA_prev, chemB_prev);
		//vel_step( u, v, u_prev, v_prev, chemB, visc, dt );
		//dens_step( field[0], chemA_prev, field[1], chemB_prev, u, v, diff, dt );
		dens_step( chemA, chemA_prev, chemB, chemB_prev, dt );
		MakeColor<<<grid,threads>>>(chemB, (int*)output->data, dimX, dimY);	
		//MakeColor<<<grid,threads>>>(chemA, (int*)output[1]->data, dimX, dimY);	
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

		// Get mouse values
		// currently naively set to 3rd input
		printf("\n----------\nMouse values:\n");
		const TCUDA_ParamInfo *td_mouse = params[2];

		int num_mouse_chans = td_mouse->chop.numChannels;
		float *mouse = (float*)malloc(sizeof(float)*num_mouse_chans);
		cudaMemcpy(mouse, (float*)td_mouse->data, sizeof(float)*num_mouse_chans, cudaMemcpyDeviceToHost);
		for (int i = 0; i < num_mouse_chans; i++) {
			printf("%s: %d, %f\n", td_mouse->name, mouse[i]);
		}
		delete[] mouse;
		printf("----------\n");


		simulate(params, output);


		return true;
	}

}