/* 
	Simulation functions that kick off CUDA kernels
	Kurt Kaminski 2016
*/

#include "kernels.cuh"
#include "common.cuh"

dim3 grid, threads;
float dt = 0.1;
float diff = 0.00001f;
float visc = 0.000f;
float force = 5.0;
float buoy;
float source_density = 5.0;
float dA = 0.0002; // diffusion constants
float dB = 0.00001;

void initVariables(const TCUDA_ParamInfo *output)
{
	threads = dim3(16,16);
	grid.x = (output->top.width + threads.x - 1) / threads.x; //replace with dimX, dimY
	grid.y = (output->top.height + threads.y - 1) / threads.y;
	
	printf("w: %d, h: %d\n", dimX, dimY);
}

void initArrays() {
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

  //buoy = 0.0;
}

void get_from_UI(const TCUDA_ParamInfo **field, float *_chemA0, float *_chemB0) 
{
//	int i, j = (N+2)*(N+2);

	ClearArray<<<grid,threads>>>(_chemA0, 1.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(_chemB0, 0.0, dimX, dimY);
//	ClearArray<<<grid,threads>>>(_u, 0.0);
//	ClearArray<<<grid,threads>>>(_v, 0.0);

	//DrawSquare<<<grid,threads>>>(_chemB0, 1.0, dimX, dimY);
	MakeSource<<<grid,threads>>>((int*)field[0]->data, _chemB0, dimX, dimY);

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
		Diffusion<<<grid,threads>>>(_chemA, laplacian, dA, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemA, laplacian, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		Diffusion<<<grid,threads>>>(_chemB, laplacian, dB, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemB, laplacian, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		React<<<grid,threads>>>( _chemA, _chemB, dt, dimX, dimY );
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
	DLLEXPORT bool
	tCudaExecuteKernel(const TCUDA_NodeInfo *info, const int nparams, const TCUDA_ParamInfo **params, const TCUDA_ParamInfo *output)
	{
		const TCUDA_ParamInfo *topParam = NULL;
		// Find the first TOP that was connected
		for (int i = 0; i < nparams; i++)
		{
			if (params[i]->dataType == TCUDA_DATA_TYPE_TOP)
			{
				topParam = params[i];
				break;
			}
		}
		if (topParam == NULL)
		{
			return false;
		}

		if (runOnce) {
			initVariables(output);
			initArrays();
			runOnce = false;
			printf("tCudaExecuteKernel: ran once.");
		}
		simulate(params, output);

		//dim3 block(16, 16, 1);
		//int extraX = 0;
		//int extraY = 0;
		//if (output->top.width % block.x > 0)
		//	extraX = 1;
		//if (output->top.height % block.y > 0)
		//	extraY = 1;
		//dim3 grid((output->top.width / block.x) + extraX , (output->top.height / block.y) + extraY , 1);

		//sampleKernel <<< grid, block >>> ((int*)params[2]->data, params[2]->top.width, params[2]->top.height,
		//								  (int*)output->data, output->top.width, output->top.height);

		return true;
	}

}