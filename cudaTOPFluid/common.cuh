#ifndef __CUDA_TOP_FLUID_COMMON__
#define __CUDA_TOP_FLUID_COMMON__

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
//#include <helper_cuda.h>         // helper functions for CUDA error check
#include "private/TCUDA_Types.h"
#include <driver_types.h>
#include "defines.h"

extern bool runOnce;
extern int dimX, dimY, size;
extern float *chemA, *chemA_prev, *chemB, *chemB_prev, *laplacian;
extern float *u, *u_prev, *v, *v_prev;

// incoming data
extern float *mouse, *mouse_old;
extern int *boundary;



#endif 
