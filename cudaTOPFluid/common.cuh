#ifndef __CUDA_TOP_FLUID_COMMON__
#define __CUDA_TOP_FLUID_COMMON__

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <map>
#include <string>

#include "defines.h"
#include "TCUDA_enum.h"
#include "private/TCUDA_Types.h"

extern int dimX, dimY, size;

extern bool hasEnding (std::string const &fullString, std::string const &ending);
extern bool hasBeginning (std::string const &fullString, std::string const &beginning);


#endif 
