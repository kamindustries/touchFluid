/* Shared Use License: This file is owned by Derivative Inc. (Derivative) and
 * can only be used, and/or modified for use, in conjunction with 
 * Derivative's TouchDesigner software, and only if you are a licensee who has
 * accepted Derivative's TouchDesigner license or assignment agreement (which
 * also govern the use of this file).  You may share a modified version of this
 * file with another authorized licensee of Derivative's TouchDesigner software.
 * Otherwise, no redistribution or sharing of this file, with or without
 * modification, is permitted.
 */
#include <map>
#include <string>
#include <iostream>

#include "private/TCUDA_Types.h"
#include "defines.h"
#include "utils.hpp"
#include "FluidSim.cuh"

using namespace std;

bool runOnce = true;
bool initialized = false;

// incoming data
map<string, const TCUDA_ParamInfo*> nodes;
const TCUDA_ParamInfo *mouseCHOP;
const TCUDA_ParamInfo *densityTOP;
const TCUDA_ParamInfo *boundaryTOP;
const TCUDA_ParamInfo *rdConstantsCHOP;
const TCUDA_ParamInfo *globalsCHOP;
const TCUDA_ParamInfo *advectionConstantsCHOP;
const TCUDA_ParamInfo *resetCHOP;
const TCUDA_ParamInfo *fluidResCHOP;


// ffmpeg -i [input] -c:v libvpx -b:v 1M [output].webm
// ffmpeg -i [input] -c:v libx264 -b:v 1M [output].mp4

///////////////////////////////////////////////////////////////////////////////
// Initialize
///////////////////////////////////////////////////////////////////////////////
bool init(const int _nparams, const TCUDA_ParamInfo **_params, const TCUDA_ParamInfo *_output)
{
	printNodeInfo(_nparams, _params);

	// fill nodes<> with key/value pairs
	nodes["mouse"] = mouseCHOP;
	nodes["density"] = densityTOP; //density and temperature: rgba = chemA, chemB, temperature
	nodes["boundary"] = boundaryTOP;
	nodes["rd"] = rdConstantsCHOP;
	nodes["globals"] = globalsCHOP;
	nodes["advection"] = advectionConstantsCHOP;
	nodes["reset"] = resetCHOP;
	nodes["fluidRes"] = fluidResCHOP;

	if ( findNodes(_nparams, _params, nodes) ) {
		init(_params, _output, nodes);

		printf("init(): done.\n\n");
		return true;
	}
	else {
		printf("init(): could not initalize. Not simulating.\n");
		return false;
	}
}


extern "C"
{
	// The main function where you should execute your CUDA kernel(s).
	// nparams is the number of parameters passed into this function
	// and params is the array of those parameters
	// output contains information about the output you need to write
	// output.data is the array that you write out to (this will be turned into a TOP by Touch)
	DLLEXPORT
	bool
	tCudaExecuteKernel(const TCUDA_NodeInfo *info, const int nparams, const TCUDA_ParamInfo **params, const TCUDA_ParamInfo *output)
	{
		if (runOnce) {
			initialized = init(nparams, params, output);
			clearArrays();
			runOnce = false;
		}

		if (initialized) {
			// Reset to initial values if reset button is pressed
			if (*(float*)nodes["reset"]->data > 0.0f) {
				clearArrays();
			}

			step( (float*)nodes["density"]->data, (float*)nodes["boundary"]->data );
			makeColor((float*)output->data);

			return true;
		}
		else {
			return false;
		}

	}
}
