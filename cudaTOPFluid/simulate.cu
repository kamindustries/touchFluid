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
#include "Fluid.cuh"

using namespace std;

Fluid fluid;

bool runOnce = true;
bool initialized = false;

// incoming data
map<string, const TCUDA_ParamInfo*> nodes;
float *mouse, *mouse_old;
float *res, *globals, *advectionConstants, *rdConstants;
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
// Find connected nodes for easier reference in a map
///////////////////////////////////////////////////////////////////////////////
bool findNodes(const int nparams, const TCUDA_ParamInfo **params){

	// fill nodes<> with key/value pairs
	nodes["mouse"] = mouseCHOP;
	nodes["density"] = densityTOP; //density and temperature: rgba = chemA, chemB, temperature
	nodes["boundary"] = boundaryTOP;
	nodes["rd"] = rdConstantsCHOP;
	nodes["globals"] = globalsCHOP;
	nodes["advection"] = advectionConstantsCHOP;
	nodes["reset"] = resetCHOP;
	nodes["fluidRes"] = fluidResCHOP;


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

void setGlobals() {
	// globals[] = {dt, nDiff, nReact, nJacobi}
	fluid.dt = globals[0];
	fluid.nDiff = (int)globals[1];
	fluid.nReact = (int)globals[2];
	fluid.nJacobi = (int)globals[3];
}

void setAdvectionConstants() {
	// advectionConstants[] = {velDiff, tempDiff, densDiff, curl, buoyancy, weight}
	fluid.velDiff = advectionConstants[0];
	fluid.tempDiff = advectionConstants[1];
	fluid.densDiff = advectionConstants[2];
	fluid.curlAmt = advectionConstants[3];
	fluid.buoy = advectionConstants[4];
	fluid.weight = advectionConstants[5];
}

void setRdConstants() {
	fluid.F = rdConstants[0];
	fluid.k = rdConstants[1];
	fluid.dA = rdConstants[2];
	fluid.dB = rdConstants[3];
	fluid.xLen = (int)rdConstants[4];
	fluid.yLen = (int)rdConstants[5];
	fluid.rdEquation = (int)rdConstants[6];
	fluid.e = rdConstants[7];
}

void setFluidConstants(){
	setGlobals();
	setAdvectionConstants();
	setRdConstants();
}

///////////////////////////////////////////////////////////////////////////////
// Set up global variables
///////////////////////////////////////////////////////////////////////////////
void initParameters(const TCUDA_ParamInfo **_params, const TCUDA_ParamInfo *_output)
{
	res = (float*)malloc(sizeof(float)*nodes["fluidRes"]->chop.numChannels);
	res = (float*)nodes["fluidRes"]->data;
	
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
	setGlobals();
	setAdvectionConstants();
	setRdConstants();

	printf("initParameters(): done.\n");
}

///////////////////////////////////////////////////////////////////////////////
// Initialize
///////////////////////////////////////////////////////////////////////////////
bool init(const int _nparams, const TCUDA_ParamInfo **_params, const TCUDA_ParamInfo *_output)
{
	printNodeInfo(_nparams, _params);

	if ( findNodes(_nparams, _params) ) {
		initParameters(_params, _output);

		// Initialize fluid container
		fluid.init(res[0], res[1]);
		
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
			// Reset to initial values if reset button is pressed
			if (*(float*)nodes["reset"]->data > 0.0f) {
				fluid.clearArrays();
			}

			setFluidConstants();
			fluid.step( (float*)nodes["density"]->data, (float*)nodes["boundary"]->data );
			fluid.makeColorLong((float*)output->data);

			return true;
		}

		else return false;

	}

}