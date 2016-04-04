#ifndef __CUDA_TOP_FLUID_UTIL_FUNCTIONS__
#define __CUDA_TOP_FLUID_UTIL_FUNCTIONS__

#include <string>
#include "TCUDA_enum.h"
#include "private/TCUDA_Types.h"

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


#endif