/* Shared Use License: This file is owned by Derivative Inc. (Derivative) and
 * can only be used, and/or modified for use, in conjunction with 
 * Derivative's TouchDesigner software, and only if you are a licensee who has
 * accepted Derivative's TouchDesigner license or assignment agreement (which
 * also govern the use of this file).  You may share a modified version of this
 * file with another authorized licensee of Derivative's TouchDesigner software.
 * Otherwise, no redistribution or sharing of this file, with or without
 * modification, is permitted.
 */

// This define is needed before each function so that is exported and Touch can find it in the .dll


#include "../common.cuh"

bool runOnce;
int dimX, dimY, size;
float *chemA, *chemA_prev, *chemB, *chemB_prev, *laplacian;

void initCUDA() 
{
	dimX = 256;
	dimY = 256;
	size = dimX * dimY;
	//cudaMalloc((void**)&u, sizeof(float)*size );
	//cudaMalloc((void**)&u_prev, sizeof(float)*size );
	//cudaMalloc((void**)&v, sizeof(float)*size );
	//cudaMalloc((void**)&v_prev, sizeof(float)*size );
	//cudaMalloc((void**)&dens, sizeof(float)*size );
	//cudaMalloc((void**)&dens_prev, sizeof(float)*size );

	cudaMalloc((void**)&chemA, sizeof(float)*size );
	cudaMalloc((void**)&chemA_prev, sizeof(float)*size );
	cudaMalloc((void**)&chemB, sizeof(float)*size );
	cudaMalloc((void**)&chemB_prev, sizeof(float)*size );
	cudaMalloc((void**)&laplacian, sizeof(float)*size );

	runOnce = true;
}

// This is needed around every function so the compiler keeps the function name exactly, so Touch can find it.
extern "C" {
	
// This function tells Touch what version of Touch's CUDA API this .dll was written against (this is not the CUDA version)
DLLEXPORT int 
tCudaGetAPIVersion()
{
	// Always return this
	return TCUDA_API_VERSION;
}

// This function is called when a new node attaches itself to the dll.
DLLEXPORT void 
tCudaNodeAttached (const TCUDA_NodeInfo *info)
{
	initCUDA();

#ifdef _DEBUG
	printf("Node %s has attached itself to this .dll\n", info->node);
	switch (info->outputType)
	{
		case TCUDA_OUTPUT_TOP:
			printf("The Node is a TOP\n");
			break;
	}
#endif

}

// This function is called when a node is deleted or detaches itself from the dll (loads a different dll for example)
DLLEXPORT void 
tCudaNodeDetached(const TCUDA_NodeInfo *info)
{
#ifdef _DEBUG
	printf("Node %s has detached itself to this .dll\n", info->node);
	switch (info->outputType)
	{
		case TCUDA_OUTPUT_TOP:
			printf("The Node is a TOP\n");
			break;
	}
#endif
}

// In this function you need to fill in the structure with all of the requested information
// return false if you dont want to fill in this information (or alternatively you can not even declare this function)
// if you do either of the two above then the resolution/aspect will be set through normal TOP means.
// return true if you filled in the information
DLLEXPORT bool 
tCudaGetTOPOutputInfo(const TCUDA_NodeInfo *info, TCUDA_TOPOutputInfo *oinfo)
{
    // Filling this in just as an example, since we are returning false this data is ignored.
	oinfo->width = 512;
	oinfo->height = 512;
	oinfo->aspectX = 1;
	oinfo->aspectY = 1;
	oinfo->pixelFormat = TCUDA_PIXEL_FORMAT_RGBA8;

	return false;
}

DLLEXPORT void 
tCudaGetGeneralInfo(const TCUDA_NodeInfo *info, TCUDA_GeneralInfo *ginfo)
{
}

// This function tells us in what format the data will be outputted from the kernel.
DLLEXPORT void 
tCudaGetTOPKernelOutputInfo(const TCUDA_NodeInfo *info, TCUDA_TOPKernelOutputInfo *oinfo)
{
	oinfo->chanOrder = TCUDA_CHAN_BGRA;
	oinfo->dataFormat = TCUDA_DATA_FORMAT_UNSIGNED_BYTE;
}

// This function will be called once for every param that could potentially be passed into the cuda function
// The request structure contains information about the param, and it's up to you to decide
// if you are interested in this parameter or not. 
// If you are fill in the relevent fields in reqResult and return true
// if you arn't then just return false.
DLLEXPORT bool 
tCudaGetParamInfo(const TCUDA_NodeInfo *info, const TCUDA_ParamRequest *request, TCUDA_ParamRequestResult *reqResult)
{
#ifdef _DEBUG
	printf("Got Param Info Request for %s, input %d \n", request->name, request->inputNumber);
#endif
	if (request->dataType == TCUDA_DATA_TYPE_TOP)
	//if (request->dataType == TCUDA_DATA_TYPE_TOP && request->inputNumber == 0)
	{
		reqResult->top.dataFormat = TCUDA_DATA_FORMAT_UNSIGNED_BYTE;
		return true;
	}
	else // We are only interested in the first TOP right now
		 // we tell Touch that we arn't interested in other params or TOPs.
		 
	{
		return false;
	}


}

// Return the number of Info CHOP channels you want to create
DLLEXPORT int 
tCudaGetNumInfoCHOPChans(const TCUDA_NodeInfo *info)
{
    return 0;
}

// index is the Info CHOP channel whos data you should return
// name is the channel name. If you are allocating data for the name
// it is up to you to free it (or keep it around)
// value is the CHOP channel value
DLLEXPORT void 
tCudaGetInfoCHOPChan(const TCUDA_NodeInfo *info, int index,
			const char** name, float &value)
{
    /* example code */
    /*
    if (index == 0)
    {
	*name = "chan1";
	value = 0.5;
    }
    else if (index == 1)
    {
    	*name = "chan2";
	value = 0.7;
    }
    */
}

// return false if you arn't specifying Info DAT data, true if you are
// if you are, fill in the rows and cols values with the table
// dimensions
DLLEXPORT bool
tCudaGetInfoDATSize(const TCUDA_NodeInfo *info, int &rows, int &cols)
{
    return false;
}

// You return one column of table data per call, specified by colIndex
// values is an array of const char* pointers that you fill in with
// strings (the pointers are NULL, so create your own or use constant strings
DLLEXPORT void 
tCudaGetInfoDATColumn(const TCUDA_NodeInfo *info, int colIndex, const char **values)
{
    // example code
    /*
    // Assuming the number of rows is 2
    values[0] = "test";
    values[1] = "test2";
    */
}

}
