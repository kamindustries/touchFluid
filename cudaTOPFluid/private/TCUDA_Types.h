/* Shared Use License: This file is owned by Derivative Inc. (Derivative) and
 * can only be used, and/or modified for use, in conjunction with 
 * Derivative's TouchDesigner software, and only if you are a licensee who has
 * accepted Derivative's TouchDesigner license or assignment agreement (which
 * also govern the use of this file).  You may share a modified version of this
 * file with another authorized licensee of Derivative's TouchDesigner software.
 * Otherwise, no redistribution or sharing of this file, with or without
 * modification, is permitted.
 */

/*
 * Produced by:
 *		
 *		Derivative Inc
 *		Toronto, Ontario
 *		Canada   M5V 3A8
 *		416-591-3555
 *
 * NAME:		CUDA_Types.h 
 *
 * COMMENTS: 		The size of these structures is very important since .dlls assume
 *				a certain size based on the API version they are written against.
 *				If you make a change that changes the size of a structure, make sure
 *				you know what you are doing.
 */

#ifndef __CUDA_Types__
#define __CUDA_Types__

#include <string>
#include <cuda_runtime_api.h>
//#include <driver_types.h>

#include "../defines.h"
//#include "../common.cuh"

typedef enum
{
    TCUDA_DATA_TYPE_PARAM = 0,	// Node parameter, i.e the float values list on parameter
    							// page of the CUDA OP, data will be an array of 4 floats

    TCUDA_DATA_TYPE_CHOP = 1,	// CHOP node, data will be an array of floats

    TCUDA_DATA_TYPE_TOP = 2,	// TOP node, data will be array of image data,
    							// unsigned chars, or flaots

    TCUDA_DATA_TYPE_OBJ = 3,	// Object node, data will be a structure
    							// One of		TCUDA_GeoTransformInfo
    							// 				TCUDA_CameraTransformInfo 
    							//				TCUDA_CameraParameterInfo

    TCUDA_DATA_TYPE_STR = 4,	// String, data will be a null terminated array of unsigned chars
} TCUDA_DataType;

typedef enum
{
    TCUDA_PROJECTION_TYPE_ORTHO = 0,
    TCUDA_PROJECTION_TYPE_PERSPECTIVE = 1
} TCUDA_ProjectionType;

typedef enum
{
    TCUDA_OBJ_SUB_TYPE_CAMERA = 0,	// The object is Camera
    TCUDA_OBJ_SUB_TYPE_GEO = 1,		// The object is a geo
} TCUDA_ObjSubType;


typedef enum
{
   TCUDA_DATA_FORMAT_UNSIGNED_BYTE = 0, // 8-bit unsigned char
   TCUDA_DATA_FORMAT_FLOAT = 1,			// 32-bit float
} TCUDA_DataFormat;

// The basic types here should match the types in TCUDA_DataFormat.
typedef enum
{
   TCUDA_PARAM_DATA_FORMAT_UNSIGNED_BYTE = 0,
   TCUDA_PARAM_DATA_FORMAT_FLOAT = 1,

   // Non-basic type
   TCUDA_PARAM_DATA_FORMAT_STRUCT = 100,
} TCUDA_ParamDataFormat;

typedef enum
{
    TCUDA_DATA_LOCATION_DEVICE = 0,
    TCUDA_DATA_LOCATION_HOST = 1,
} TCUDA_DataLocation;

typedef enum
{
    TCUDA_CHAN_BGRA = 0,
    TCUDA_CHAN_A = 1,

} TCUDA_ChanOrder;

typedef enum
{
    TCUDA_OUTPUT_TOP = 0,
} TCUDA_OutputType;

typedef enum
{
    // NOTE: These to not imply the channel order, the TCUDA_ChanOrder enum does this
    TCUDA_PIXEL_FORMAT_RGBA8 = 0, 	// 8-bit per pixel fixed point
    TCUDA_PIXEL_FORMAT_RGBA32 = 1 	// 32-bits per pixel floating point
} TCUDA_PixelFormat;

typedef enum
{
    TCUDA_OBJ_PARAM_TYPE_TRANSFORMS = 0,
    TCUDA_OBJ_PARAM_TYPE_PARAMETERS = 1,
} TCUDA_ObjParamType;

typedef enum
{
    TCUDA_FOG_TYPE_OFF = 0,
    TCUDA_FOG_TYPE_LINEAR = 1,
    TCUDA_FOG_TYPE_EXPONENTIAL = 2,
    TCUDA_FOG_TYPE_EXPONENTIAL_SQUARED = 3,
} TCUDA_FogType;

typedef enum
{
    TCUDA_MEM_TYPE_LINEAR = 0,
    TCUDA_MEM_TYPE_ARRAY = 1,
} TCUDA_MemType;

struct TCUDA_NodeInfo
{
    const char			*node;		// This will be the name of the node
    unsigned int		 uniqueNodeId;  // This is the unique id of the node
    TCUDA_OutputType	 outputType;// This is the type of data this node outputs


    int					 reserved[9];
} ;

struct TCUDA_GeneralInfo
{
    bool	timeDependent;	// If you set this to true, the node will cook next frame
    						// regardless if any if it's input change,
    						// if it's false (default), it will only cook if
    						// if one or more of it's inputs change.

    
    int		reserved[9];
} ;


struct TCUDA_TOPOutputInfo
{
    int					 width;		// The width of the TOP
    int					 height;	// The height of the TOP
    float				 aspectX;		// The aspectx of the TOP
    float				 aspectY;		// The aspecty of the TOP
    TCUDA_PixelFormat	 pixelFormat;	// The pixel format of the top (8-bit RGBA etc.)

    
    int					 reserved[7];
} ;

struct TCUDA_TOPKernelOutputInfo
{
    TCUDA_ChanOrder 	chanOrder;		// The order that the channels are layed out in memory
    TCUDA_DataFormat	dataFormat;		// The data format the output will be in memory

    
    int					reserved[7];
};

struct TCUDA_ParamRequest
{
    const char*		 name;		// The full path to the node, or the name of the parameter
    TCUDA_DataType	 dataType;		// The type of data the parameter will be
    unsigned int	 inputNumber;	// The input number, this number restarts at 0
    									// for each type of parameter

    int				 reserved[7];
    
    union
    {
		struct
		{
		    int		numValues;	// The number of values in the parameter

		    
		    int		reserved[9];
		} param;
		struct
		{
		    int		numChannels;	// The number of channels in the CHOP
		    int		length;			// The number of samples in the CHOP
		    float	sampleRate; 	// The sample rate of the CHOP

		    
		    int				reserved[7];
		} chop;
		struct
		{
		    int				width;		// The width of the TOP
		    int				height;		// The height of the TOP
		    TCUDA_ChanOrder chanOrder;	// The channel order of the TOP

		    
		    int				reserved[7];
		} top;
		struct
		{
		    TCUDA_ObjSubType	subType; // Differentiats between Geos, Cameras, lights etc.

		    
		    int				reserved[9];
		} obj;
		struct
		{
		    int				reserved[10];
		} string;
		    
    } ;

};

struct TCUDA_ParamRequestResult
{
    int reserved[9];
    union
    {
		struct
		{
		    int 		reserved[9];
		} param;
		struct 
		{
		    TCUDA_DataLocation	dataLocation;	// Tells me where you want the data to be
		    									// located (CPU or GPU)
		    int			reserved[8];
		} chop;
		struct
		{
		    TCUDA_DataFormat	dataFormat;	// The data format you want the TOP data in
		    								// float, or 8-bit unsigned chars
		    TCUDA_MemType 		memType;	// The type of CUDA memory to alloc


		    int					reserved[7];
		} top;
		struct
		{
		    // You will get multiple parameters from Touch in your tCudaExecuteKernel call
		    // if you set more than one of these to true
		    bool		interestedInTransforms;	// set to true if you are interested in the node's
		    									// transform data. 
		    									// The 'data' you get will be a structure like
		    									// TCUDA_CameraTransformInfo or
		    									// TCUDA_GeoTransformInfo, depending
		    									// On the Object type of the COMP.

		    bool		interestedInParameters;	// set to true if you are interested in the node's
		    									// subtype specific parameters
		    									// for example for camera you will get a
		    									// TCUDA_CameraParameterInfo structure of data

		    union 
			{
				struct 
				{
				    int		projWidth;	// When calculating the projection matrix
				    int projHeight;		// the w and h of the rendered image affects
				    					// the calculations, these values default to
				    					// the w/h of the output TOP
				} cam;
		    };
		    int 		reserved[6];
		} obj;
		struct
		{
		    int				reserved[9];
		} string;
    } ;
    
};

struct TCUDA_ParamInfo
{
    const char*			 name;			// The full path to the node or the name 
    									// of the paremeter.
    
    TCUDA_DataType       dataType;		// The data type, param, TOP, CHOP etc.

    TCUDA_ParamDataFormat	dataFormat;	// The format of the data that the 'data'
    									// pointer points to, can be an array of
    									// floats for example, or a structure
    
    TCUDA_DataLocation   dataLocation;	// Where the data is located, on the CUDA device
    									// or on the host CPU

    void                *data;			// Pointer to the data

    int					inputNumber;	// The index number of this param, 0 based
    									// for each dataType.

    int					reserved[5];
    void				*touchUseOnly;	// This one is for use by Touch only

    union 
    {
		struct
		{
		    int numValues;

		    
		    int		reserved[9];
		} param;
		struct
		{
		    int 		numChannels;
		    int 		length;
		    float 		sampleRate; 		// The sample rate of the CHOP

		    
		    int		reserved[7];
		} chop;
		struct 
		{
		    int 						width;
		    int 						height;
		    TCUDA_ChanOrder 			chanOrder;
		    TCUDA_MemType				memType;	// The memory type of the memory

		    cudaChannelFormatDesc		*desc;		// The channel desc
													// This will be NULL if memType
													// is not TCUDA_MEM_TYPE_ARRAY

		    
		    int		reserved[5];
		} top;
		struct
		{
		    TCUDA_ObjSubType	 subType;	// Lets you know what kind of Object this is
		    								// geometry, camera etc.

		    TCUDA_ObjParamType	 paramType;	// Tells you if this param is the Transform
		    								// or parameters from the Object
		    int		reserved[8];
		} obj;
		struct
		{
		    int		 reserved[10];
		} string;
		
    };
};

struct matrix3x3
{
    float3 m[3];
};

struct matrix4x3
{
    float4 m[3];
};

struct matrix4x4
{
    float4 m[4];
};

// These structures are what the 'data' member of TCUDA_ParamInfo is
// in the case of Object COMPs with transform data
struct TCUDA_GeoTransformInfo
{
    matrix4x4		localTransform;
    matrix4x4		localITransform;
    matrix4x4		worldTransform;
    matrix4x4		worldITransform;
} ;

struct TCUDA_CameraTransformInfo
{
    matrix4x4		localTransform;
    matrix4x4		localITransform;
    matrix4x4		worldTransform;
    matrix4x4		worldITransform;
    matrix4x4		projTransform;
    matrix4x4		projITransform;
};

struct TCUDA_CameraParameterInfo
{
    TCUDA_ProjectionType		projection;
    float		orthoWidth;
    float		nearPlane;
    float		farPlane;
    float 		focalLength;
    float		aperture;
    float		winx;
    float		winy;
    float		winroll;
    float		winsize;
    float4		bgColor;
    TCUDA_FogType		fogType;
    float		fogDensity;
    float		fogNear;
    float		fogFar;
    float4		fogColor;
};

#ifdef WIN32 
typedef void (__cdecl *TCUDAGETGENERALINFO) (const TCUDA_NodeInfo *, TCUDA_GeneralInfo *ginfo);
typedef void (__cdecl *TCUDANODEATTACHED) (const TCUDA_NodeInfo *);
typedef bool (__cdecl *TCUDAGETTOPOUTPUTINFO) (const TCUDA_NodeInfo *, TCUDA_TOPOutputInfo*);
typedef int  (__cdecl *TCUDAGETAPIVERSION) (void);
typedef bool (__cdecl *TCUDAGETPARAMINFO) (const TCUDA_NodeInfo *, const TCUDA_ParamRequest *request, TCUDA_ParamRequestResult *reqResult);
typedef bool (__cdecl *TCUDAEXECUTEKERNEL)(const TCUDA_NodeInfo *, const int nparams, const TCUDA_ParamInfo **params, const TCUDA_ParamInfo *output);
typedef void (__cdecl *TCUDAGETTOPKERNELOUTPUTINFO) (const TCUDA_NodeInfo *, TCUDA_TOPKernelOutputInfo *kinfo);
typedef void (__cdecl *TCUDANODEDETACHED) (const TCUDA_NodeInfo *);
typedef int (__cdecl *TCUDAGETNUMINFOCHOPCHANS) (const TCUDA_NodeInfo *);
typedef void (__cdecl *TCUDAGETINFOCHOPCHAN)(const TCUDA_NodeInfo *, int index,
												const char**, float &value);
typedef bool (__cdecl *TCUDAGETINFODATSIZE)(const TCUDA_NodeInfo *,
												int &rows,
												int &cols);
typedef void (__cdecl *TCUDAGETINFODATCOLUMN)(const TCUDA_NodeInfo *,
												int colIndex,
												const char **values);
#endif 

#endif

