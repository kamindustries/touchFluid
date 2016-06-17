/* 
	CUDA kernels and functions
	Kurt Kaminski 2016
*/


#ifndef __FLUID__
#define __FLUID__

#include <iostream>
#include <map>

#include "common.cuh"
#include "private/TCUDA_Types.h"

using namespace std;

class Fluid
{
public:
	Fluid();
	virtual ~Fluid();

	// Methods
	void init(int xRes, int yRes);
	void getFromUI(float* inDensity, float* inObstacle);
	void step(float* inDensity, float* inObstacle);
	void clearArrays();
	void makeColor();
	void makeColorLong(float* output);

	// Data
	float *chemA, *chemA_prev, *chemB, *chemB_prev, *laplacian;
	float *vel[2], *vel_prev[2];
	float *pressure, *pressure_prev;
	float *temperature, *temperature_prev;
	float *divergence;
	float *boundary;
	float *newObstacle; //incoming obstacles and velocities

	// Global constants
	dim3 grid, threads;
	int dimX, dimY;
	float dt;
	int nDiff;
	int nReact;
	int nJacobi;

	// Advection constants
	float velDiff;
	float tempDiff;
	float densDiff;
	float curlAmt;
	float buoy;
	float weight;
	float diff;
	float visc;
	float force;
	float source_density;
	float source_temp;

	// RD constants
	float F;
	float k;
	float dA;
	float dB;
	float xLen;
	float yLen;
	int rdEquation;
	float e;

protected:


	void reactDiffAdvect(float* inObstacle);

};

#endif