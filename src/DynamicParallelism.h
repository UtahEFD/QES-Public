#pragma once

#include "URBInputData.h"
#include "Solver.h"
#include "NetCDFData.h"
//#include "DynamicParallelism.cu"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>


#define BLOCKSIZE 1024
#define Blocksize_x 16
#define Blocksize_y 16
#define Blocksize_z 4
#define cudaCheck(x) _cudaCheck(x, #x ,__FILE__, __LINE__)


using namespace std::chrono;
using namespace std;
using std::ofstream;
using std::ifstream;
using std::istringstream;
using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;
using std::to_string;
__device__ double error;


// Divergence kernel
extern void divergence(double *uin, double *vin, double *win, double *R_out, int alpha1, int  nx, int  ny, int nz, float dx,float dy, float dz);

extern void SOR_RB(double *d_lambda, int nx, int ny, int nz, float omega, float  A, float  B, float  dx, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, float *d_o, float *d_p, float *d_q, double *d_R, int offset);

extern void assign_lambda_to_lambda_old(double *d_lambda, double *d_lambda_old, int nx, int ny, int nz);

extern void applyNeumannBC(double *d_lambda, int nx, int ny);

extern void calculateError(double *d_lambda, double *d_lambda_old, int nx, int ny, int nz, double *d_value, double *d_bvalue);

// Euler Final Velocity kernel
extern void finalVelocity(double *uin, double *vin, double *win, double *lambdain, double *uf, double *vf,double *wf, int alpha1, float dx, float dy, float dz, int  nx, int  ny, int nz);

extern void SOR_iteration(double *d_lambda, double *d_lambda_old, int nx, int ny, int nz, float omega, float  A, float  B, float  dx, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, float *d_o, float *d_p, float *d_q, double *d_R, int itermax, double tol, double *d_value, double *d_bvalue, double *d_u0, double *d_v0, double *d_w0,int alpha1, float dy, float dz, double *d_u, double *d_v, double *d_w);

	template<typename T>
extern void _cudaCheck(T e, const char* func, const char* call, const int line);

class DynamicParallelism : public Solver
{
private:


public:
	DynamicParallelism(URBInputData* UID)
		: Solver(UID)
		{

		}

	virtual void solve(NetCDFData* netcdfDat);

};