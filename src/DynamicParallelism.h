#pragma once

/*
 * This is child class of the solver that runs the convergence
 * algorithm using Dynamic Parallelism on a single GPU.
 */

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "URBInputData.h"
#include "Solver.h"


/**
 *
 */
class DynamicParallelism : public Solver
{
private:

    template<typename T>
    void _cudaCheck(T e, const char* func, const char* call, const int line);

public:
    DynamicParallelism(const URBInputData* UID, URBGeneralData* UGD)
		: Solver(UID, UGD)
    {
    }

protected:
    float *d_e, *d_f, *d_g, *d_h, *d_m, *d_n;		/**< Solver coefficients on device (GPU) */
    double *d_R;              /**< Divergence of initial velocity field on device (GPU) */
    double *d_lambda, *d_lambda_old;		/**< Lagrange multipliers on device (GPU) */

    virtual void solve(const URBInputData* UID, URBGeneralData* UGD, bool solveWind);
};
