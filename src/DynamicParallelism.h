#pragma once

#include "URBInputData.h"
#include "Solver.h"
#include "NetCDFData.h"
#include "DTEHeightField.h"
#include "RectangularBuilding.h"
#include "Sensor.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include "cuda.h"

/**
 *
 */
class DynamicParallelism : public Solver
{
private:

    template<typename T>
    void _cudaCheck(T e, const char* func, const char* call, const int line);

protected:
    float *d_e, *d_f, *d_g, *d_h, *d_m, *d_n;		/**< Solver coefficients on device (GPU) */
    double *d_R;              /**< Divergence of initial velocity field on device (GPU) */
    double *d_lambda, *d_lambda_old;		/**< Lagrange multipliers on device (GPU) */

public:
    DynamicParallelism(URBInputData* UID, DTEHeightField* DTEHF)
        : Solver(UID, DTEHF)
    {
    }

    virtual void solve(bool solveWind);

    void outputDataFile();
    void outputNetCDF( NetCDFData* netcdfDat );

};
