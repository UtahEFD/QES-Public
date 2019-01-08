#pragma once

/*
 * This is child class of the solver that runs the convergence
 * algorithm in serial order on a CPU.
 */

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

using namespace std;

class CPUSolver : public Solver
{
public:
    CPUSolver(URBInputData* UID, DTEHeightField* DTEHF)
        : Solver(UID, DTEHF)
    {
    }
  
    virtual void solve(bool solveWind);

    void outputDataFile();
    void outputNetCDF( NetCDFData* netcdfDat );
};
