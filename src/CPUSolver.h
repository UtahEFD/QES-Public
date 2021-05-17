#pragma once

/*
 * This is child class of the solver that runs the convergence
 * algorithm in serial order on a CPU.
 */

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>

#include "WINDSInputData.h"
#include "Solver.h"


class CPUSolver : public Solver
{
public:
    CPUSolver(const WINDSInputData* WID, WINDSGeneralData* WGD)
        : Solver(WID, WGD)
    {
    }
protected:

    virtual void solve(const WINDSInputData* WID, WINDSGeneralData* WGD, bool solveWind);
};
