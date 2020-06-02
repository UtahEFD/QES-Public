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

#include "URBInputData.h"
#include "Solver.h"


class CPUSolver : public Solver
{
public:
    CPUSolver(const URBInputData* UID, URBGeneralData* UGD)
        : Solver(UID, UGD)
    {
    }
protected:

    virtual void solve(const URBInputData* UID, URBGeneralData* UGD, bool solveWind);
};
