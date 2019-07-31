#pragma once

/*
 * This is child class of the solver that runs the convergence
 * algorithm in serial order on a CPU.
 */

#include "URBInputData.h"
#include "Solver.h"
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
    CPUSolver(const URBInputData* UID, URBGeneralData* UGD)
        : Solver(UID, UGD)
    {
    }
    
    virtual void solve(const URBInputData* UID, URBGeneralData* ugd, bool solveWind);
};
