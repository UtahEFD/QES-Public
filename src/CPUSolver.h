#pragma once

/*
 * This is child class of the solver that runs the convergence
 * algorithm in serial order on a CPU.
 */

#include "URBInputData.h"
#include "Solver.h"
#include "Output.hpp"
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
	CPUSolver(const URBInputData* UID, URBGeneralData* UGD, Output* output)
		: Solver(UID, UGD, output)
		{
		}

	virtual void solve(const URBInputData* UID, const URBGeneralData* ugd, bool solveWind);
};
