#pragma once

/*
 * This is child class of the solver that runs the convergence
 * algorithm in serial order on a CPU.
 */

#include "URBInputData.h"
#include "Solver.h"
#include "NetCDFData.h"
#include "DTEHeightField.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>


using namespace std;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;

class CPUSolver : public Solver
{
public:
	CPUSolver(URBInputData* UID, DTEHeightField* DTEHF)
		: Solver(UID, DTEHF)
		{

		}

	virtual void solve(NetCDFData* netcdfDat, bool solveWind, bool cellFace);

};