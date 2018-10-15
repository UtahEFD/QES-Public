#pragma once

#include "URBInputData.h"
#include "Solver.h"
#include "NetCDFData.h"
#include "DTEHeightField.h"
<<<<<<< HEAD
#include "RectangularBuilding.h"
#include "Sensor.h"
#include "inputWindProfile.h"
=======
>>>>>>> origin/doxygenAdd
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
<<<<<<< HEAD
    

	virtual void solve(NetCDFData* netcdfDat, bool solveWind);

};
=======

	virtual void solve(NetCDFData* netcdfDat, bool solveWind);

};
>>>>>>> origin/doxygenAdd
