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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include "cuda.h"
#include <string.h>

#define BLOCKSIZE 1024
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




class DynamicParallelism : public Solver
{
private:

	template<typename T>
	void _cudaCheck(T e, const char* func, const char* call, const int line);

public:
	DynamicParallelism(URBInputData* UID, DTEHeightField* DTEHF)
		: Solver(UID, DTEHF)
		{

		}

	virtual void solve(NetCDFData* netcdfDat, bool solveWind);

<<<<<<< HEAD
};
=======
};
>>>>>>> origin/doxygenAdd
