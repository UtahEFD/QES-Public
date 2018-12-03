#pragma once

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
	CPUSolver(const URBInputData* UID, const DTEHeightField* DTEHF)
		: Solver(UID, DTEHF)
		{
                    /// Total number of cell-centered values in domain
                    long numcell_cent = (nx-1)*(ny-1)*(nz-1); 
                    icellflag = new int [ numcell_cent ];
		}

	virtual void solve(bool solveWind);

    void outputDataFile();
    void outputNetCDF( NetCDFData* netcdfDat );

private:
    int *icellflag;
};
