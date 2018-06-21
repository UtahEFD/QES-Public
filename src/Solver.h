#pragma once

#include "URBInputData.h"
#include "SimulationParameters.h"
#include "Buildings.h"
#include "RectangularBuilding.h"
#include "Vector3.h"
#include "NetCDFData.h"
#include <math.h>
#include <vector>


class Solver
{
protected:
	int nx, ny, nz;
	float dx, dy, dz;
    std::vector<float> dzArray, zm;
    std::vector<float> x,y,z;
	int itermax;

    float z_ref;             /// Height of the measuring sensor (m)
    float U_ref;             /// Measured velocity at the sensor height (m/s)
    float z0;
    std::vector<Building*> buildings;

    const int alpha1 = 1;        /// Gaussian precision moduli
    const int alpha2 = 1;        /// Gaussian precision moduli
    const float eta = pow(alpha1/alpha2, 2.0);
    const float A = pow(dx/dy, 2.0);
    const float B = eta*pow(dx/dz, 2.0);
    const float tol = 1e-9;     /// Error tolerance
    const float omega = 1.78;   /// Over-relaxation factor

public:
	Solver(URBInputData* UID);

	virtual void solve(NetCDFData* netcdfDat) = 0;
};