#pragma once

#include "URBInputData.h"
#include "SimulationParameters.h"
#include "Buildings.h"
#include "RectangularBuilding.h"
#include "Vector3.h"
#include "NetCDFData.h"
#include <math.h>

class Solver
{
protected:
	int nx, ny, nz;
	float dx, dy, dz;
	int itermax;

	float z0;                /// Surface roughness
    float z_ref;             /// Height of the measuring sensor (m)
    float U_ref;             /// Measured velocity at the sensor height (m/s)
    float H;                 /// Building height
    float W;                 /// Building width
    float L;                 /// Building length
    float x_start;           /// Building start location in x-direction
    float y_start;           /// Building start location in y-direction

    const int alpha1 = 1;        /// Gaussian precision moduli
    const int alpha2 = 1;        /// Gaussian precision moduli
    const float eta = pow(alpha1/alpha2, 2.0);
    const float A = pow(dx/dy, 2.0);
    const float B = eta*pow(dx/dz, 2.0);
    const float tol = 1e-9;     /// Error tolerance
    const float omega = 1.78;   /// Over-relaxation factor

public:
	Solver(URBInputData* UID)
	{
		Vector3<int> v;
		v = *(UID->simParams->domain);
		nx = v[0];
		ny = v[1];
		nz = v[2];
		Vector3<float> w;
		w = *(UID->simParams->grid);
		dx = w[0];
		dy = w[1];
		dz = w[2];
		itermax = UID->simParams->maxIterations;

		z0 = UID->buildings->wallRoughness;
		z_ref = UID->metParams->sensor->height;
		U_ref = UID->metParams->sensor->speed;

		if (UID->buildings->buildings[0]->buildingType == 1)
		{
			RectangularBuilding* rB = (RectangularBuilding*)UID->buildings->buildings[0];
			H = rB->height;
			W = rB->width;
			L = rB->length;
			x_start = rB->xFo;
			y_start = rB->yFo;
		}

	}

	virtual void solve(NetCDFData* netcdfDat) = 0;
};