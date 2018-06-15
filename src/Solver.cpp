#include "Solver.h"

Solver::Solver(URBInputData* UID)
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