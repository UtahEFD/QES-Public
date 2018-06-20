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

	z_ref = UID->metParams->sensor->height;
	U_ref = UID->metParams->sensor->speed;
	z0 = UID->buildings->wallRoughness;

	for (int i = 0; i < UID->buildings->buildings.size(); i++)
	if (UID->buildings->buildings[i]->buildingType == 1)
	{
		buildings.push_back(UID->buildings->buildings[i]);
	}

}