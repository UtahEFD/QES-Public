#include "Solver.h"

Solver::Solver(URBInputData* UID)
{
	Vector3<int> v;
	v = *(UID->simParams->domain);
	nx = v[0];
	ny = v[1];
	nz = v[2];

    nx += 1;        /// +1 for Staggered grid
    ny += 1;        /// +1 for Staggered grid
    nz += 2;        /// +2 for staggered grid and ghost cell

	
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
	{
		if (UID->buildings->buildings[i]->buildingGeometry == 1)
		{
			buildings.push_back(UID->buildings->buildings[i]);
		}
	}

	int j = 0;
	for (int i = 0; i < nz; i++)
	{
		if (UID->simParams->verticalStretching == 0)
			dzArray.push_back(dz);
		else
			dzArray.push_back(UID->simParams->dzArray[j]);

		if (i != 0 && i != nz - 2)
			j++;
	}

	zm.push_back(-0.5*dzArray[0]);
	z.push_back(0.0f);
	for (int i = 1; i < nz; i++)
	{
		z.push_back(z[i - 1] + dzArray[i]);
		zm.push_back(z[i] - 0.5f * dzArray[i]);
	} 

}