#include "Solver.h"


#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void Solver::printProgress (float percentage)
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}

Solver::Solver(URBInputData* UID, DTEHeightField* DTEHF)
{

	rooftopFlag = UID->simParams->rooftopFlag;
	upwindCavityFlag = UID->simParams->upwindCavityFlag;
	streetCanyonFlag = UID->simParams->streetCanyonFlag;
	streetIntersectionFlag = UID->simParams->streetIntersectionFlag;
	wakeFlag = UID->simParams->wakeFlag;
	sidewallFlag = UID->simParams->sidewallFlag;

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
	dxy = MIN_S(dx, dy);

	num_sites = UID->metParams->sensor->num_sites;
	site_blayer_flag = UID->metParams->sensor->site_blayer_flag;
	site_one_overL = UID->metParams->sensor->site_one_overL;
	site_xcoord = UID->metParams->sensor->site_xcoord;
	site_ycoord = UID->metParams->sensor->site_ycoord;
	site_wind_dir = UID->metParams->sensor->site_wind_dir;

	site_z0 = UID->metParams->sensor->site_z0;
	site_z_ref = UID->metParams->sensor->site_z_ref;
	site_U_ref = UID->metParams->sensor->site_U_ref;
	Sensor1 = UID->metParams->sensor;

	//U_ref = UID->metParams->sensor->site_U_ref;

	if (UID->buildings)
	{

		z0 = UID->buildings->wallRoughness;

		for (int i = 0; i < UID->buildings->buildings.size(); i++)
		{
			if (UID->buildings->buildings[i]->buildingGeometry == 1)
			{
				buildings.push_back(UID->buildings->buildings[i]);
			}
		}
	}
	else
	{
		buildings.clear();
		z0 = 0.1f;
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

	//zm.push_back(-0.5*dzArray[0]);
	//z.push_back(0.0f);
	for (int k = 0; k < nz; k++)
	{
		z.push_back((k-1)*dz);
	} 

	mesh = 0;
	if (DTEHF)
		mesh = new Mesh(DTEHF->getTris());

	
	cells = 0;
	this->DTEHF = DTEHF;
	if (DTEHF)
		{
			cells = new Cell[(nx-1)*(ny-1)*(nz-1)];
			DTEHF->setCells(cells, nx - 1, ny - 1, nz - 1, dx, dy, dz);

		}



}

void Solver::defineWalls(int* iCellFlag, float* n, float* m, float* f, float* e, float* h, float* g)
{
	for (int k = 1; k < nz-2; k++){
		for (int j = 1; j < ny-2; j++){
			for (int i = 1; i < nx-2; i++){
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				if (iCellFlag[icell_cent] != 0) {
					
					/// Wall bellow
					if (iCellFlag[icell_cent-(nx-1)*(ny-1)]==0) {
						n[icell_cent] = 0.0f; 

					}
					/// Wall above
					if (iCellFlag[icell_cent+(nx-1)*(ny-1)]==0) {
						m[icell_cent] = 0.0f;
					}
					/// Wall in back
					if (iCellFlag[icell_cent-1]==0){
						f[icell_cent] = 0.0f; 
					}
					/// Wall in front
					if (iCellFlag[icell_cent+1]==0){
						e[icell_cent] = 0.0f; 
					}
					/// Wall on right
					if (iCellFlag[icell_cent-(nx-1)]==0){
						h[icell_cent] = 0.0f;
					}
					/// Wall on left
					if (iCellFlag[icell_cent+(nx-1)]==0){
						g[icell_cent] = 0.0f; 
					}
				}
			}
		}
	}

		/// New boundary condition implementation
	for (int k = 1; k < nz-1; k++){
		for (int j = 0; j < ny-1; j++){
			for (int i = 0; i < nx-1; i++){
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				e[icell_cent] /= (dx*dx);
				f[icell_cent] /= (dx*dx);
				g[icell_cent] /= (dy*dy);
				h[icell_cent] /= (dy*dy);
				m[icell_cent] /= (dz*dz);
				n[icell_cent] /= (dz*dz);
				//denom(:,:,k)=omegarelax/(e(:,:,k)+f(:,:,k)+g(:,:,k)+h(:,:,k)+m(:,:,k)+n(:,:,k))
			}
		}
	}
}


