#include "Solver.h"


#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

/**< This function is showing progress of the solving process by printing the percentage */

void Solver::printProgress (float percentage)
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}


/**< \fn Solver
* This function is assigning values read by URBImputData to variables used in the solvers
 */

Solver::Solver(URBInputData* UID, DTEHeightField* DTEHF)
{

	rooftopFlag = UID->simParams->rooftopFlag;		
	upwindCavityFlag = UID->simParams->upwindCavityFlag;		
	streetCanyonFlag = UID->simParams->streetCanyonFlag;		
	streetIntersectionFlag = UID->simParams->streetIntersectionFlag;		
	wakeFlag = UID->simParams->wakeFlag;		
	sidewallFlag = UID->simParams->sidewallFlag;		

	Vector3<int> D;
	D = *(UID->simParams->domain);
	nx = D[0];      /**< number of cells in x-direction */
	ny = D[1];		/**< number of cells in y-direction */
	nz = D[2];		/**< number of cells in z-direction */

	nx += 1;        /// +1 for Staggered grid
	ny += 1;        /// +1 for Staggered grid
	nz += 2;        /// +2 for staggered grid and ghost cell


	Vector3<float> G;
	G = *(UID->simParams->grid);
	dx = G[0];		/**< Grid resolution in x-direction */
	dy = G[1];		/**< Grid resolution in y-direction */
	dz = G[2];		/**< Grid resolution in z-direction */
	itermax = UID->simParams->maxIterations;		
	dxy = MIN_S(dx, dy);		
		
	numcell_cent = (nx-1)*(ny-1)*(nz-1);         /**< Total number of cell-centered values in domain */
    numface_cent = nx*ny*nz;                     /**< Total number of face-centered values in domain */

	num_sites = UID->metParams->num_sites;
	for (int i=0; i<num_sites; i++){
		site_blayer_flag.push_back (UID->metParams->sensors[i]->site_blayer_flag);
		site_one_overL.push_back (UID->metParams->sensors[i]->site_one_overL);
		site_xcoord.push_back (UID->metParams->sensors[i]->site_xcoord);
		site_ycoord.push_back (UID->metParams->sensors[i]->site_ycoord);
		site_wind_dir.push_back (UID->metParams->sensors[i]->site_wind_dir);
		site_z0.push_back (UID->metParams->sensors[i]->site_z0);
		site_z_ref.push_back (UID->metParams->sensors[i]->site_z_ref);
		site_U_ref.push_back (UID->metParams->sensors[i]->site_U_ref);
	}


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
		z.push_back((k-0.5)*dz);
	} 

    for ( int i = 0; i < nx-1; i++)
        x.push_back((i+0.5)*dx);         /**< Location of face centers in x-dir */

    for ( int j = 0; j < ny-1; j++){
        y.push_back((j+0.5)*dy);         /**< Location of face centers in y-dir */
    }

    for ( int k = 0; k < nz-1; k++){
        for (int j = 0; j < ny-1; j++){
            for (int i = 0; i < nx-1; i++){
				e.push_back(1.0);
				f.push_back(1.0);	
				g.push_back(1.0);
				h.push_back(1.0);
				m.push_back(1.0);
				n.push_back(1.0);
				R.push_back(0.0);
				icellflag.push_back(1);
				lambda.push_back(0.0);
				lambda_old.push_back(0.0);
			}
		}    
	}	


    for ( int k = 0; k < nz; k++){
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){

				u0.push_back(0.0);
				v0.push_back(0.0);
				w0.push_back(0.0);
				u.push_back(0.0);
				v.push_back(0.0);
				w.push_back(0.0);
			}
		}    
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


