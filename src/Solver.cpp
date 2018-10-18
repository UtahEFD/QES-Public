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


void Solver::inputWindProfile(float dx, float dy, float dz, int nx, int ny, int nz, double *u0, double *v0, double *w0, int num_sites, int *site_blayer_flag, float *site_one_overL, float *site_xcoord, float *site_ycoord, float *site_wind_dir, float *site_z0, float *site_z_ref, float *site_U_ref, float *x, float *y, float *z)
	{

		double **u_prof, **v_prof;
		u_prof = new double* [num_sites];
		v_prof = new double* [num_sites];

		for (int i = 0; i < num_sites; i++){
			u_prof[i] = new double [nz];
			v_prof[i] = new double [nz];
		}

		float domain_rotation = 0.0;
		float theta = (domain_rotation*M_PI/180);
		float psi, x_temp, u_star; 
		float rc_sum, rc_value, xc, yc, rc, dn, lamda, s_gamma;
		float sum_wm, sum_wu, sum_wv;
		int iwork, jwork,rc_val;
		float dxx, dyy, u12, u34, v12, v34;
		float *u0_int, *v0_int, *site_theta;
		int icell_face, icell_cent;
		u0_int = new float [num_sites]();
		v0_int = new float [num_sites]();	
		site_theta = new float [num_sites]();
		float vk = 0.4;			/// Von Karman's constant

		for (int i = 0 ; i < num_sites; i++){
			site_theta[i] = (270.0-site_wind_dir[i])*M_PI/180.0;
			if (site_blayer_flag[i] == 0){
				for (int k = 1; k < nz; k++){
					u_prof[i][k] = 0.0;
					v_prof[i][k] = 0.0;
				}
			}

			if (site_blayer_flag[i] == 1){
				for (int k = 1; k < nz; k++){
					if (k == 1){
						if (site_z_ref[i]*site_one_overL[i] >= 0){
							psi = 4.7*site_z_ref[i]*site_one_overL[i];
						}
						else {
							x_temp = pow((1-15*site_z_ref[i]*site_one_overL[i]),0.25);
							psi = -2*log(0.5*(1+x_temp))-log(0.5*(1+pow(x_temp,2.0)))+2*atan(x_temp)-0.5*M_PI;
						}
					
						u_star = site_U_ref[i]*vk/(log((site_z_ref[i]+site_z0[i])/site_z0[i])+psi);
					}
					if (z[k]*site_one_overL[i] >= 0){
						psi = 4.7*z[k]*site_one_overL[i];
					}
					else {
						x_temp = pow((1-15*z[k]*site_one_overL[i]),0.25);
						psi = -2*log(0.5*(1+x_temp))-log(0.5*(1+pow(x_temp,2.0)))+2*atan(x_temp)-0.5*M_PI;
					}
	
					u_prof[i][k] = (cos(site_theta[i])*u_star/vk)*(log((z[k]+site_z0[i])/site_z0[i])+psi);
					v_prof[i][k] = (sin(site_theta[i])*u_star/vk)*(log((z[k]+site_z0[i])/site_z0[i])+psi);				
				}
			}

			if (site_blayer_flag[i] == 2){
				for (int k = 1; k < nz; k++){
					u_prof[i][k] = cos(site_theta[i])*site_U_ref[i]*pow((z[k]/site_z_ref[i]),site_z0[i]);
					v_prof[i][k] = sin(site_theta[i])*site_U_ref[i]*pow((z[k]/site_z_ref[i]),site_z0[i]);
				}
			}
		}

		double ***wm, ***wms;
		wm = new double** [num_sites]();
		wms = new double** [num_sites]();
		for (int i = 0; i < num_sites; i++){
			wm[i] = new double* [nx]();
			wms[i] = new double* [nx]();
			for (int j = 0; j < nx; j++){
				wm[i][j] = new double [ny]();
				wms[i][j] = new double [ny]();
			}
		}

		if (num_sites == 1){
			for ( int k = 0; k < nz; k++){
				for (int j = 0; j < ny; j++){
					for (int i = 0; i < nx; i++){
					
						icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values                                
		                u0[icell_face] = u_prof[0][k];
						v0[icell_face] = v_prof[0][k];
						w0[icell_face] = 0.0;         /// Perpendicular wind direction
					}
    	        }
    	    }
    	}
		else {
			rc_sum = 0.0;
			for (int i = 0; i < num_sites; i++){
				rc_val = 1000000.0;
				for (int ii = 0; ii < num_sites; ii++){
					xc = site_xcoord[ii] - site_xcoord[i];
					yc = site_ycoord[ii] - site_ycoord[i];
					rc = sqrt(pow(xc,2.0)+pow(yc,2.0));
					if (rc < rc_val && ii != i){
						rc_val = rc;
					}
				}
				rc_sum += rc_val;
			}
			dn = rc_sum/num_sites;
			lamda = 5.052*pow((2*dn/M_PI),2.0);
			s_gamma = 0.2;
			for (int j=0; j<ny; j++){
				for (int i=0; i<nx; i++){
					sum_wm = 0.0;
					for (int ii=0; ii<num_sites; ii++){
						wm[ii][i][j] = exp(-1/lamda*pow(site_xcoord[ii]-x[i]-dx,2.0)-1/lamda*pow(site_ycoord[ii]-y[j]-dy,2.0));
						wms[ii][i][j] = exp(-1/(s_gamma*lamda)*pow(site_xcoord[ii]-x[i]-dx,2.0)-1/(s_gamma*lamda)*pow(site_ycoord[ii]-y[j]-dy,2.0));
						sum_wm += wm[ii][i][j];
					}
					if (sum_wm == 0){
						for (int ii = 0; ii<num_sites; ii++){
							wm[ii][i][j] = 1e-20;
						}
					}
				}
			}
	
			for (int k=1; k<nz; k++){
				for (int j=0; j<ny; j++){
					for (int i=0; i<nx; i++){
						sum_wu = 0.0;
						sum_wv = 0.0;
						sum_wm = 0.0;
						for (int ii=0; ii<num_sites; ii++){
							sum_wu += wm[ii][i][j]*u_prof[ii][k];
							sum_wv += wm[ii][i][j]*v_prof[ii][k];	
							sum_wm += wm[ii][i][j];
						}
						icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values 
						u0[icell_face] = sum_wu/sum_wm;
						v0[icell_face] = sum_wv/sum_wm;	
						w0[icell_face] = 0.0;
					}
				}
	
				for (int ii=0; ii<num_sites; ii++){
					if(site_xcoord[ii]>0 && site_xcoord[ii]<(nx-1)*dx && site_ycoord[ii]>0 && site_ycoord[ii]>(ny-1)*dy){
						for (int j=0; j<ny; j++){
							if (y[j]<site_ycoord[ii]){
								jwork = j;
							}
						}
						for (int i=0; i<nx; i++){
							if (x[i]<site_xcoord[ii]){
								iwork = i;
							}
						} 
						dxx = site_xcoord[ii]-x[iwork];
						dyy = site_ycoord[ii]-y[jwork];
						int index_work = iwork+jwork*nx+k*nx*ny;
						u12 = (1-dxx/dx)*u0[index_work+nx]+(dxx/dx)*u0[index_work+1+nx];
						u34 = (1-dxx/dx)*u0[index_work]+(dxx/dx)*u0[index_work+1];
						u0_int[ii] = (dyy/dy)*u12+(1-dyy/dy)*u34;
		
						v12 = (1-dxx/dx)*v0[index_work+nx]+(dxx/dx)*v0[index_work+1+nx];
						v34 = (1-dxx/dx)*v0[index_work]+(dxx/dx)*v0[index_work+1];
						v0_int[ii] = (dyy/dy)*v12+(1-dyy/dy)*v34;
					}
					else{
						u0_int[ii] = u_prof[ii][k];
						v0_int[ii] = v_prof[ii][k];
					}
				}
	
				for (int j=0; j<ny; j++){
					for (int i=0; i<nx; i++){
						sum_wu = 0.0;
						sum_wv = 0.0;
						sum_wm = 0.0;
						for (int ii=0; ii<num_sites; ii++){
							sum_wu += wm[ii][i][j]*u_prof[ii][k];
							sum_wv += wm[ii][i][j]*v_prof[ii][k];	
							sum_wm += wm[ii][i][j];
						}
						if (sum_wm != 0){
							icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values 
							u0[icell_face] = u0[icell_face]+sum_wu/sum_wm;
							v0[icell_face] = v0[icell_face]+sum_wv/sum_wm;
						}	
					}
				}
			}
		}
	}

void Solver::defineWalls(int* iCellFlag, float* n, float* m, float* f, float* e, float* h, float* g, std::vector<std::vector<std::vector<float>>> x_cut, std::vector<std::vector<std::vector<float>>> y_cut, std::vector<std::vector<std::vector<float>>> z_cut, std::vector<std::vector<int>> num_points, std::vector<std::vector<float>> coeff)

{
    	for ( int k = 1; k < nz-2; k++){
    	    for (int j = 1; j < ny-2; j++){
    	        for (int i = 1; i < nx-2; i++){
					icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
	
					if (icellflag[icell_cent]==7){
						for (int ii=0; ii<6; ii++){
							coeff[icell_cent][ii] = 0;
							if (num_points[icell_cent][ii] !=0){
								/// calculate area fraction coeeficient for each face of the cut-cell
								for (int jj=0; jj<num_points[icell_cent][ii]-1; jj++){
									coeff[icell_cent][ii] += (0.5*(y_cut[icell_cent][ii][jj+1]+y_cut[icell_cent][ii][jj])*(z_cut[icell_cent][ii][jj+1]-z_cut[icell_cent][ii][jj]))/(dy*dz) + (0.5*(x_cut[icell_cent][ii][jj+1]+x_cut[icell_cent][ii][jj])*(z_cut[icell_cent][ii][jj+1]-z_cut[icell_cent][ii][jj]))/(dx*dz) + (0.5*(x_cut[icell_cent][ii][jj+1]+x_cut[icell_cent][ii][jj])*(y_cut[icell_cent][ii][jj+1]-y_cut[icell_cent][ii][jj]))/(dx*dy);
								}

								coeff[icell_cent][ii] += (0.5*(y_cut[icell_cent][ii][0]+y_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*(z_cut[icell_cent][ii][0]-z_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dy*dz) + (0.5*(x_cut[icell_cent][ii][0]+x_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*(z_cut[icell_cent][ii][0]-z_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dx*dz) + (0.5*(x_cut[icell_cent][ii][0]+x_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*(y_cut[icell_cent][ii][0]-y_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dx*dy);

							}
						}

					/// Assign solver coefficients
					f[icell_cent] = coeff[icell_cent][0];
					e[icell_cent] = coeff[icell_cent][1];
					h[icell_cent] = coeff[icell_cent][2];
					g[icell_cent] = coeff[icell_cent][3];
					n[icell_cent] = coeff[icell_cent][4];
					m[icell_cent] = coeff[icell_cent][5];
					}	
			
					if (icellflag[icell_cent] !=0) {
					
						/// Wall bellow
						if (icellflag[icell_cent-(nx-1)*(ny-1)]==0) {
			    			n[icell_cent] = 0.0; 

						}
						/// Wall above
						if (icellflag[icell_cent+(nx-1)*(ny-1)]==0) {
			    			m[icell_cent] = 0.0;
						}
						/// Wall in back
						if (icellflag[icell_cent-1]==0){
							f[icell_cent] = 0.0; 
						}
						/// Wall in front
						if (icellflag[icell_cent+1]==0){
							e[icell_cent] = 0.0; 
						}
						/// Wall on right
						if (icellflag[icell_cent-(nx-1)]==0){
							h[icell_cent] = 0.0;
						}
						/// Wall on left
						if (icellflag[icell_cent+(nx-1)]==0){
							g[icell_cent] = 0.0; 
						}
					}				 
				}
			}    
		}

		/// New boundary condition implementation
	for (int k = 0; k < nz-1; k++){
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




