#include "Solver.h"

// duplication of this macro
#define CELL(i,j,k,w) ((i) + (j) * (nx+(w)) + (k) * (nx+(w)) * (ny+(w)))

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
	mesh_type_flag = UID->simParams->meshTypeFlag;

	Vector3<int> domainInfo;
	domainInfo = *(UID->simParams->domain);
	nx = domainInfo[0];
	ny = domainInfo[1];
	nz = domainInfo[2];

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
	

	Vector3<float> gridInfo;
	gridInfo = *(UID->simParams->grid);
	dx = gridInfo[0];
	dy = gridInfo[1];
	dz = gridInfo[2];
	itermax = UID->simParams->maxIterations;
	dxy = MIN_S(dx, dy);

	if (UID->metParams)
	{
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
	}

	if (UID->canopies)			// If there are canopies specified in input files
	{
		num_canopies = UID->canopies->num_canopies;
		landuse_flag = UID->canopies->landuse_flag;
		landuse_veg_flag = UID->canopies->landuse_veg_flag;
		landuse_urb_flag = UID->canopies->landuse_urb_flag;
		for (int i = 0; i < UID->canopies->canopies.size(); i++)
		{
			canopies.push_back(UID->canopies->canopies[i]);		// Add a new canopy element
		}
	}

        /// Total number of face-centered values in domain
        long numface_cent = nx*ny*nz;
        u.resize(numface_cent);
        v.resize(numface_cent);
        w.resize(numface_cent);

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

	for (int k = 0; k < nz-1; k++)
	{
		z.push_back((k-0.5)*dz);		/**< Location of face centers in x-dir */
		zm.push_back(k*dz);
	} 

    for ( int i = 0; i < nx-1; i++)
	{
        x.push_back((i+0.5)*dx);         /**< Location of face centers in x-dir */
	}

    for ( int j = 0; j < ny-1; j++)
	{
        y.push_back((j+0.5)*dy);         /**< Location of face centers in y-dir */
    }

	/// Initializing variables
    for ( int k = 0; k < nz-1; k++)
	{
        for (int j = 0; j < ny-1; j++)
		{
            for (int i = 0; i < nx-1; i++)
			{
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

				u_out.push_back(0.0);
				v_out.push_back(0.0);
				w_out.push_back(0.0);

			}
		}    
	}	

	// Initialize u0,v0,w0,u,v and w
    for ( int k = 0; k < nz; k++)
	{
        for (int j = 0; j < ny; j++)
		{
            for (int i = 0; i < nx; i++)
			{
                u0.push_back(0.0);
				v0.push_back(0.0);
				w0.push_back(0.0);
				u.push_back(0.0);
				v.push_back(0.0);
				w.push_back(0.0);
			}
		}    
	}	


    max_velmag = 0.0f;
    for (int i = 0; i < nx; i++)
       for ( int j = 1; j < ny; j++)
          max_velmag = MAX(max_velmag , sqrt( pow(u0[CELL(i,j,nz,0)], 2) + pow(v0[CELL(i,j,nz,0)],2) ));
    max_velmag = 1.2 * max_velmag;


	// Calling inputWindProfile function to generate initial velocity field from sensors information (located in Sensor.cpp)
	sensor->inputWindProfile(dx, dy, dz, nx, ny, nz, u0.data(), v0.data(), w0.data(), num_sites, site_blayer_flag.data(), 
							site_one_overL.data(), site_xcoord.data(), site_ycoord.data(), site_wind_dir.data(),
							site_z0.data(), site_z_ref.data(), site_U_ref.data(), x.data(), y.data(), z.data());



	mesh = 0;
	if (DTEHF)
	{
		mesh = new Mesh(DTEHF->getTris());
		if (mesh_type_flag == 0)			// Stair-step (original QUIC)
		{
			if (mesh)
			{
				std::cout << "Creating terrain blocks...\n";
				for (int i = 0; i < nx; i++)
				{
					for (int j = 0; j < ny; j++)
					{      //get height, then add half a cell, if the height exceeds half of a cell partially, it will round up.
						float heightToMesh = mesh->getHeight(i * dx + dx * 0.5f, j * dy + dy * 0.5f) + 0.5f * dz;
						for (int k = 0; k < (int)(heightToMesh / dz); k++)
							buildings.push_back(new RectangularBuilding(i * dx, j * dy, k * dz, dx, dy, dz));
					}
					printProgress( (float)i / (float)nx);
				}
				std::cout << "blocks created\n";
			}
		}
		else							// Cut-cell method
		{
			// Calling calculateCoefficient function to calculate area fraction coefficients for cut-cells
			cut_cell->calculateCoefficient(cells, DTEHF, nx, ny, nz, dx, dy, dz, n, m, f, e, h, g, pi, icellflag);
		}
	}


	/////////////////////////////////////////////////////////////
	//      Apply canopy vegetation parameterization           //   
	/////////////////////////////////////////////////////////////
    
	if (num_canopies>0)
	{

		std::vector<std::vector<std::vector<float>>> canopy_atten(nx-1, std::vector<std::vector<float>>(ny-1, std::vector<float>(nz-1,0.0)));
		std::vector<std::vector<float>> canopy_top(nx-1, std::vector<float>(ny-1,0.0));
		std::vector<std::vector<float>> canopy_top_index(nx-1, std::vector<float>(ny-1,0.0));
		std::vector<std::vector<float>> canopy_z0(nx-1, std::vector<float>(ny-1,0.0));
		std::vector<std::vector<float>> canopy_ustar(nx-1, std::vector<float>(ny-1,0.0));
		std::vector<std::vector<float>> canopy_d(nx-1, std::vector<float>(ny-1,0.0));

		// Read in canopy information
		canopy->readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag, canopy_atten, canopy_top);

		for (int i=0; i<canopies.size();i++)
		{
			((Canopy*)canopies[i])->defineCanopy(dx, dy, dz, nx, ny, nz, icellflag.data(), num_canopies, lu_canopy_flag, 
													canopy_atten, canopy_top);			// Defininf canopy bounderies
		}
		canopy->plantInitial(nx, ny, nz, vk, icellflag.data(), z, u0, v0, canopy_atten, canopy_top, canopy_top_index, 
													canopy_ustar, canopy_z0, canopy_d);		// Apply canopy parameterization
	}
   

	/////////////////////////////////////////////////////////////
	//                Apply building effect                    //   
	/////////////////////////////////////////////////////////////

	if (mesh_type_flag == 0)
	{
		for (int i = 0; i < buildings.size(); i++)
		{
			((RectangularBuilding*)buildings[i])->setCellsFlag(dx, dy, dz, nx, ny, nz, icellflag.data(), mesh_type_flag);
        
		}
		
		std::cout << "Defining Solid Walls...\n";
		/// Boundary condition for building edges
		defineWalls(dx,dy,dz,nx,ny,nz, icellflag.data(), n.data(), m.data(), f.data(), e.data(), h.data(), g.data());
		std::cout << "Walls Defined...\n";
	}
	else
	{	
		std::vector<std::vector<std::vector<float>>> x_cut(numcell_cent, std::vector<std::vector<float>>(6, std::vector<float>(6,0.0)));
		std::vector<std::vector<std::vector<float>>> y_cut(numcell_cent, std::vector<std::vector<float>>(6, std::vector<float>(6,0.0)));
		std::vector<std::vector<std::vector<float>>> z_cut(numcell_cent, std::vector<std::vector<float>>(6, std::vector<float>(6,0.0)));
	
		std::vector<std::vector<int>> num_points(numcell_cent, std::vector<int>(6,0));
		std::vector<std::vector<float>> coeff(numcell_cent, std::vector<float>(6,0.0));
	
	
    	for (int i = 0; i < buildings.size(); i++)
    	{
			((RectangularBuilding*)buildings[i])->setCellsFlag(dx, dy, dz, nx, ny, nz, icellflag.data(), mesh_type_flag);
    	    ((RectangularBuilding*)buildings[i])->setCutCells(dx, dy, dz, nx, ny, nz, icellflag.data(), x_cut, y_cut, z_cut, 
																num_points, coeff);    /// located in RectangularBuilding.h
    	    
	    }
		
		std::cout << "Defining Solid Walls...\n";
		/// Boundary condition for building edges
		defineWalls(dx, dy, dz, nx, ny, nz, icellflag.data(), n.data(), m.data(), f.data(), e.data(), h.data(), g.data(), 
					x_cut, y_cut, z_cut, num_points, coeff);
		std::cout << "Walls Defined...\n";
	}

	/// defining ground solid cells
	for (int j = 0; j < ny-1; j++)
	{
		for (int i = 0; i < nx-1; i++)
		{
			int icell_cent = i + j*(nx-1);
			icellflag[icell_cent] = 0.0;
		}
	}


    for (int k = 0; k < nz; k++)
	{
        for (int j = 0; j < ny; j++)
		{
            for (int i = 0; i < nx; i++)
			{
				icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);		/// Lineralized index for cell centered values
                icell_face = i + j*nx + k*nx*ny;			/// Lineralized index for cell faced values 
				if (icellflag[icell_cent] == 0) {
					u0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
					u0[icell_face+1] = 0.0;
					v0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
					v0[icell_face+nx] = 0.0;
					w0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
					w0[icell_face+nx*ny] = 0.0;
				}
	     	}
		}
	}


    /// New boundary condition implementation
    for (int k = 0; k < nz-1; k++)
	{
        for (int j = 0; j < ny-1; j++)
		{
            for (int i = 0; i < nx-1; i++)
			{
				icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				e[icell_cent] = e[icell_cent]/(dx*dx);
				f[icell_cent] = f[icell_cent]/(dx*dx);
				g[icell_cent] = g[icell_cent]/(dy*dy);
				h[icell_cent] = h[icell_cent]/(dy*dy);
				m[icell_cent] = m[icell_cent]/(dz*dz);
				n[icell_cent] = n[icell_cent]/(dz*dz);
			}
		}
	}
	 



}



void Solver::defineWalls(float dx, float dy, float dz, int nx, int ny, int nz, int* icellflag, float* n, float* m, float* f, 
						float* e, float* h, float* g, std::vector<std::vector<std::vector<float>>> x_cut,
						std::vector<std::vector<std::vector<float>>>y_cut,std::vector<std::vector<std::vector<float>>> z_cut, 
						std::vector<std::vector<int>> num_points, std::vector<std::vector<float>> coeff)

{

	/*std::vector<std::vector<float>> x_centroid((nx-1)*(ny-1)*(nz-1), std::vector<float>(6,0.0));
	std::vector<std::vector<float>> y_centroid((nx-1)*(ny-1)*(nz-1), std::vector<float>(6,0.0));
	std::vector<std::vector<float>> z_centroid((nx-1)*(ny-1)*(nz-1), std::vector<float>(6,0.0));
	std::vector<std::vector<std::vector<float>>> angle((nx-1)*(ny-1)*(nz-1), std::vector<std::vector<float>>(6, std::vector<float>(6,0.0)));
	float sum_x, sum_y, sum_z;
	std::vector<float> angle_temp (6, 0.0);
	std::vector<float> angle_max (6, 0.0);
	std::vector<float> xcut_temp (6, 0.0);
	std::vector<float> ycut_temp (6, 0.0);
	std::vector<float> zcut_temp (6, 0.0);
	std::vector<int> imax (6,0);

	for ( int k = 1; k < nz-2; k++){
    	for (int j = 1; j < ny-2; j++){
	    	for (int i = 1; i < nx-2; i++){
				icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
	
				if (icellflag[icell_cent]==7){
					for (int ii=0; ii<6; ii++){
						if (num_points[icell_cent][ii] !=0){
							sum_x = 0;
							sum_y = 0;
							sum_z = 0;
							for (int jj=0; jj<num_points[icell_cent][ii]; jj++){
								sum_x += x_cut[icell_cent][ii][jj];
								sum_y += y_cut[icell_cent][ii][jj];
								sum_z += z_cut[icell_cent][ii][jj];
							}
							x_centroid[icell_cent][ii] = sum_x/num_points[icell_cent][ii];
							y_centroid[icell_cent][ii] = sum_y/num_points[icell_cent][ii];
							z_centroid[icell_cent][ii] = sum_z/num_points[icell_cent][ii];
						}
					}
				}
			}
		}
	}

	for ( int k = 1; k < nz-2; k++){
	    for (int j = 1; j < ny-2; j++){
	   		for (int i = 1; i < nx-2; i++){
				icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
	
				if (icellflag[icell_cent]==7){
					for (int ii=0; ii<6; ii++){
						if (num_points[icell_cent][ii] !=0){
							for (int jj=0; jj<num_points[icell_cent][ii]; jj++){
								if (ii==0 || ii==1){
									angle[icell_cent][ii][jj] = (180/pi)*atan2((z_cut[icell_cent][ii][jj]-z_centroid[icell_cent][ii]),(y_cut[icell_cent][ii][jj]-y_centroid[icell_cent][ii]));
								}
								if (ii==2 || ii==3){
									angle[icell_cent][ii][jj] = (180/pi)*atan2((z_cut[icell_cent][ii][jj]-z_centroid[icell_cent][ii]),(x_cut[icell_cent][ii][jj]-x_centroid[icell_cent][ii]));
								}
								if (ii==4 || ii==5){
									angle[icell_cent][ii][jj] = (180/pi)*atan2((y_cut[icell_cent][ii][jj]-y_centroid[icell_cent][ii]),(x_cut[icell_cent][ii][jj]-x_centroid[icell_cent][ii]));
								}
							}
						}
					}
				}
			}
		}
	}


        // Write data to file
        ofstream outdata3;
        outdata3.open("test.dat");
        if( !outdata3 ) {                 // File couldn't be opened
            cerr << "Error: file could not be opened" << endl;
            exit(1);
        }
        // Write data to file
        for (int k = 0; k < nz-1; k++){
            for (int j = 0; j < ny-1; j++){
                for (int i = 0; i < nx-1; i++){
    				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
    				int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values
					for (int ii=0; ii<6; ii++){
                    	outdata3 << "\t" << i << "\t" << j << "\t" << k << "\t \t"<< x[i] << "\t \t" << y[j] << "\t \t" << z[k] << "\t \t"<< "\t \t" << angle[icell_cent][ii][0] <<"\t \t"<< "\t \t"<<angle[icell_cent][ii][1]<<"\t \t"<< "\t \t"<<angle[icell_cent][ii][2]<< "\t \t"<< "\t \t" << angle[icell_cent][ii][3] <<"\t \t"<< "\t \t"<<angle[icell_cent][ii][4]<<"\t \t"<< "\t \t"<<angle[icell_cent][ii][5]<<"\t \t"<<icellflag[icell_cent]<< endl;
					}   
                }
            }
        }
        outdata3.close();



	for ( int k = 1; k < nz-2; k++){
    	for (int j = 1; j < ny-2; j++){
	    	for (int i = 1; i < nx-2; i++){
				icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
	
				if (icellflag[icell_cent]==7){
					for (int ii=0; ii<6; ii++){
						if (num_points[icell_cent][ii] !=0){
							for (int jj=0; jj<num_points[icell_cent][ii]; jj++){
								imax[jj] = jj;
								angle_max[jj] = -180;
								xcut_temp [jj] = x_cut[icell_cent][ii][jj];
								ycut_temp [jj] = y_cut[icell_cent][ii][jj];
								zcut_temp [jj] = z_cut[icell_cent][ii][jj];
								angle_temp [jj] = angle[icell_cent][ii][jj];
							}

							for (int iii=0; iii<num_points[icell_cent][ii]; iii++){
								for (int jjj=0; jjj<num_points[icell_cent][ii]; jjj++){
									if (angle[icell_cent][ii][jjj] > angle_max[iii]){
										angle_max[iii] = angle[icell_cent][ii][jjj];
										imax[iii] = jjj;
									}
								}
								angle[icell_cent][ii][imax[iii]] = -999;
							}

							for (int jj=0; jj<num_points[icell_cent][ii]; jj++){
								x_cut[icell_cent][ii][jj] = xcut_temp[imax[num_points[icell_cent][ii]-1-jj]];
								y_cut[icell_cent][ii][jj] = ycut_temp[imax[num_points[icell_cent][ii]-1-jj]];
								z_cut[icell_cent][ii][jj] = zcut_temp[imax[num_points[icell_cent][ii]-1-jj]];
								angle[icell_cent][ii][jj] = angle_temp[imax[num_points[icell_cent][ii]-1-jj]];
								
							}
							
						}
					}
				}
			}
		}
	}

       // Write data to file
        ofstream outdata4;
        outdata4.open("test1.dat");
        if( !outdata4 ) {                 // File couldn't be opened
            cerr << "Error: file could not be opened" << endl;
            exit(1);
        }
        // Write data to file
        for (int k = 0; k < nz-1; k++){
            for (int j = 0; j < ny-1; j++){
                for (int i = 0; i < nx-1; i++){
    				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
    				int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values
					for (int ii=0; ii<6; ii++){
                    	outdata4 << "\t" << i << "\t" << j << "\t" << k << "\t \t"<< x[i] << "\t \t" << y[j] << "\t \t" << z[k] << "\t \t"<< "\t \t" << angle[icell_cent][ii][0] <<"\t \t"<< "\t \t"<<angle[icell_cent][ii][1]<<"\t \t"<< "\t \t"<<angle[icell_cent][ii][2]<< "\t \t"<< "\t \t" << angle[icell_cent][ii][3] <<"\t \t"<< "\t \t"<<angle[icell_cent][ii][4]<<"\t \t"<< "\t \t"<<angle[icell_cent][ii][5]<<"\t \t"<<icellflag[icell_cent]<< endl;
					}   
                }
            }
        }
        outdata4.close();*/

	for ( int k = 1; k < nz-2; k++)
	{
		for (int j = 1; j < ny-2; j++)
		{
			for (int i = 1; i < nx-2; i++)
			{
				icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
	
				if (icellflag[icell_cent]==7)
				{
					for (int ii=0; ii<6; ii++)
					{
						coeff[icell_cent][ii] = 0;
						if (num_points[icell_cent][ii] !=0)
						{
							/// calculate area fraction coeeficient for each face of the cut-cell
							for (int jj=0; jj<num_points[icell_cent][ii]-1; jj++)
							{
								coeff[icell_cent][ii] += (0.5*(y_cut[icell_cent][ii][jj+1]+y_cut[icell_cent][ii][jj])*
														(z_cut[icell_cent][ii][jj+1]-z_cut[icell_cent][ii][jj]))/(dy*dz) + 
														(0.5*(x_cut[icell_cent][ii][jj+1]+x_cut[icell_cent][ii][jj])*
														(z_cut[icell_cent][ii][jj+1]-z_cut[icell_cent][ii][jj]))/(dx*dz) + 
														(0.5*(x_cut[icell_cent][ii][jj+1]+x_cut[icell_cent][ii][jj])*
														(y_cut[icell_cent][ii][jj+1]-y_cut[icell_cent][ii][jj]))/(dx*dy);
							}

				coeff[icell_cent][ii] +=(0.5*(y_cut[icell_cent][ii][0]+y_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*
									(z_cut[icell_cent][ii][0]-z_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dy*dz)+ 
									(0.5*(x_cut[icell_cent][ii][0]+x_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*
									(z_cut[icell_cent][ii][0]-z_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dx*dz)+ 
									(0.5*(x_cut[icell_cent][ii][0]+x_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*
									(y_cut[icell_cent][ii][0]-y_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dx*dy);

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

				if (icellflag[icell_cent] !=0) 
				{
					
					/// Wall bellow
					if (icellflag[icell_cent-(nx-1)*(ny-1)]==0) 
					{
			    		n[icell_cent] = 0.0; 
					}
					/// Wall above
					if (icellflag[icell_cent+(nx-1)*(ny-1)]==0) 
					{
		    			m[icell_cent] = 0.0;
					}
					/// Wall in back
					if (icellflag[icell_cent-1]==0)
					{
						f[icell_cent] = 0.0; 
					}
					/// Wall in front
					if (icellflag[icell_cent+1]==0)
					{
						e[icell_cent] = 0.0; 
					}
					/// Wall on right
					if (icellflag[icell_cent-(nx-1)]==0)
					{
						h[icell_cent] = 0.0;
					}
					/// Wall on left
					if (icellflag[icell_cent+(nx-1)]==0)
					{
						g[icell_cent] = 0.0; 
					}
				}
			}    
		}
	}
}


void Solver::defineWalls(float dx, float dy, float dz, int nx, int ny, int nz, int* icellflag, float* n, float* m, float* f, 
						float* e, float* h, float* g)

{

	for (int i=0; i<nx-1; i++)
	{
		for (int j=0; j<ny-1; j++)
		{
			for (int k=0; k<nz-1; k++)
			{
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
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
}		

