#include "Solver.h"


// This now exists in multiple locations after the merge!!!! We need
// to consolidate!
#define CELL(i,j,k) ((i) + (j) * (nx) + (k) * (nx) * (ny))

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



void Solver::upWind(Building* build, int* iCellFlag, double* u0, double* v0, double* w0, float* z, float* zm)
{

		 int perpendicular_flag,ns_flag;
		 int upIstart,upIstop,upJstart,upJstop;
		 float uo_h,vo_h,upwind_dir,upwind_rel,xco,yco;
		 std::vector<float> x,y;
		 float xf1,yf1,xf2,yf2,tol,ynorm,lfcoeff;
		 float zf,x_u,y_u,x_v,y_v,x_w,y_w;
		 float xs_u,xs_v,xs_w,xv_u,xv_v,xv_w,xrz_u,xrz_v;
		 float urot,vrot,uhrot,vhrot,vel_mag;
		 float vortex_height,build_width,retarding_factor;
		 float length_factor,height_factor,rz_end,retarding_height,eff_height;
		 //float totalLength,perpendicularDir,gamma_eff;
		 int ktop,kbottom,iface,ivert/*,x_idx,y_idx*/;

		 if(build->buildingGeometry == 4)
		 {
			eff_height=0.8*(build->height - build->baseHeightActual ) + build->baseHeightActual;
			/*  This building geometry doesn't exist yet. Update this when it does.
			 xco = (RectangularBuilding*)(build)->xfo + (RectangularBuilding*)(build)->length*cos((RectangularBuilding*)(build)->rotation) //!CENTER of building in QUIC domain coordinates
			 yco = (RectangularBuilding*)(build)->yfo + (RectangularBuilding*)(build)->length*sin((RectangularBuilding*)(build)->rotation)
			*/
		 }
		 else if (build->buildingGeometry == 6)
		 {
			eff_height = build->height;
			xco = build->centroidX;
			yco = build->centroidY;
		 }
		 else //must be 1 which is rectangular building
		 {
			 eff_height = build->height;
			 xco = ((RectangularBuilding*)(build))->xFo + ((RectangularBuilding*)(build))->length*cos(((RectangularBuilding*)(build))->rotation); //!CENTER of building in QUIC domain coordinates
			 yco = ((RectangularBuilding*)(build))->yFo + ((RectangularBuilding*)(build))->length*sin(((RectangularBuilding*)(build))->rotation);
		 }
		 
		 // find upwind direction and deterMIN_Se the type of flow regime
		 uo_h = u0[ CELL((int)(xco/dx), (int)(yco/dy), build->kEnd+1, 0)];
		 vo_h = v0[ CELL((int)(xco/dx), (int)(yco/dy), build->kEnd+1, 0)];
		 upwind_dir = atan2(vo_h,uo_h);
		 upwind_rel = upwind_dir - build->rotation;
		 uhrot = uo_h * cos(build->rotation) + vo_h * sin(build->rotation);
		 vhrot = -uo_h * sin(build->rotation) + vo_h * cos(build->rotation);
		 vel_mag = sqrt( (uo_h * uo_h) + ( vo_h * vo_h) );
		 tol = 10 * pi / 180.0f;
		 retarding_factor = 0.4f;
		 length_factor = 0.4f;
		 height_factor = 0.6;

		 if(upwindCavityFlag == 1)
			lfcoeff=2;
		 else
			lfcoeff=1.5;

		 if( upwind_rel > pi) upwind_rel = upwind_rel - 2 * pi;
		 if(upwind_rel < -pi) upwind_rel = upwind_rel + 2 * pi;


		 if(build->buildingGeometry == 6)
		 {
			/*
				NOTE::buildingGeo 6 isn't being implemented right now, so just leave this blank.

			allocate(LfFace(bldstopidx(ibuild)-bldstartidx(ibuild)),LengthFace(bldstopidx(ibuild)-bldstartidx(ibuild)))
			iface=0
			do ivert=bldstartidx(ibuild),bldstopidx(ibuild)
			   x1=0.5*(bldx(ivert)+bldx(ivert+1))
			   y1=0.5*(bldy(ivert)+bldy(ivert+1))
			   xf1=(bldx(ivert)-x1)*cos(upwind_dir)+(bldy(ivert)-y1)*sin(upwind_dir)
			   yf1=-(bldx(ivert)-x1)*sin(upwind_dir)+(bldy(ivert)-y1)*cos(upwind_dir)
			   xf2=(bldx(ivert+1)-x1)*cos(upwind_dir)+(bldy(ivert+1)-y1)*sin(upwind_dir)
			   yf2=-(bldx(ivert+1)-x1)*sin(upwind_dir)+(bldy(ivert+1)-y1)*cos(upwind_dir)
			   upwind_rel=atan2(yf2-yf1,xf2-xf1)+0.5*pi
			   if(upwind_rel .gt. pi)upwind_rel=upwind_rel-2*pi
			   if(abs(upwind_rel) .gt. pi-tol)then
				  perpendicularDir=atan2(bldy(ivert+1)-bldy(ivert),bldx(ivert+1)-bldx(ivert))+0.5*pi
				  if(perpendicularDir.le.-pi)perpendicularDir=perpendicularDir+2*pi
				  if(abs(perpendicularDir) .ge. 0.25*pi .and. abs(perpendicularDir) .le. 0.75*pi)then
					 ns_flag=1
				  else
					 ns_flag=0
				  endif
				  gamma_eff=perpendicularDir
				  if(gamma_eff .ge. 0.75*pi)then
					 gamma_eff=gamma_eff-pi
				  elseif(gamma_eff .ge. 0.25*pi)then
					 gamma_eff=gamma_eff-0.5*pi
				  elseif(gamma_eff .lt. -0.75*pi)then
					 gamma_eff=gamma_eff+pi
				  elseif(gamma_eff .lt. -0.25*pi)then
					 gamma_eff=gamma_eff+0.5*pi
				  endif
				  uhrot=uo_h*cos(gamma_eff)+vo_h*sin(gamma_eff)
				  vhrot=-uo_h*sin(gamma_eff)+vo_h*cos(gamma_eff)
				  iface=iface+1
				  LengthFace(iface)=sqrt(((xf2-xf1)**2.)+((yf2-yf1)**2.))
				  LfFace(iface)=abs(lfcoeff*LengthFace(iface)*cos(upwind_rel)/(1+0.8*LengthFace(iface)/eff_height))
				  if(upwindflag .eq. 3)theniCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
					 vortex_height=MIN_S(LengthFace(iface),eff_height)
					 retarding_height=eff_height
				  else
					 vortex_height=eff_height
					 retarding_height=eff_height
				  endif
				  ! MAN 07/25/2008 stretched vertical grid
				  do k=2,kstart(ibuild)
					 kbottom=k
					 if(zfo(ibuild) .le. zm(k))exit
				  enddo
				  do k=kstart(ibuild),nz-1
					 ktop=k
					 if(height_factor*retarding_height+zfo_actual(ibuild) .le. z(k))exit
				  enddo
				  upIstart=MAX_S(nint(MIN_S(bldx(ivert),bldx(ivert+1))/dx)-nint(1.5*LfFace(iface)/dx),2)
				  upIstop=MIN_S(nint(MAX_S(bldx(ivert),bldx(ivert+1))/dx)+nint(1.5*LfFace(iface)/dx),nx-1)
				  upJstart=MAX_S(nint(MIN_S(bldy(ivert),bldy(ivert+1))/dy)-nint(1.5*LfFace(iface)/dy),2)
				  upJstop=MIN_S(nint(MAX_S(bldy(ivert),bldy(ivert+1))/dy)+nint(1.5*LfFace(iface)/dy),ny-1)
				  ynorm=abs(yf2)
				  do k=kbottom,ktop
					 zf=zm(k)-zfo(ibuild)
					 do j=upJstart,upJstop
						do i=upIstart,upIstop
						   x_u=((real(i)-1)*dx-x1)*cos(upwind_dir)+ &
										((real(j)-0.5)*dy-y1)*sin(upwind_dir)
						   y_u=-((real(i)-1)*dx-x1)*sin(upwind_dir)+ &
										((real(j)-0.5)*dy-y1)*cos(upwind_dir)
						   x_v=((real(i)-0.5)*dx-x1)*cos(upwind_dir)+ &
										((real(j)-1)*dy-y1)*sin(upwind_dir)
						   y_v=-((real(i)-0.5)*dx-x1)*sin(upwind_dir)+	&
										((real(j)-1)*dy-y1)*cos(upwind_dir)
						   x_w=((real(i)-0.5)*dx-x1)*cos(upwind_dir)+ &
										((real(j)-0.5)*dy-y1)*sin(upwind_dir)
						   y_w=-((real(i)-0.5)*dx-x1)*sin(upwind_dir)+	&
										((real(j)-0.5)*dy-y1)*cos(upwind_dir)
!u values
						   if(abs(y_u) .le. ynorm .and. height_factor*vortex_height .gt. zf)then
							  xs_u=((xf2-xf1)/(yf2-yf1))*(y_u-yf1)+xf1
							  xv_u=-LfFace(iface)*sqrt((1-((y_u/ynorm)**2.))*(1-((zf/(height_factor*vortex_height))**2.)))
							  xrz_u=-LfFace(iface)*sqrt((1-((y_u/ynorm)**2.))*(1-((zf/(height_factor*retarding_height))**2.)))
							  if(zf .gt. height_factor*vortex_height)then
								 rz_end=0.
							  else
								 rz_end=length_factor*xv_u
							  endif
							  if(upwindflag .eq. 1)then
								 if(x_u-xs_u .ge. xv_u .and. x_u-xs_u .le. 0.1*dxy .and. iCellFlag(i,j,k) .ne. 0)then
									uo(i,j,k)=0.
								 endif
							  else
								 if(x_u-xs_u .ge. xrz_u .and. x_u-xs_u .lt. rz_end &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									if(upwindflag .eq. 3)then
									   uo(i,j,k)=((x_u-xs_u-xrz_u)*(retarding_factor-1.)/(rz_end-xrz_u)+1.)*uo(i,j,k)
									else
									   uo(i,j,k)=retarding_factor*uo(i,j,k)
									endif
									if(abs(uo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized U exceeds MAX_S in upwind',&
										  uo(i,j,k),max_velmag,i,j,k
									endif
								 endif
								 if(x_u-xs_u .ge. length_factor*xv_u .and. x_u-xs_u .le. 0.1*dxy &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									urot=uo(i,j,k)*cos(gamma_eff)
									vrot=-uo(i,j,k)*sin(gamma_eff)
									if(ns_flag .eq. 1)then
									   vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
											*(-height_factor*sin(((pi*abs(x_u-xs_u))/(length_factor*LfFace(iface)))+0))
									else
									   urot=-uhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
											*(-height_factor*sin(((pi*abs(x_u-xs_u))/(length_factor*LfFace(iface)))+0))
									endif
									uo(i,j,k)=urot*cos(-gamma_eff)+vrot*sin(-gamma_eff)
									if(abs(uo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized U exceeds MAX_S in upwind',&
										  uo(i,j,k),max_velmag,i,j,k
									endif
								 endif
							  endif
						   endif
!v values
						   if(abs(y_v) .le. ynorm .and. height_factor*vortex_height .gt. zf)then
							  xs_v=((xf2-xf1)/(yf2-yf1))*(y_v-yf1)+xf1
							  xv_v=-LfFace(iface)*sqrt((1-((y_v/ynorm)**2.))*(1-((zf/(height_factor*vortex_height))**2.)))
							  xrz_v=-LfFace(iface)*sqrt((1-((y_v/ynorm)**2.))*(1-((zf/(height_factor*retarding_height))**2.)))
							  if(zf .ge. height_factor*vortex_height)then
								 rz_end=0.
							  else
								 rz_end=length_factor*xv_v
							  endif
							  if(upwindflag .eq. 1)then
								 if(x_v-xs_v .ge. xv_v .and. x_v-xs_v .le. 0.1*dxy .and. iCellFlag(i,j,k) .ne. 0)then
									vo(i,j,k)=0.
								 endif
							  else
								 if(x_v-xs_v .ge. xrz_v .and. x_v-xs_v .lt. rz_end &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									if(upwindflag .eq. 3)then
									   vo(i,j,k)=((x_v-xs_v-xrz_v)*(retarding_factor-1.)/(rz_end-xrz_v)+1.)*vo(i,j,k)
									else
									   vo(i,j,k)=retarding_factor*vo(i,j,k)
									endif
									if(abs(vo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized V exceeds MAX_S in upwind',&
										  vo(i,j,k),max_velmag,i,j,k
									endif
								 endif
								 if(x_v-xs_v .ge. length_factor*xv_v .and. x_v-xs_v .le. 0.1*dxy &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									urot=vo(i,j,k)*sin(gamma_eff)
									vrot=vo(i,j,k)*cos(gamma_eff)
									if(ns_flag .eq. 1)then
									   vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
											*(-height_factor*sin(((pi*abs(x_v-xs_v))/(length_factor*LfFace(iface)))+0))
									else
									   urot=-uhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
											*(-height_factor*sin(((pi*abs(x_v-xs_v))/(length_factor*LfFace(iface)))+0))
									endif
									vo(i,j,k)=-urot*sin(-gamma_eff)+vrot*cos(-gamma_eff)
									if(abs(vo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized V exceeds MAX_S in upwind',&
										  vo(i,j,k),max_velmag,i,j,k
									endif
								 endif
							  endif
						   endif
!w values
						   if(abs(y_w) .le. ynorm .and. height_factor*vortex_height .gt. zf)then
							  xs_w=((xf2-xf1)/(yf2-yf1))*(y_w-yf1)+xf1
							  xv_w=-LfFace(iface)*sqrt((1-((y_w/ynorm)**2.))*(1-((zf/(height_factor*vortex_height))**2.)))
							  if(upwindflag .eq. 1)then
								 if(x_w-xs_w .ge. xv_w .and. x_w-xs_w .le. 0.1*dxy .and. iCellFlag(i,j,k) .ne. 0)then
									wo(i,j,k)=0.
									if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
									   iCellFlag(i,j,k)=2
									endif
								 endif
							  else
								 if(x_w-xs_w .ge. xv_w .and. x_w-xs_w .lt. length_factor*xv_w &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									wo(i,j,k)=retarding_factor*wo(i,j,k)
									if(abs(wo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized W exceeds MAX_S in upwind',&
										  wo(i,j,k),max_velmag,i,j,k
									endif
									if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
									   iCellFlag(i,j,k)=2
									endif
								 endif
								 if(x_w-xs_w .ge. length_factor*xv_w .and. x_w-xs_w .le. 0.1*dxy &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									wo(i,j,k)=-vel_mag*(0.1*cos(((pi*abs(x_w-xs_w))/(length_factor*LfFace(iface))))-0.05)
									if(abs(wo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized W exceeds MAX_S in upwind',&
										  wo(i,j,k),max_velmag,i,j,k
									endif
									if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
									   iCellFlag(i,j,k)=2
									endif
								 endif
							  endif
						   endif
						enddo
					 enddo
				  enddo
			   endif
			   if(bldx(ivert+1) .eq. bldx(bldstartidx(ibuild)) &
					 .and. bldy(ivert+1) .eq. bldy(bldstartidx(ibuild)))exit
			enddo
			if(iface .gt. 0)then
			   totalLength=0.
			   Lf(ibuild)=0.
			   do ivert=1,iface
				  Lf(ibuild)=Lf(ibuild)+LfFace(ivert)*LengthFace(ivert)
				  totalLength=totalLength+LengthFace(ivert)
			   enddo
			   Lf(ibuild)=Lf(ibuild)/totalLength
			else
			   Lf(ibuild)=-999.0
			endif
			deallocate(LfFace,LengthFace)
			*/
			build->iStart += 1; //dummy code for build, throw out on implementation
		 }
		 else
			//Location of corners relative to the center of the building
			x.push_back(((NonPolyBuilding*)build)->xFo + ((NonPolyBuilding*)build)->width * sin((build)->rotation) - xco);
			y.push_back(((NonPolyBuilding*)build)->yFo  - ((NonPolyBuilding*)build)->width * cos(build->rotation) - yco);
			x.push_back(x[0] + ((NonPolyBuilding*)build)->length * cos(build->rotation));
			y.push_back(y[0] + ((NonPolyBuilding*)build)->length * sin(build->rotation));
			x.push_back(((NonPolyBuilding*)build)->xFo - ((NonPolyBuilding*)build)->width * sin(build->rotation) - xco);
			y.push_back(((NonPolyBuilding*)build)->yFo + ((NonPolyBuilding*)build)->width * cos(build->rotation) - yco);
			x.push_back(x[2] + ((NonPolyBuilding*)build)->length * cos(build->rotation));
			y.push_back(y[2] + ((NonPolyBuilding*)build)->length * sin(build->rotation));


			//flip the last two values to maintain order
			float tempx, tempy;
			tempx = x[3];
			tempy = y[3];
			x[3] = x[2];
			y[3] = y[2];
			x[2] = tempx;
			y[2] = tempy;



			perpendicular_flag = 0;

			int num = -1;
			if(upwind_rel > 0.5 * pi - tol && upwind_rel < 0.5 * pi + tol )
			{ 
			  num = 2;
			   perpendicular_flag=1;
			   ns_flag=1;
			   ((NonPolyBuilding*)build)->length = abs(lfcoeff * ((NonPolyBuilding*)build)->length * sin(upwind_rel) / ( 1 + 0.8 * ((NonPolyBuilding*)build)->length / eff_height));
			   build_width=((NonPolyBuilding*)build)->length;
			}
			else if(upwind_rel > -tol && upwind_rel > tol)
			{
			   num = 1;
			   perpendicular_flag=1;
			   ns_flag=0;
			   ((NonPolyBuilding*)build)->length = abs(lfcoeff * ((NonPolyBuilding*)build)->width * cos(upwind_rel) / ( 1 + 0.8 * ((NonPolyBuilding*)build)->width / eff_height));
			   build_width=((NonPolyBuilding*)build)->width;
		   }
			else if(upwind_rel > -0.5 * pi - tol && upwind_rel < -0.5 * pi + tol)
			{
			   num = 4;
			   perpendicular_flag=1;
			   ns_flag=1;
			   ((NonPolyBuilding*)build)->length = abs(lfcoeff * ((NonPolyBuilding*)build)->length * sin(upwind_rel) / ( 1 + 0.8 * ((NonPolyBuilding*)build)->length / eff_height));
			   build_width=((NonPolyBuilding*)build)->length;
			}
			else if(upwind_rel > pi - tol || upwind_rel < -pi + tol)
			{
			   num = 3;
			   perpendicular_flag=1;
			   ns_flag=0;
			   ((NonPolyBuilding*)build)->length = abs(lfcoeff * ((NonPolyBuilding*)build)->width * cos(upwind_rel) / ( 1 + 0.8 * ((NonPolyBuilding*)build)->width / eff_height));
			   build_width=((NonPolyBuilding*)build)->width;
			}

			if (num > -1)
			{
			   xf1=x[ (num % 4) + 1 ]*cos(upwind_dir)+y[ (num % 4) + 1 ]*sin(upwind_dir);
			   yf1=-x[ (num % 4) + 1 ]*sin(upwind_dir)+y[ (num % 4) + 1 ]*cos(upwind_dir);
			   xf2=x[ ( (num - 1) % 4) + 1]*cos(upwind_dir)+y[((num - 1) % 4) + 1]*sin(upwind_dir);
			   yf2=-x[ ((num - 1) % 4) + 1 ]*sin(upwind_dir)+y[((num - 1) % 4) + 1]*cos(upwind_dir);
			}

			ynorm = abs(yf1);
			if(perpendicular_flag == 1)
			{
			   if(upwindCavityFlag == 3)
			   {
				  vortex_height = MIN_S(build_width , eff_height);
				  retarding_height = eff_height;
			  }
			   else
			   {
				  vortex_height = eff_height;
				  retarding_height = eff_height;
			   }
			   // MAN 07/25/2008 stretched vertical grid
			   for (int k = 1; k <= build->kStart; k++)
			   {
				  kbottom = k;
				  if( build->baseHeight <= zm[k]) break;
			   }
			   for ( int k = build->kStart; k <= nz-1; k++)
				{
				  ktop = k;
				  if(height_factor * retarding_height + build->baseHeightActual <= z[k] ) break;
			   }
			   upIstart = MAX_S(build->iStart - (int)(1.5 * build->Lf / dx), 2);
			   upIstop = MIN_S(build->iEnd + (int)(1.5 * build->Lf / dx), nx - 1);
			   upJstart = MAX_S(build->jStart - (int)(1.5 * build->Lf / dy), 2);
			   upJstop = MIN_S(build->jEnd + (int)(1.5 * build->Lf / dy), ny-1);
				for ( int k = kbottom; k <= ktop; k++)
				{
				  zf = zm[k]- build->baseHeight;
					for (int j = upJstart; j <= upJstop; j++)
					{
					   for ( int i = upIstart; i <= upIstop; i++)
					   {
						x_u = (((float)(i) - 1) * dx - xco) * cos(upwind_dir) + 
									 (((float)(j) - 0.5 ) * dy - yco) * sin(upwind_dir);
						y_u = -(((float)(i)-1)*dx-xco)*sin(upwind_dir)+ 
									 (((float)(j)-0.5)*dy-yco)*cos(upwind_dir);
						x_v = (((float)(i)-0.5)*dx-xco)*cos(upwind_dir)+ 
									 (((float)(j)-1)*dy-yco)*sin(upwind_dir);
						y_v = -(((float)(i)-0.5)*dx-xco)*sin(upwind_dir)+	
									 (((float)(j)-1)*dy-yco)*cos(upwind_dir);
						x_w = (((float)(i)-0.5)*dx-xco)*cos(upwind_dir)+ 
									 (((float)(j)-0.5)*dy-yco)*sin(upwind_dir);
						y_w= -(((float)(i)-0.5)*dx-xco)*sin(upwind_dir)+	
									 (((float)(j)-0.5)*dy-yco)*cos(upwind_dir);
// u values
						if(y_u >= -ynorm && y_u <= ynorm)
						{
						   xs_u =((xf2-xf1)/(yf2-yf1))*(y_u-yf1)+xf1;
						   
						   if(zf > height_factor * vortex_height )
						   {
							  rz_end = 0.0f;
							  xv_u = 0.0f;
							  xrz_u = 0.0f;
						   }
						   else
						   {
							  xv_u = -build->Lf * sqrt( (1 - (pow(y_u/ynorm,2))) * (1 - pow((zf/(height_factor*vortex_height)),2)));
							  xrz_u = -build->Lf * sqrt( (1 - (pow(y_u/ynorm,2))) * (1 - pow((zf/(height_factor*retarding_height)),2)));
							  rz_end = length_factor * xv_u;
						   }
						   if(upwindCavityFlag == 1)
						   {
							  if(x_u-xs_u >= xv_u && x_u - xs_u <= 0.1 * dxy && iCellFlag[CELL(i,j,k,1)] != 0)
								 u0[CELL(i,j,k,0)] = 0.0f;
						   }
						   else
						   {
							  if(x_u - xs_u >= xrz_u && x_u - xs_u < rz_end &&
									iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 if(upwindCavityFlag == 3)
									u0[CELL(i,j,k,0)] *= ((x_u - xs_u - xrz_u) * (retarding_factor - 1.0f) / (rz_end - xrz_u) + 1.0f);
								 else
									u0[CELL(i,j,k,0)] *= retarding_factor;
								 if( abs(u0[CELL(i,j,k,0)]) > max_velmag)
									printf("Parameterized U exceeds MAX_S in upwind: %lf : %lf i:%d j:%d k:%d\n",u0[CELL(i,j,k,0)],max_velmag,i,j,k);
							  }
							  if(x_u - xs_u >= length_factor * xv_u && x_u - xs_u <= 0.1f * dxy &&
									iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 urot = u0[CELL(i,j,k,0)]*cos(build->rotation);
								 vrot = -u0[CELL(i,j,k,0)]*sin(build->rotation);
								 if(ns_flag == 1)
								 {
									vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05) * (-height_factor*sin(((pi*abs(x_u-xs_u))/(length_factor*build->Lf))+0));
								 }
								 else
								 {
									urot=-uhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05) * (-height_factor*sin(((pi*abs(x_u-xs_u))/(length_factor*build->Lf))+0));
								 }
								 u0[CELL(i,j,k,0)]=urot*cos(-build->rotation)+vrot*sin(-build->rotation);
								 if(abs(u0[CELL(i,j,k,0)]) > max_velmag)
									printf("Parameterized U exceeds MAX_S in upwind: %lf : %lf i:%d j:%d k:%d\n",u0[CELL(i,j,k,0)],max_velmag,i,j,k);
							  }
						   }
						}
//v values
						if(y_v >= -ynorm && y_v <= ynorm)
						{
						   xs_v =((xf2-xf1)/(yf2-yf1))*(y_v-yf1)+xf1;
						   
						   if(zf >= height_factor * vortex_height )
						   {
							  rz_end = 0.0f;
							  xv_v = 0.0f;
							  xrz_v = 0.0f;
						   }
						   else
						   {
							  xv_v = -build->Lf * sqrt( (1 - (pow(y_v/ynorm,2))) * (1 - pow((zf/(height_factor*vortex_height)),2)));
							  xrz_v = -build->Lf * sqrt( (1 - (pow(y_v/ynorm,2))) * (1 - pow((zf/(height_factor*retarding_height)),2)));
							  rz_end = length_factor * xv_v;
						   }
						   if(upwindCavityFlag == 1)
						   {
							  if(x_v-xs_v >= xv_v && x_v - xs_v <= 0.1 * dxy && iCellFlag[CELL(i,j,k,1)] != 0)
								 v0[CELL(i,j,k,0)] = 0.0f;
						   }
						   else
						   {
							  if(x_v - xs_v >= xrz_v && x_v - xs_v < rz_end &&
									iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 if(upwindCavityFlag == 3)
									v0[CELL(i,j,k,0)] *= ((x_v - xs_v - xrz_v) * (retarding_factor - 1.0f) / (rz_end - xrz_v) + 1.0f);
								 else
									v0[CELL(i,j,k,0)] *= retarding_factor;
								 if( abs(v0[CELL(i,j,k,0)]) > max_velmag)
									printf("Parameterized V exceeds MAX_S in upwind: %lf : %lf i:%d j:%d k:%d\n",v0[CELL(i,j,k,0)],max_velmag,i,j,k);
							  }
							  if(x_v - xs_v >= length_factor * xv_v && x_v - xs_v <= 0.1f * dxy &&
									iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 urot = v0[CELL(i,j,k,0)]*sin(build->rotation);
								 vrot = -v0[CELL(i,j,k,0)]*cos(build->rotation);
								 if(ns_flag == 1)
								 {
									vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05) * (-height_factor*sin(((pi*abs(x_v-xs_v))/(length_factor*build->Lf))+0));
								 }
								 else
								 {
									vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05) * (-height_factor*sin(((pi*abs(x_v-xs_v))/(length_factor*build->Lf))+0));
								 }
								 v0[CELL(i,j,k,0)]=-urot*sin(-build->rotation)+vrot*cos(-build->rotation);
								 if(abs(v0[CELL(i,j,k,0)]) > max_velmag)
									printf("Parameterized V exceeds MAX_S in upwind: %lf : %lf i:%d j:%d k:%d\n",v0[CELL(i,j,k,0)],max_velmag,i,j,k);
							  }
						   }
						}
//w values
						if(y_w >= -ynorm && y_w <= ynorm)
						{
						   xs_w =((xf2-xf1)/(yf2-yf1))*(y_w-yf1)+xf1;
						   
						   if(zf >= height_factor * vortex_height )
						   {
							  xv_w = 0.0f;
						   }
						   else
						   {
							  xv_w = -build->Lf * sqrt( (1 - (pow(y_w/ynorm,2))) * (1 - pow((zf/(height_factor*vortex_height)),2)));
						   }
						   if(upwindCavityFlag == 1)
						   {
							  if(x_w-xs_w >= xv_w && x_w-xs_w <= 0.1*dxy && iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 w0[CELL(i,j,k,0)] = 0.0f;
								 if(i < nx && j < ny && k < nz)
								 {
									iCellFlag[CELL(i,j,k,1)]=2;
								 }
							  }
						   }
						   else
						   {
							  if(x_w - xs_w >= xv_w && x_w - xs_w  < length_factor * xv_w &&
									iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 w0[CELL(i,j,k,0)] *= retarding_factor;
								 if(abs(w0[CELL(i,j,k,0)]) > max_velmag)
									printf("Parameterized W exceeds MAX_S in upwind: %lf : %lf i:%d j:%d k:%d\n",w0[CELL(i,j,k,0)],max_velmag,i,j,k);
								 if(i < nx - 1 && j < ny - 1 && k < nz - 1)
									iCellFlag[CELL(i,j,k,1)] = 2;
							  }
							  if(x_w - xs_w >= length_factor * xv_w && x_w - xs_w <= 0.0f && iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 //w0[CELL(i,j,k,0)] = -vel_mag*(0.1*cos(((pi*abs(x_w-xs_w))/(length_factor*build->Lf)))-0.05);
								 if(abs(w0[CELL(i,j,k,0)]) > max_velmag)
								 {
									printf("Parameterized W exceeds MAX_S in upwind: %lf : %lf i:%d j:%d k:%d\n",w0[CELL(i,j,k,0)],max_velmag,i,j,k);
								 }
								 if(i < nx - 1 && j < ny - 1 && k < nz - 1)
								 {
									iCellFlag[CELL(i,j,k,1)] = 2;
								 }
							  }
						   }
						}
					 } 
				  } 
			   }
		   }
			else
			   build->Lf = -999.0f;

}


void Solver::reliefWake(NonPolyBuilding* build, float* u0, float* v0)
{
	 int perpendicular_flag/*, uwakeflag, vwakeflag, wwakeflag*/;
	 float uo_h, vo_h, upwind_dir, upwind_rel, xco, yco;
	 float x1, y1, x2, y2, x3, y3, x4, y4;
	 float xw1, yw1, xw2, yw2, xw3, yw3, xf2, yf2, tol, zb, ynorm;
	 float farwake_exponent, farwake_factor, farwake_velocity;
	 float upwind_rel_norm, eff_height;
	 //float cav_fac, wake_fac, beta, LoverH, WoverH, eff_height;
	 //float canyon_factor, xc, yc, dNu, dNv, xwall, xu, yu, xv, yv, xp, yp, xwallu, xwallv, xwallw;
	 //int x_idx, y_idx, x_idx_min, iu, ju, iv, jv, kk, iw, jw;
	 //float vd, hd, Bs, BL, shell_height, xw, yw, dNw;
	 //int roof_perpendicular_flag, ns_flag;
	 //int ktop, kbottom, nupwind;
	 //float LrRect[3], LrLocal, LrLocalu, LrLocalv, LrLocalw;
	 float epsilon;
	 
	 epsilon = 10e-10;
	 
	 if(build->buildingGeometry == 4 && build->buildingRoof > 0)  //no current data param for buildingRoof
		eff_height = 0.8 * (build->height - build->baseHeightActual) + build->baseHeightActual;
	 else
		eff_height = build->height;

	 xco = build->xFo + build->Lt * cos(build->rotation); //!CENTER of building in QUIC domain coordinates
	 yco = build->yFo + build->Lt * sin(build->rotation);

	 //! find upwind direction and determine the type of flow regime
	 uo_h = u0[CELL( (int)(xco/dx), (int)(yco/dy), build->kEnd + 1, 0)];
	 vo_h = v0[CELL( (int)(xco/dx), (int)(yco/dy), build->kEnd + 1, 0)];
	 upwind_dir = atan2(vo_h,uo_h);
	 upwind_rel = upwind_dir - build->rotation;

	 if(upwind_rel > pi) upwind_rel = upwind_rel - 2 * pi;

	 if(upwind_rel <= -pi) upwind_rel = upwind_rel + 2 * pi;

	 upwind_rel_norm = upwind_rel + 0.5 * pi;

	 if(upwind_rel_norm > pi) upwind_rel_norm = upwind_rel_norm - 2 * pi;
	 tol = 0.01f * pi / 180.0f;

	 //!Location of corners relative to the center of the building

	 x1 = build->xFo + build->Wt * sin(build->rotation) - xco;
	 y1 = build->yFo - build->Wt * cos(build->rotation) - yco;
	 x2 = x1 + build->length * cos(build->rotation);
	 y2 = y1 + build->length * sin(build->rotation);
	 x4 = build->xFo - build->Wt * sin(build->rotation) - xco;
	 y4 = build->yFo + build->Wt * cos(build->rotation) - yco;
	 x3 = x4 + build->length * cos(build->rotation);
	 y3 = y4 + build->length * sin(build->rotation);
	 if(upwind_rel > 0.5f * pi + tol && upwind_rel < pi - tol)
	 {
		xw1=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yw1=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		xw2=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yw2=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		xf2=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yf2=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		xw3=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yw3=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		perpendicular_flag=0;
	}
	 else if (upwind_rel >= 0.5f * pi - tol && upwind_rel <= 0.5f * pi + tol)
	 {
		xw1=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yw1=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		xw3=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yw3=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		xf2=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		perpendicular_flag=1;
	}
	 else if(upwind_rel > tol && upwind_rel < 0.5f * pi - tol)
	 {
		xw1=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yw1=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		xw2=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yw2=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		xf2=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yf2=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		xw3=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yw3=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		perpendicular_flag=0;
	}
	 else if( abs(upwind_rel) <= tol)
	 {
		xw1=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yw1=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		xw3=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yw3=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		xf2=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		perpendicular_flag=1;
	}
	 else if(upwind_rel < -tol && upwind_rel > -0.5f * pi + tol)
	 {
		xw1=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yw1=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		xw2=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yw2=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		xf2=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yf2=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		xw3=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yw3=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		perpendicular_flag=0;
	}
	 else if(upwind_rel < -0.5f * pi + tol && upwind_rel > -0.5f * pi - tol)
	 {
		xw1=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yw1=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		xw3=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yw3=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		xf2=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		perpendicular_flag=1;
	}
	 else if(upwind_rel < -0.5f * pi - tol && upwind_rel > -pi + tol)
	 {
		xw1=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yw1=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		xw2=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yw2=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		xf2=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yf2=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		xw3=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yw3=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		perpendicular_flag=0;
	}
	 else
	 {
		xw1=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yw1=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		xw3=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yw3=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		xf2=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		perpendicular_flag=1;
	 }
	 build->Leff = build->width * build->length / abs(yw3-yw1);
	 if(perpendicular_flag == 1)
		build->Weff = build->width * build->length / abs(xf2-xw1);
	 else
		build->Weff = build->width * build->length / abs(xf2-xw2);
	 return;
}
