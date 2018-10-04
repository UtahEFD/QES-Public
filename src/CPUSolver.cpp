#include "CPUSolver.h"

void CPUSolver::solve(NetCDFData* netcdfDat, bool solveWind)
{




    
    auto start = std::chrono::high_resolution_clock::now(); // Start recording execution time    

	long numcell_cent = (nx-1)*(ny-1)*(nz-1);         /// Total number of cell-centered values in domain
    long numface_cent = nx*ny*nz;                     /// Total number of face-centered values in domain
    
    

    /// Declare coefficients for SOR solver
    float *e, *f, *g, *h, *m, *n;
    e = new float [numcell_cent];
    f = new float [numcell_cent];
    g = new float [numcell_cent];
    h = new float [numcell_cent];
    m = new float [numcell_cent];
    n = new float [numcell_cent];

    /// Declaration of initial wind components (u0,v0,w0)
    double *u0, *v0, *w0;
    u0 = new double [numface_cent];
    v0 = new double [numface_cent];
    w0 = new double [numface_cent];
    
    
    double * R;              //!> Divergence of initial velocity field
    R = new double [numcell_cent];
  
    /// Declaration of final velocity field components (u,v,w)
    double *u, *v, *w;
    u = new double [numface_cent];
    v = new double [numface_cent];
    w = new double [numface_cent];

    /// Declaration of Lagrange multipliers
    double *lambda, *lambda_old;
    lambda = new double [numcell_cent];
    lambda_old = new double [numcell_cent];
	int icell_face, icell_cent;
    
    for ( int i = 0; i < nx; i++){
        x.push_back(i*dx);         /// Location of face centers in x-dir
	}
    for ( int j = 0; j < ny; j++){
        y.push_back(j*dy);         /// Location of face centers in y-dir
    }

    for ( int k = 0; k < nz; k++){
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){

				icell_face = i + j*nx + k*nx*ny;
				u0[icell_face] = v0[icell_face] = w0[icell_face] = 0.0;
			}
		}    
	}	
	
	std::cout << "num_sites:" << Sensor1->num_sites << "\n";
	std::cout << "site_blayer_flag:" << Sensor1->site_blayer_flag[num_sites-1] << "\n";
	Sensor1->inputWindProfile(dx, dy, dz, nx, ny, nz, u0, v0, w0);
    /*
    Set Terrain buildings
    Deprecate
    */
    if (mesh)
    {
        std::cout << "Creating terrain blocks...\n";
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {           //get height, then add half a cell, if the height exceeds half of a cell partially, it will round up.
                float heightToMesh = mesh->getHeight(i * dx + dx * 0.5f, j * dy + dy * 0.5f) + 0.5f * dz;
                for (int k = 0; k < (int)(heightToMesh / dz); k++)
                    buildings.push_back(new RectangularBuilding(i * dx, j * dy, k * dz, dx, dy, dz));
            }
             printProgress( (float)i / (float)nx);
        }
        std::cout << "blocks created\n";
    }

    int *icellflag;
    icellflag = new int [numcell_cent];       /// Cell index flag (0 = building, 1 = fluid)

    for ( int k = 0; k < nz-1; k++){
        for (int j = 0; j < ny-1; j++){
            for (int i = 0; i < nx-1; i++){

				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);            /// Lineralized index for cell centered values
                e[icell_cent] = f[icell_cent] = g[icell_cent] = h[icell_cent] = m[icell_cent] = n[icell_cent] = 1.0;  /// Assign initial values to the coefficients for SOR solver
				icellflag[icell_cent] = 1;                                  /// Initialize all cells to fluid	
				lambda[icell_cent] = lambda_old[icell_cent] = 0.0;
			}
		}    
	}	


    float* zm;
    zm = new float[nz];
    int* iBuildFlag;
    iBuildFlag = new int[nx*ny*nz];
    for (int i = 0; i < buildings.size(); i++)
    {
        ((RectangularBuilding*)buildings[i])->setBoundaries(dx, dy, dz, nx, ny, nz, zm, e, f, g, h, m, n, icellflag);    /// located in RectangularBuilding.h
        //((RectangularBuilding*)buildings[i])->setCells(nx, ny, nz, icellflag, iBuildFlag, i);
    }



    for (int k = 0; k < nz; k++){
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){
				icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
                icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values 
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

    for (int k = 1; k < nz-1; k++){
        for (int j = 0; j < ny-1; j++){
            for (int i = 0; i < nx-1; i++){
                
                icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values 
                /// Calculate divergence of initial velocity field
                R[icell_cent] = (-2*pow(alpha1, 2.0))*(((e[icell_cent]*u0[icell_face+1]-f[icell_cent]*u0[icell_face])/dx)+((g[icell_cent]*v0[icell_face + nx]-h[icell_cent]*v0[icell_face])/dy)+((m[icell_cent]*w0[icell_face + nx*ny]-n[icell_cent]*w0[icell_face])/dz));
            }
        }
    }

    /// New boundary condition implementation
    for (int k = 0; k < nz-1; k++){
        for (int j = 0; j < ny-1; j++){
            for (int i = 0; i < nx-1; i++){
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
   

    if (solveWind)
    {
        
        
        /////////////////////////////////////////////////
        //                 SOR solver              //////
        /////////////////////////////////////////////////
        int iter = 0;
        double error = 1.0;
    	double reduced_error = 0.0;
        
        std::cout << "Solving...\n";
        while (iter < itermax && error > tol) {
            
    		/// Save previous iteration values for error calculation  
            for (int k = 0; k < nz-1; k++){
                for (int j = 0; j < ny-1; j++){
                    for (int i = 0; i < nx-1; i++){
                        int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
                        lambda_old[icell_cent] = lambda[icell_cent];
                    }
                }
            }
            
        	for (int k = 1; k < nz-2; k++){
            	for (int j = 1; j < ny-2; j++){
            	    for (int i = 1; i < nx-2; i++){
                    
            	        icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
						lambda[icell_cent] = (omega/(e[icell_cent]+f[icell_cent]+g[icell_cent]+h[icell_cent]+m[icell_cent]+n[icell_cent]))*(e[icell_cent]*lambda[icell_cent+1]+f[icell_cent]*lambda[icell_cent-1]+g[icell_cent]*lambda[icell_cent + (nx-1)]+h[icell_cent]*lambda[icell_cent - (nx-1)]+m[icell_cent]*lambda[icell_cent + (nx-1)*(ny-1)]+n[icell_cent]*lambda[icell_cent - (nx-1)*(ny-1)]-R[icell_cent])+(1-omega)*lambda[icell_cent];    /// SOR formulation
					}
				}
			}
            
            /// Mirror boundary condition (lambda (@k=0) = lambda (@k=1))
            for (int j = 0; j < ny-1; j++){
                for (int i = 0; i < nx-1; i++){
                    int icell_cent = i + j*(nx-1);         /// Lineralized index for cell centered values
                    lambda[icell_cent] = lambda[icell_cent + (nx-1)*(ny-1)];
                }
            }
            
            error = 0.0;                   /// Reset error value before error calculation 

    		/// Error calculation
            for (int k = 0; k < nz-1; k++){
                for (int j = 0; j < ny-1; j++){
                    for (int i = 0; i < nx-1; i++){
                        int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
                        error += fabs(lambda[icell_cent] - lambda_old[icell_cent])/((nx-1)*(ny-1)*(nz-1));
                    }
                }
            }


    		if (iter == 0){
    			reduced_error = error*1e-3;
    		}

            
            iter += 1;
        }
        std::cout << "Solved!\n";
        
        std::cout << "Number of iterations:" << iter << "\n";   // Print the number of iterations
        std::cout << "Error:" << error << "\n";
        std::cout << "Reduced Error:" << reduced_error << "\n";     
        
        for (int k = 0; k < nz; k++){
            for (int j = 0; j < ny; j++){
                for (int i = 0; i < nx; i++){
                    int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values 
                    u[icell_face] = u0[icell_face];
                    v[icell_face] = v0[icell_face];
                    w[icell_face] = w0[icell_face];
                }
            }
        }
        
    	/// Update velocity field using Euler equations
		for (int k = 1; k < nz-1; k++){
	        for (int j = 1; j < ny-1; j++){
	            for (int i = 1; i < nx-1; i++){
	                icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
					icell_face = i + j*nx + k*nx*ny;               /// Lineralized index for cell faced values 
	                u[icell_face] = u0[icell_face]+(1/(2*pow(alpha1, 2.0)*dx))*f[icell_cent]*dx*dx*(lambda[icell_cent]-lambda[icell_cent-1]);
	                v[icell_face] = v0[icell_face]+(1/(2*pow(alpha1, 2.0)*dy))*h[icell_cent]*dy*dy*(lambda[icell_cent]-lambda[icell_cent - (nx-1)]);
	                w[icell_face] = w0[icell_face]+(1/(2*pow(alpha2, 2.0)*dz))*n[icell_cent]*dz*dz*(lambda[icell_cent]-lambda[icell_cent - (nx-1)*(ny-1)]);
	                
	            }
	        }
	    }

    	/// Setting velocity field inside the building to zero
        for (int k = 1; k < nz-1; k++){
            for (int j = 1; j < ny-1; j++){
                for (int i = 1; i < nx-1; i++){
                    int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
    				int icell_face = i + j*nx + k*nx*ny;               /// Lineralized index for cell faced values 
    				if (icellflag[icell_cent]==0) {
						std::cout << "i:" << i << "\n"<< "j:" << j << "\n"<< "k:" << k << "\n";
    					u[icell_face] = 0;
    					u[icell_face+1] = 0;
    					v[icell_face] = 0;
    					v[icell_face+nx] = 0;
    					w[icell_face] = 0;
    					w[icell_face+nx*ny] = 0;
    				}
                    
                }
            }
        }


        auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time
        std::chrono::duration<float> elapsed = finish - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s\n";   // Print out elapsed execution time

        ///////////////////////////////////////////////////////////////////////////////////////////////
        //////    Writing output data                            //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////

    	/// Declare cell center positions
        float *x_out, *y_out, *z_out;
        x_out = new float [nx-1];
        y_out = new float [ny-1];
        z_out = new float [nz-1];


    	for ( int i = 0; i < nx-1; i++) {
    		x_out[i] = (i+0.5)*dx;         /// Location of cell centers in x-dir
    	}
        for ( int j = 0; j < ny-1; j++){
            y_out[j] = (j+0.5)*dy;         /// Location of cell centers in y-dir
        }
        for ( int k = 0; k < nz-1; k++){
            z_out[k] = (k-0.5)*dz;         /// Location of cell centers in z-dir
        }

    	/// Declare output velocity field arrays
        double ***u_out, ***v_out, ***w_out;
        u_out = new double** [nx-1];
        v_out = new double** [nx-1];
        w_out = new double** [nx-1];
    	
    	for (int i = 0; i < nx-1; i++){
    		u_out[i] = new double* [ny-1];
    		v_out[i] = new double* [ny-1];
    		w_out[i] = new double* [ny-1];
    		for (int j = 0; j < ny-1; j++){
    			u_out[i][j] = new double [nz-1];
    			v_out[i][j] = new double [nz-1];
    			w_out[i][j] = new double [nz-1];
    		}
    	}


    	for (int k = 0; k < nz-1; k++){
            for (int j = 0; j < ny-1; j++){
                for (int i = 0; i < nx-1; i++){
                    int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values 
    				u_out[i][j][k] = 0.5*(u[icell_face+1]+u[icell_face]);
    				v_out[i][j][k] = 0.5*(v[icell_face+nx]+v[icell_face]);
    				w_out[i][j][k] = 0.5*(w[icell_face+nx*ny]+w[icell_face]);
    			}
    		}	
    	}

        // Write data to file
        ofstream outdata1;
        outdata1.open("Final velocity.dat");
        if( !outdata1 ) {                 // File couldn't be opened
            cerr << "Error: file could not be opened" << endl;
            exit(1);
        }
        // Write data to file
        for (int k = 0; k < nz-1; k++){
            for (int j = 0; j < ny-1; j++){
                for (int i = 0; i < nx-1; i++){
    				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
    				int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values
                    outdata1 << "\t" << i << "\t" << j << "\t" << k << "\t \t"<< x[i] << "\t \t" << y[j] << "\t \t" << z[k] << "\t \t"<< "\t \t" << u[icell_face] <<"\t \t"<< "\t \t"<<v[icell_face]<<"\t \t"<< "\t \t"<<w[icell_face]<< "\t \t"<< "\t \t" << u0[icell_face] <<"\t \t"<< "\t \t"<<v0[icell_face]<<"\t \t"<< "\t \t"<<w0[icell_face]<<"\t \t"<<icellflag[icell_cent]<< endl;   
                }
            }
        }


        outdata1.close();


        netcdfDat->getData(x.data(),y.data(),z.data(),u,v,w,nx,ny,nz);
        netcdfDat->getDataICell(icellflag, x_out, y_out, z_out, nx-1, ny - 1, nz - 1, numcell_cent);
        if (DTEHF)
            netcdfDat->getCutCellFlags(cells);



        // Write data to file
        ofstream outdata;
        outdata.open("Final velocity, cell-centered.dat");
        if( !outdata ) {                 // File couldn't be opened
            cerr << "Error: file could not be opened" << endl;
            exit(1);
        }
        // Write data to file
        for (int k = 0; k < nz-1; k++){
            for (int j = 0; j < ny-1; j++){
                for (int i = 0; i < nx-1; i++){
                    outdata << "\t" << i << "\t" << j << "\t" << k << "\t"<< x_out[i] << "\t" << y_out[j] << "\t" << z_out[k] << "\t" << u_out[i][j][k] << "\t" << v_out[i][j][k] << "\t" << w_out[i][j][k] 					<< endl;   
    			}
    		}
    	}
        outdata.close();
    }
    {
        float *x_out, *y_out, *z_out;
        x_out = new float [nx-1];
        y_out = new float [ny-1];
        z_out = new float [nz-1];


        for ( int i = 0; i < nx-1; i++) {
            x_out[i] = (i+0.5)*dx;         /// Location of cell centers in x-dir
        }
        for ( int j = 0; j < ny-1; j++){
            y_out[j] = (j+0.5)*dy;         /// Location of cell centers in y-dir
        }
        for ( int k = 0; k < nz-1; k++){
            z_out[k] = (k-0.5)*dz;         /// Location of cell centers in z-dir
        }

        netcdfDat->getDataICell(icellflag, x_out, y_out, z_out, nx-1, ny - 1, nz - 1, numcell_cent);
    }
}
