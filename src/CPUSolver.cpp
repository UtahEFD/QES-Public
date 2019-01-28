#include "CPUSolver.h"

// If you want to use these types of statements, they should go in the
// CPP file.
using std::cerr;
using std::endl;
using std::vector;
using std::cout;

void CPUSolver::solve(bool solveWind)
{
    auto startOfSolveMethod = std::chrono::high_resolution_clock::now(); // Start recording execution time             
    
    /////////////////////////////////////////////////////////////////
    ////////      Divergence of the initial velocity field   ////////
    /////////////////////////////////////////////////////////////////

    for (int k = 1; k < nz-1; k++)
    {
        for (int j = 0; j < ny-1; j++)
        {
            for (int i = 0; i < nx-1; i++)
            {
                icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                icell_face = i + j*nx + k*nx*ny;

                /// Calculate divergence of initial velocity field
                R[icell_cent] = (-2*pow(alpha1, 2.0))*((( e[icell_cent] * u0[icell_face+1]       - f[icell_cent] * u0[icell_face]) * dx ) +
                                                       (( g[icell_cent] * v0[icell_face + nx]    - h[icell_cent] * v0[icell_face]) * dy ) +
                                                       (( m[icell_cent] * w0[icell_face + nx*ny] - n[icell_cent] * w0[icell_face]) * dz ));
            }
        }
    }

    if (solveWind)
    {
        auto startSolveSection = std::chrono::high_resolution_clock::now();
        
        //INSERT CANOPY CODE

        /////////////////////////////////////////////////
        //                 SOR solver              //////
        /////////////////////////////////////////////////
        int iter = 0;
        double error = 1.0;
    	double reduced_error = 0.0;
        
        std::cout << "Solving...\n";
        while (iter < itermax && error > tol && error > reduced_error) {
            
            // Save previous iteration values for error calculation
            //    uses stl vector's assignment copy function, assign
            lambda_old.assign( lambda.begin(), lambda.end() );   
            
            //
            // main SOR formulation loop
            // 
            for (int k = 1; k < nz-2; k++){
            	for (int j = 1; j < ny-2; j++){
            	    for (int i = 1; i < nx-2; i++){
                    
            	        icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values

                        lambda[icell_cent] = (omega / ( e[icell_cent] + f[icell_cent] + g[icell_cent] +
                                                        h[icell_cent] + m[icell_cent] + n[icell_cent])) *
                            ( e[icell_cent] * lambda[icell_cent+1]        + f[icell_cent] * lambda[icell_cent-1] +
                              g[icell_cent] * lambda[icell_cent + (nx-1)] + h[icell_cent] * lambda[icell_cent-(nx-1)] +
                              m[icell_cent] * lambda[icell_cent+(nx-1)*(ny-1)] +
                              n[icell_cent] * lambda[icell_cent-(nx-1)*(ny-1)] - R[icell_cent] ) +
                            (1.0 - omega) * lambda[icell_cent];    /// SOR formulation
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
            
            /// Error calculation
            error = 0.0;                   /// Reset error value before error calculation 
            for (int k = 0; k < nz-1; k++){
                for (int j = 0; j < ny-1; j++){
                    for (int i = 0; i < nx-1; i++){
                        int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
                        error += fabs(lambda[icell_cent] - lambda_old[icell_cent])/((nx-1)*(ny-1)*(nz-1));
                    }
                }
            }

            if (iter == 0) {
                reduced_error = error * 1.0e-3;
            }
            
            iter += 1;
        }
        std::cout << "Solved!\n";
        
        std::cout << "Number of iterations:" << iter << "\n";   // Print the number of iterations
        std::cout << "Error:" << error << "\n";
        std::cout << "Reduced Error:" << reduced_error << "\n";   

        ////////////////////////////////////////////////////////////////////////
        /////   Update the velocity field using Euler-Lagrange equations   /////
        ////////////////////////////////////////////////////////////////////////  
        
        // Ideally only do for boundary planes.... on external faces
        // of domain - Pete
        for (int k = 0; k < nz; k++){
            for (int j = 0; j < ny; j++){
                for (int i = 0; i < nx; i++) {
                    int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values 
                    u[icell_face] = u0[icell_face];
                    v[icell_face] = v0[icell_face];
                    w[icell_face] = w0[icell_face];
                }
            }
        }
//  For only the boundary space....
        // Ideally, for k={0,nz-1}
//        u[ ? ] = u0[ ? ]
//            v[ ? ] = v0[ ? ]
//            w[ ? ] = w0[ ? ]
            
        // /////////////////////////////////////////////
    	/// Update velocity field using Euler equations
        // /////////////////////////////////////////////
        for (int k = 1; k < nz-1; k++){
            for (int j = 1; j < ny-1; j++){
                for (int i = 1; i < nx-1; i++){
                    icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
                    icell_face = i + j*nx + k*nx*ny;               /// Lineralized index for cell faced values 

                    // Calculate correct wind velocity
                    u[icell_face] = u0[icell_face] + (1/(2*pow(alpha1, 2.0)*dx)) *
                        f[icell_cent]*dx*dx*(lambda[icell_cent]-lambda[icell_cent-1]);
                    
                    v[icell_face] = v0[icell_face] + (1/(2*pow(alpha1, 2.0)*dy)) *
                        h[icell_cent]*dy*dy*(lambda[icell_cent]-lambda[icell_cent - (nx-1)]);
                    
                    w[icell_face] = w0[icell_face]+(1/(2*pow(alpha2, 2.0)*dz)) *
                        n[icell_cent]*dz*dz*(lambda[icell_cent]-lambda[icell_cent - (nx-1)*(ny-1)]);



                }
            }
        }

                for (int k = 1; k < nz-1; k++){
            for (int j = 1; j < ny-1; j++){
                for (int i = 1; i < nx-1; i++){
                    icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
                    icell_face = i + j*nx + k*nx*ny;               /// Lineralized
                                                                   /// index
                                                                   /// for
                                                                   /// cell
                                                                   /// faced
                                                                   /// values

                                        // If we are inside a building, set velocities to 0.0
                    if (icellflag[icell_cent] == 0) {
                        /// Setting velocity field inside the building to zero
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
        std::chrono::duration<float> elapsedTotal = finish - startOfSolveMethod;
        std::chrono::duration<float> elapsedSolve = finish - startSolveSection;
        std::cout << "Elapsed total time: " << elapsedTotal.count() << " s\n";   // Print out elapsed execution time
        std::cout << "Elapsed solve time: " << elapsedSolve.count() << " s\n";   // Print out elapsed execution time
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////    Writing output data                            //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
void CPUSolver::outputDataFile()
{

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

	for (int k = 0; k < nz-1; k++){
		for (int j = 0; j < ny-1; j++){
			for (int i = 0; i < nx-1; i++){
				int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values 
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1); 
				u_out[icell_cent] = 0.5*(u[icell_face+1]+u[icell_face]);
				v_out[icell_cent] = 0.5*(v[icell_face+nx]+v[icell_face]);
				w_out[icell_cent] = 0.5*(w[icell_face+nx*ny]+w[icell_face]);
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
				outdata1 << "\t" << i << "\t" << j << "\t" << k << "\t \t"<< x[i] << "\t \t" << y[j] << "\t \t" << z[k] 
								<< "\t \t"<< "\t \t" << u[icell_face] <<"\t \t"<< "\t \t"<<v[icell_face]<<"\t \t"<< "\t \t"
								<<w[icell_face]<< "\t \t"<< "\t \t" << u0[icell_face] <<"\t \t"<< "\t \t"<<v0[icell_face]
								<<"\t \t"<< "\t \t"<<w0[icell_face]<<"\t \t"<<R[icell_cent]<< endl;   
			}
		}
	}
	outdata1.close();

	// Write data to file
	ofstream outdata2;
	outdata2.open("Final velocity1.dat");
	if( !outdata2 ) {                 // File couldn't be opened
		cerr << "Error: file could not be opened" << endl;
		exit(1);
	}
	// Write data to file
	for (int k = 0; k < nz-1; k++){
		for (int j = 0; j < ny-1; j++){
			for (int i = 0; i < nx-1; i++){
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values
				outdata2 << "\t" << i << "\t" << j << "\t" << k << "\t \t"<< x[i] << "\t \t" << y[j] << "\t \t" << z[k] 
								<< "\t \t"<< "\t \t" << f[icell_cent] <<"\t \t"<< "\t \t"<<e[icell_cent]<<"\t \t"<< "\t \t"
								<<h[icell_cent]<< "\t \t"<< "\t \t" << g[icell_cent] <<"\t \t"<< "\t \t"<<n[icell_cent]
								<<"\t \t"<< "\t \t"<<m[icell_cent]<<"\t \t"<<icellflag[icell_cent]<< endl;   
			}
		}
	}
	outdata2.close();      
}

void CPUSolver::outputNetCDF( NetCDFData* netcdfDat )
{
    /// Declare cell center positions
    std::vector<float> x_out(nx-1) , y_out(ny-1), z_out(nz-1);

    for ( int i = 0; i < nx-1; i++) {
    	x_out[i] = (i+0.5)*dx;         /// Location of cell centers in x-dir
   	}
    for ( int j = 0; j < ny-1; j++){
		y_out[j] = (j+0.5)*dy;         /// Location of cell centers in y-dir
	}
	for ( int k = 0; k < nz-1; k++){
		z_out[k] = (k-0.5)*dz;         /// Location of cell centers in z-dir
	}

    long numcell_cent = (nx-1)*(ny-1)*(nz-1);         /// Total number of cell-centered values in domain

    netcdfDat->getDataFace(x.data(),y.data(),z.data(),u.data(),v.data(),w.data(),nx,ny,nz);
    netcdfDat->getDataICell(icellflag.data(), x_out.data(), y_out.data(), z_out.data(), nx-1, ny - 1, nz - 1, numcell_cent);
    
      /* if (DTEHFExists)
            netcdfDat->getCutCellFlags(cells);

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

        //netcdfDat->getDataICell(icellflag.data(), x_out, y_out, z_out, nx-1, ny - 1, nz - 1, numcell_cent);
    }*/
}



