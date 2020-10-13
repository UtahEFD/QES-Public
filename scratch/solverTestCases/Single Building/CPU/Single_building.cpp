/**
 * \author Behnam Bozorgmehr
 * \version 1.0
 * \date 06/04/2018
 * \copyright GNU Public License
 * \mainpage CUDA-URB code 
 * \section intro_sec Introduction
 * This code is developed to compute wind field around a single building case without parameterization and with wind normal to the building.
 * \section compile_sec Compilation
 * This section will be developed later
*/



#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>


using namespace std;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;


int main(int argc, const char * argv[]) {
    auto start = std::chrono::high_resolution_clock::now(); // Start recording execution time    

    int nx = 40;    /// Number of cells in x-dir 
    int ny = 40;    /// Number of cells in y-dir
    int nz = 40;    /// Number of cells in z-dir
    nx += 1;        /// +1 for Staggered grid
    ny += 1;        /// +1 for Staggered grid
    nz += 2;        /// +2 for staggered grid and ghost cell
	long numcell_cent = (nx-1)*(ny-1)*(nz-1);         /// Total number of cell-centered values in domain
    long numface_cent = nx*ny*nz;                     /// Total number of face-centered values in domain
    
    /// Grid resolution
    float dx = 5.0;
    float dy = 5.0;
    float dz = 5.0;
    
    int alpha1 = 1;        /// Gaussian precision moduli
    int alpha2 = 1;        /// Gaussian precision moduli
    float eta = pow(alpha1/alpha2, 2.0);
    float A = pow(dx/dy, 2.0);
    float B = eta*pow(dx/dz, 2.0);
    float tol = 1e-9;     /// Error tolerance
    float omega = 1.78;   /// Over-relaxation factor
    int itermax = 500;    /// Maximum number of iterations
    
    /// Declare face-centered positions
    float *x, *y, *z;
    x = new float [nx];
    y = new float [ny];
    z = new float [nz];  

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
    
    for ( int i = 0; i < nx-1; i++){
        x[i] = (i+0.5)*dx;         /// Location of face centers in x-dir
    }
    for ( int j = 0; j < ny-1; j++){
        y[j] = (j+0.5)*dy;         /// Location of face centers in y-dir
    }
    for ( int k = 0; k < nz-1; k++){
        z[k] = (k-0.5)*dz;         /// Location of face centers in z-dir
    }
    
    
    float z0 = 0.1;                 /// Surface roughness
    float z_ref = 10.0;             /// Height of the measuring sensor (m)
    float U_ref = 5.0;              /// Measured velocity at the sensor height (m/s)
    float H = 20.0;                 /// Building height
    float W = 20.0;                 /// Building width
    float L = 20.0;                 /// Building length
    float x_start = 90.0;           /// Building start location in x-direction
    float y_start = 90.0;           /// Building start location in y-direction
    float i_start = round(x_start/dx);     /// Index of building start location in x-direction
    float i_end = round((x_start+L)/dx);   /// Index of building end location in x-direction
    float j_start = round(y_start/dy);     /// Index of building start location in y-direction
    float j_end = round((y_start+W)/dy);   /// Index of building end location in y-direction 
    float k_end = round(H/dz);             /// Index of building end location in z-direction
    int *icellflag;
    icellflag = new int [numcell_cent];       /// Cell index flag (0 = building, 1 = fluid)

    std::cout << "i_start:" << i_start << "\n";   // Print the number of iterations
    std::cout << "i_end:" << i_end << "\n";       // Print the number of iterations
    std::cout << "j_start:" << j_start << "\n";   // Print the number of iterations
    std::cout << "j_end:" << j_end << "\n";       // Print the number of iterations    
    std::cout << "k_end:" << k_end << "\n";       // Print the number of iterations 

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

    for ( int k = 1; k < nz; k++){
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){
				
				int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values                                
                /// Define logarithmic wind profile
                u0[icell_face] = U_ref*(log((z[k]+z0)/z0)/log((z_ref+z0)/z0));
				v0[icell_face] = w0 [icell_face] = 0.0;         /// Perpendicular wind direction

            }
        }
    }

    for (int k = 0; k < k_end+1; k++){
        for (int j = j_start; j < j_end; j++){
            for (int i = i_start; i < i_end; i++){

                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				icellflag[icell_cent] = 0;                         /// Set cell index flag to building

			}
		}
    }

    for (int j = 0; j < ny-1; j++){
        for (int i = 0; i < nx-1; i++){
            int icell_cent = i + j*(nx-1);   /// Lineralized index for cell centered values
            icellflag[icell_cent] = 0.0;
        }
    }

   
    for (int k = 0; k < nz; k++){
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
                int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values 
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
                
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values 
                /// Calculate divergence of initial velocity field
                R[icell_cent] = (-2*pow(alpha1, 2.0))*(((u0[icell_face+1]-u0[icell_face])/dx)+((v0[icell_face + nx]-v0[icell_face])/dy)+((w0[icell_face + nx*ny]-w0[icell_face])/dz));
            }
        }
    }
    
	std::cout << "i_start:" << i_start << "\n";   // Print the number of iterations

    /// Boundary condition for building edges
    for (int k = 1; k < nz-2; k++){
        for (int j = 1; j < ny-2; j++){
            for (int i = 1; i < nx-2; i++){
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				if (icellflag[icell_cent] != 0) {
					
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
    for (int k = 1; k < nz-1; k++){
        for (int j = 0; j < ny-1; j++){
            for (int i = 0; i < nx-1; i++){
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				e[icell_cent] = e[icell_cent]/(dx*dx);
				f[icell_cent] = f[icell_cent]/(dx*dx);
				g[icell_cent] = g[icell_cent]/(dy*dy);
				h[icell_cent] = h[icell_cent]/(dy*dy);
				m[icell_cent] = m[icell_cent]/(dz*dz);
				n[icell_cent] = n[icell_cent]/(dz*dz);
			}
		}
	}
   
    
    /////////////////////////////////////////////////
    //                 SOR solver              //////
    /////////////////////////////////////////////////
    int iter = 0;
    double error = 1.0;
	double reduced_error = 0.0;
    
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
                    
                    int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
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
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				int icell_face = i + j*nx + k*nx*ny;               /// Lineralized index for cell faced values 
                u[icell_face] = u0[icell_face]+(1/(2*pow(alpha1, 2.0)*dx))*(lambda[icell_cent]-lambda[icell_cent-1]);
                v[icell_face] = v0[icell_face]+(1/(2*pow(alpha1, 2.0)*dy))*(lambda[icell_cent]-lambda[icell_cent - (nx-1)]);
                w[icell_face] = w0[icell_face]+(1/(2*pow(alpha2, 2.0)*dz))*(lambda[icell_cent]-lambda[icell_cent - (nx-1)*(ny-1)]);
                
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
    for (int k = 0; k < nz; k++){
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values
                outdata1 << "\t" << i << "\t" << j << "\t" << k << "\t \t"<< x[i] << "\t \t" << y[j] << "\t \t" << z[k] << "\t \t"<< "\t \t" << u[icell_face] <<"\t \t"<< "\t \t"<<v[icell_face]<<"\t \t"<< "\t \t"<<w[icell_face]<< endl;   
}
        }
    }
    outdata1.close();

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
   
    return 0;
}



