#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>


//using namespace std::chrono;
using namespace std;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;
//using std::to_string;


int main(int argc, const char * argv[]) {
    
    auto start = std::chrono::high_resolution_clock::now(); // Start recording execution time
    int nx = 20;     // Number of cells in x-dir
    int ny = 20;     // Number of cells in y-dir
    int nz = 20;      // Number of cells in z-dir
    long d_size = nx*ny*nz;       // Total number of nodes in domain
    
    // Grid resolution
    float dx = 5.0;
    float dy = 5.0;
    float dz = 5.0;
    
    int alpha1 = 1;        // Gaussian precision moduli
    int alpha2 = 1;        // Gaussian precision moduli
    float eta = pow(alpha1/alpha2, 2.0);
    float A = pow(dx/dy, 2.0);
    float B = eta*pow(dx/dz, 2.0);
    double tol = 1e-9;     // Error tolerance
    float omega = 1.78;   // Over-relaxation factor
    int itermax = 1000;    // Maximum number of iterations
    
    float *x, *y, *z;
    x = new float [nx];
    y = new float [ny];
    z = new float [nz];  

    // Declare coefficients for SOR solver
    float *e, *f, *g, *h, *m, *n, *o, *p, *q;
    e = new float [d_size];
    f = new float [d_size];
    g = new float [d_size];
    h = new float [d_size];
    m = new float [d_size];
    n = new float [d_size];
    o = new float [d_size];
    p = new float [d_size];
    q = new float [d_size];

    // Declare initial wind profile (u0,v0,w0)
    double *u0, *v0, *w0;
    u0 = new double [d_size];
    v0 = new double [d_size];
    w0 = new double [d_size];
    
    // Declare divergence of initial velocity field
    double * R;
    R = new double [d_size];
  
    // Declare final velocity field
    double *u, *v, *w;
    u = new double [d_size];
    v = new double [d_size];
    w = new double [d_size];

    // Declare Lagrange multipliers
    double *lambda, *lambda_old;
    lambda = new double [d_size];
    lambda_old = new double [d_size];
    
    for ( int i = 0; i < nx; i++){
        x[i] = (i*dx) + (dx/2);     // Location of middle cell nodes in x-dir
    }
    for ( int j = 0; j < ny; j++){
        y[j] = (j*dy) + (dy/2);         // Location of middle cell nodes in y-dir
    }
    for ( int k = 0; k < nz; k++){
        z[k] = (k*dz) + (dz/2);         // Location of middle cell nodes in z-dir
    }
    
    
    float z0[] = {0.01,0.05,0.1,1.0};       // Surface roughness (m)
    float z_ref = 10.0;                  // Height of the measuring sensor (m)
    float U_ref = 5.0;                   // Measured velocity at the sensor height (m/s)
    
    
    for ( int k = 0; k < nz; k++){
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){
                
                int ii = i + j*nx + k*nx*ny;   // Lineralize the vectors (make it 1D)
                e[ii] = f[ii] = g[ii] = h[ii] = m[ii] = n[ii] = o[ii] = p[ii] = q[ii] = 1.0;  // Assign initial values to the coefficients for SOR solver
                
                // Define logarithmic wind profile for four subdomains
                if (i < (nx-1)/2 && j < (ny-1)/2) {
                    int kk = 0;
                    u0[ii] = U_ref*(log((z[k]+z0[kk])/z0[kk])/log((z_ref+z0[kk])/z0[kk]));
                } else if (i < (nx-1)/2 && j >= (ny-1)/2) {
                    int kk = 1;
                    u0[ii] = U_ref*(log((z[k]+z0[kk])/z0[kk])/log((z_ref+z0[kk])/z0[kk]));
                } else if (i >= (nx-1)/2 && j < (ny-1)/2) {
                    int kk = 2;
                    u0[ii] = U_ref*(log((z[k]+z0[kk])/z0[kk])/log((z_ref+z0[kk])/z0[kk]));
                } else {
                    int kk = 3;
                    u0[ii] = U_ref*(log((z[k]+z0[kk])/z0[kk])/log((z_ref+z0[kk])/z0[kk]));
                }
                
                lambda[ii] = lambda_old[ii] = 0.0;
                
            }
        }
    }
    
    for ( int k = 0; k < nz-1; k++){
        for (int j = 0; j < ny-1; j++){
            for (int i = 0; i < nx-1; i++){
                
                int ii = i + j*nx + k*nx*ny;   // Lineralize the vectors (make it 1D)
                // Calculate divergence of initial velocity field
                R[ii] = (-2*pow(alpha1, 2.0))*(((u0[ii+1]-u0[ii])/dx)+((v0[ii + nx]-v0[ii])/dy)+((w0[ii + nx*ny]-w0[ii])/dy));
            }
        }
    }
    
    // Boundary condition at the surface (wall below)
    for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            int ii = i + j*nx;   // Lineralize the vectors (make it 1D)
            n[ii] = 0.0;
            q[ii] = 0.5;
        }
    }
    
    
    /////////////////////////////////////////////////
    //                 SOR solver              //////
    /////////////////////////////////////////////////
    int iter = 0;
    double error = 1.0;
    //double reduced_error = 0.0;
    
    while ( iter < itermax && error > tol /*&& error > reduced_error*/) {
        
	// Save previous iteration values for error calculation  
        for (int k = 0; k < nz; k++){
            for (int j = 0; j < ny; j++){
                for (int i = 0; i < nx; i++){
                    int ii = i + j*nx + k*nx*ny;   // Lineralize the vectors (make it 1D)
                    lambda_old[ii] = lambda[ii];
                }
            }
        }
        
        for (int k = 1; k < nz-1; k++){
            for (int j = 1; j < ny-1; j++){
                for (int i = 1; i < nx-1; i++){
                    
                    int ii = i + j*nx + k*nx*ny;      // Lineralize the vectors (make it 1D)
                    lambda[ii] = (omega/(2*(o[ii] + A*p[ii] + B*q[ii]))) * ((-1*(pow(dx, 2.0))*R[ii])+e[ii]*lambda[ii+1]+f[ii]*lambda[ii-1]+A*g[ii]*lambda[ii + nx]+A*h[ii]*lambda[ii - nx]+B*m[ii]*lambda[ii + nx*ny]+B*n[ii]*lambda[ii - nx*ny])+(1-omega)*lambda_old[ii];    // SOR formulation
                }
            }
        }
        
        // Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){
                int ii = i + j*nx;          // Lineralize the vectors (make it 1D)
                lambda[ii] = lambda[ii + nx*ny];
            }
        }
        
        error = 0.0;                   // Reset error value before error calculation 
        
	// Error calculation
        for (int k = 0; k < nz; k++){
            for (int j = 0; j < ny; j++){
                for (int i = 0; i < nx; i++){
                    int ii = i + j*nx + k*nx*ny;       // Lineralize the vectors (make it 1D)
                    error += fabs(lambda[ii] - lambda_old[ii])/((nx-1)*(ny-1)*(nz-1));
                }
            }
        }

        
        iter += 1;
    }
    
    std::cout << "Number of iterations:" << iter << "\n";   // Print the number of iterations
    std::cout << "Error:" << error << "\n";   // Print the number of iterations   
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //           Using Euler function to calculate u, v and w (update the velocity field)               //////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    for (int k = 0; k < nz; k++){
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){
                int ii = i + j*nx + k*nx*ny;        // Lineralize the vectors (make it 1D)
                u[ii] = u0[ii];
                v[ii] = v0[ii];
                w[ii] = w0[ii];
            }
        }
    }
    
    for (int k = 0; k < nz; k++){
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){
                int ii = i + j*nx + k*nx*ny;   // Lineralize the vectors (make it 1D)
                u[ii] = u0[ii]+(1/(2*(alpha1^2)*dx))*(lambda[ii]-lambda[ii-1]);
                v[ii] = v0[ii]+(1/(2*(alpha1^2)*dy))*(lambda[ii]-lambda[ii - nx]);
                w[ii] = w0[ii]+(1/(2*(alpha1^2)*dz))*(lambda[ii]-lambda[ii - nx*ny]);
                
            }
        }
    }

    auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";   // Print out elapsed execution time
    


    // Write data to file
    ofstream outdata;
    outdata.open("Lagrange.dat");
    if( !outdata ) {                 // File couldn't be opened
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    // Write data to file
    for (int k = 0; k < nz; k++){
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){
                int ii = i + j*nx + k*nx*ny;       // Lineralize the vectors (make it 1D)
                outdata << "\t" << i+1 << "\t" << j+1 << "\t" << k+1 << "\t" << lambda[ii] << endl;   
}
        }
    }
    outdata.close();
    

    
    
    return 0;
}



