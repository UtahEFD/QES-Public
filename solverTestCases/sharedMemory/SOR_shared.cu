

#include <stdio.h>
#include "cuda.h"
#include <string.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCKSIZE 32
#define Blocksize_x 4
#define Blocksize_y 4
#define Blocksize_z 2
#define cudaCheck(x) _cudaCheck(x, #x ,__FILE__, __LINE__)


using namespace std;
using std::ofstream;
using std::cerr;
using std::endl;
using std::vector;

template<typename T>
void _cudaCheck(T e, const char* func, const char* call, const int line){
    if(e != cudaSuccess){
        printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

// Divergence kernel
__global__ void divergence(double *uin, double *vin, double *win, double *R_out, int alpha1, int  nx, int  ny, int nz, float dx,float dy, float dz){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int ii = i + j*nx + k*nx*ny;
    if((i<nx-1)&&(j<ny-1)&&(k<nz-1)){
        R_out[ii] = (-2*pow(alpha1, 2.0))*(((uin[ii+1]-uin[ii])/dx)+((vin[ii + nx]-vin[ii])/dy)+((win[ii + nx*ny]-win[ii])/dy));   // Divergence equation
    }
}

// Euler Final Velocity kernel
__global__ void finalVelocity(double *uin, double *vin, double *win, double *lambdain, double *uf, double *vf,double *wf, int alpha1, float dx, float dy, float dz, int  nx, int  ny, int nz){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int ii = i + j*nx + k*nx*ny; 
      
    if((i<nx)&&(j<ny)&&(i>0)&&(j>0)&&(k>0)&&(k<nz)){
        uf[ii] = uin[ii]+(1/(2*(alpha1^2)*dx))*(lambdain[ii]-lambdain[ii-1]);
        vf[ii] = vin[ii]+(1/(2*(alpha1^2)*dy))*(lambdain[ii]-lambdain[ii - nx]);
        wf[ii] = win[ii]+(1/(2*(alpha1^2)*dz))*(lambdain[ii]-lambdain[ii - nx*ny]);
    }
}


__global__ void SOR_RB_shared(double *d_lambda, int nx, int ny, int nz, float omega, float  A, float  B, float  dx, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, float *d_o, float *d_p, float *d_q, double *d_R, int offset){
    
    
    __shared__ double lambda_cn[BLOCKSIZE+2];
    __shared__ double lambda_bi[BLOCKSIZE];
    __shared__ double lambda_fi[BLOCKSIZE];
    __shared__ double lambda_di[BLOCKSIZE];
    __shared__ double lambda_ui[BLOCKSIZE];
    
    int tidx = threadIdx.x;
    int ii = blockDim.x*blockIdx.x+threadIdx.x+(nx*ny);
    int bi = ii - nx;              // j-1 index (bi)
    int fi = ii + nx;             // j+1 index (fi)
    int di = ii - nx*ny;           // k-1 index (di)
    int ui = ii + nx*ny;          // k+1 index (ui)

    if (tidx == 0){
        lambda_cn[0] = d_lambda[ii-1];
        int index = blockDim.x*(blockIdx.x+1)+(nx*ny);
        lambda_cn[blockDim.x+1] = d_lambda[index];
    }

    // Load data to the shared memory
    lambda_cn[tidx+1] = d_lambda[ii];
    lambda_bi[tidx] = d_lambda[bi];
    lambda_fi[tidx] = d_lambda[fi];
    lambda_di[tidx] = d_lambda[di];
    lambda_ui[tidx] = d_lambda[ui];
	
    int k = ii/(nx*ny);
    int j = (ii - k*nx*ny)/nx;
    int i = ii - k*nx*ny - j*nx;
 
    __syncthreads();


    if ( (i > 0) && (i < nx-1) && (j > 0) && (j < ny-1) && (k < nz-1) && ((i+j+k)%2) == offset){
            
             
              lambda_cn[tidx+1] = (omega/(2*(d_o[ii] + A*d_p[ii] + B*d_q[ii]))) * ((-1*(pow(dx, 2.0))*d_R[ii])+d_e[ii]*lambda_cn[tidx+2]+d_f[ii]*lambda_cn[tidx]+A*d_g[ii]*lambda_fi[tidx]+
                               A*d_h[ii]*lambda_bi[tidx]+B*d_m[ii]*lambda_ui[tidx]+B*d_n[ii]*lambda_di[tidx])+(1-omega)*lambda_cn[tidx+1];    // SOR formulation*/ 
   	      d_lambda[ii] = lambda_cn[tidx+1];
    }


    }


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
  
    float *d_e, *d_f, *d_g, *d_h, *d_m, *d_n, *d_o, *d_p, *d_q;
    double *d_u, *d_v, *d_w, *d_R, *d_lambda;
    
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

    cudaMalloc((void **) &d_e, d_size * sizeof(float));
    cudaMalloc((void **) &d_f, d_size * sizeof(float));
    cudaMalloc((void **) &d_g, d_size * sizeof(float));
    cudaMalloc((void **) &d_h, d_size * sizeof(float));
    cudaMalloc((void **) &d_m, d_size * sizeof(float));
    cudaMalloc((void **) &d_n, d_size * sizeof(float));
    cudaMalloc((void **) &d_o, d_size * sizeof(float));
    cudaMalloc((void **) &d_p, d_size * sizeof(float));
    cudaMalloc((void **) &d_q, d_size * sizeof(float));

    // Declare initial wind profile (u0,v0,w0)
    double *u0, *v0, *w0;
    u0 = new double [d_size];
    v0 = new double [d_size];
    w0 = new double [d_size];
    
    // Declare divergence of initial velocity field
    double * R;
    R = new double [d_size];
    cudaMalloc((void **) &d_R, d_size * sizeof(double));    
    // Declare final velocity field
    double *u, *v, *w;
    u = new double [d_size];
    v = new double [d_size];
    w = new double [d_size];

    cudaMalloc((void **) &d_u, d_size * sizeof(double));
    cudaMalloc((void **) &d_v, d_size * sizeof(double));
    cudaMalloc((void **) &d_w, d_size * sizeof(double));

    // Declare Lagrange multipliers
    double *lambda, *lambda_old;
    lambda = new double [d_size];
    cudaMalloc((void **) &d_lambda, d_size * sizeof(double));
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
                v0[ii] = w0[ii] = 0.0;
                
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
    

    // Calculate divergence of initial velocity field
    // Threads per block and number of blocks
    dim3 dimGrid(ceil(nx/(double)Blocksize_x),ceil(ny/(double)Blocksize_y),ceil(nz/(double)Blocksize_z));
    dim3 dimBlock(Blocksize_x,Blocksize_y,Blocksize_z);
    // Allocate GPU memory
    double *u_in, *v_in, *w_in, *R_out;
    cudaMalloc((void **) &u_in,d_size*sizeof(double));
    cudaMalloc((void **) &v_in,d_size*sizeof(double));
    cudaMalloc((void **) &w_in,d_size*sizeof(double));
    cudaMalloc((void **) &R_out,d_size*sizeof(double));
    // Initialize GPU input/output
    cudaMemcpy(u_in,u0,d_size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(v_in,v0,d_size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(w_in,w0,d_size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(R_out,R,d_size*sizeof(double),cudaMemcpyHostToDevice);
    // Invoke kernel
    divergence<<<dimGrid,dimBlock>>>(u_in,v_in,w_in,R_out,alpha1,nx,ny,nz,dx,dy,dz);
    cudaCheck(cudaGetLastError());  
    // Copy data back to host
    cudaMemcpy(R,R_out,d_size*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(u_in);
    cudaFree(v_in);
    cudaFree(w_in);
    cudaFree(R_out);
        
    // Boundary condition at the surface (wall below)
    for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            int ii = i + j*nx;   // Lineralize the vectors (make it 1D)
            n[ii] = 0.0;
            q[ii] = 0.5;
        }
    }
    
      
	// Copy data from host to device
	cudaMemcpy(d_e , e , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_f , f , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_g , g , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_h , h , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_m , m , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_n , n , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_o , o , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_p , p , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_q , q , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_R , R , d_size * sizeof(double) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_lambda , lambda , d_size * sizeof(double) , cudaMemcpyHostToDevice);

    /////////////////////////////////////////////////
    //                 SOR solver              //////
    /////////////////////////////////////////////////
    int iter = 0;
    double error = 1.0;

    // Threads per block and number of blocks
    dim3 numberOfThreadsPerBlock(BLOCKSIZE,1,1);
    dim3 numberOfBlocks(ceil((nx*ny*(nz-2))/(double) (BLOCKSIZE)),1,1);
    printf("number of threads per block = %d\n", numberOfThreadsPerBlock);
    printf("number of blocks = %d\n", numberOfBlocks);
    
    // Main solver loop
    while ( (iter < itermax) && (error > tol)) {
        
	// Save previous iteration values for error calculation 
        for (int k = 0; k < nz; k++){
            for (int j = 0; j < ny; j++){
                for (int i = 0; i < nx; i++){
                    int ii = i + j*nx + k*nx*ny;   // Lineralize the vectors (make it 1D)
                    lambda_old[ii] = lambda[ii];
                }
            }
        }
        

        cudaMemcpy(d_lambda , lambda , d_size * sizeof(double) , cudaMemcpyHostToDevice);
        int offset = 0;                     // Red nodes pass
	// Invoke red-black SOR kernel
	SOR_RB_shared<<<numberOfBlocks,numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, dx, d_e, d_f, d_g, d_h, d_m, d_n, d_o, d_p, d_q, d_R,offset);
    	cudaCheck(cudaGetLastError());    
        
	offset = 1;                         // Black nodes pass
        SOR_RB_shared<<<numberOfBlocks,numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, dx, d_e, d_f, d_g, d_h, d_m, d_n, d_o, d_p, d_q, d_R,offset);
        cudaMemcpy (lambda , d_lambda , d_size * sizeof(double) , cudaMemcpyDeviceToHost);        
        
        // Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
        for (int j = 0; j < ny; j++){
            for (int i = 0; i < nx; i++){
                int ii = i + j*nx;          // Lineralize the vectors (make it 1D)
                lambda[ii] = lambda[ii + nx*ny];
            }
        }
        
        error = 0.0;                     // Reset error value before error calculation 

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
       
    cudaFree (d_lambda);
    cudaFree (d_e);
    cudaFree (d_f);
    cudaFree (d_g);
    cudaFree (d_h);
    cudaFree (d_m);
    cudaFree (d_n);
    cudaFree (d_o);
    cudaFree (d_p);
    cudaFree (d_q);
    cudaFree (d_R);

    std::cout << "Error:" << error << "\n";   // Print the number of iterations 
    std::cout << "Number of iterations:" << iter << "\n";   // Print the number of iterations 
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //           Using Euler function to calculate u, v and w (update the velocity field)               //////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Allocate GPU memory
    double *u_out, *v_out, *w_out,*lambda_in;
    cudaMalloc((void **) &u_in,d_size*sizeof(double));
    cudaMalloc((void **) &v_in,d_size*sizeof(double));
    cudaMalloc((void **) &w_in,d_size*sizeof(double));
    cudaMalloc((void **) &u_out,d_size*sizeof(double));
    cudaMalloc((void **) &v_out,d_size*sizeof(double));
    cudaMalloc((void **) &w_out,d_size*sizeof(double));
    cudaMalloc((void **) &lambda_in,d_size*sizeof(double));
    // Initialize GPU input/output
    cudaMemcpy(u_in,u0,d_size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(v_in,v0,d_size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(w_in,w0,d_size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(u_out,u,d_size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(v_out,v,d_size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(w_out,w,d_size*sizeof(double),cudaMemcpyHostToDevice);    
    cudaMemcpy(lambda_in,lambda,d_size*sizeof(double),cudaMemcpyHostToDevice);
    // Invoke kernel
    finalVelocity<<<dimGrid,dimBlock>>>(u_in,v_in,w_in,lambda_in,u_out,v_out,w_out,alpha1,dx,dy,dz,nx,ny,nz);
    // Copy data back to host
    cudaMemcpy(u,u_out,d_size*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(v,v_out,d_size*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(w,w_out,d_size*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(u_in);
    cudaFree(v_in);
    cudaFree(w_in);
    cudaFree(lambda_in);
    cudaFree(u_out);
    cudaFree(v_out);
    cudaFree(w_out);

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

