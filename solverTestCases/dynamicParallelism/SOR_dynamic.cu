
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

#define BLOCKSIZE 1024
#define Blocksize_x 16
#define Blocksize_y 16
#define Blocksize_z 4
#define cudaCheck(x) _cudaCheck(x, #x ,__FILE__, __LINE__)


using namespace std::chrono;
using namespace std;
using std::ofstream;
using std::ifstream;
using std::istringstream;
using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;
using std::to_string;
__device__ double error;

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
        R_out[ii] = (-2*pow(alpha1, 2.0))*(((uin[ii+1]-uin[ii])/dx)+((vin[ii + nx]-vin[ii])/dy)+((win[ii + nx*ny]-win[ii])/dy));    // Divergence equation
    }
}


__global__ void SOR_RB(double *d_lambda, int nx, int ny, int nz, float omega, float  A, float  B, float  dx, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, float *d_o, float *d_p, float *d_q, double *d_R, int offset){
    
    int ii = blockDim.x*blockIdx.x+threadIdx.x+(nx*ny);
    int k = ii/(nx*ny);
    int j = (ii - k*nx*ny)/nx;
    int i = ii - k*nx*ny - j*nx;
    
    if ( (i > 0) && (i < nx-1) && (j > 0) && (j < ny-1) && (k < nz-1) && ((i+j+k)%2) == offset ){
        
        d_lambda[ii] = (omega/(2*(d_o[ii] + A*d_p[ii] + B*d_q[ii]))) * ((-1*(pow(dx, 2.0))*d_R[ii])+d_e[ii]*d_lambda[ii+1]+d_f[ii]*d_lambda[ii-1]+
                       A*d_g[ii]*d_lambda[ii + nx]+A*d_h[ii]*d_lambda[ii - nx]+B*d_m[ii]*d_lambda[ii + nx*ny]+B*d_n[ii]*d_lambda[ii - nx*ny])+(1-omega)*d_lambda[ii];    // SOR formulation
    }
}

__global__ void assign_lambda_to_lambda_old(double *d_lambda, double *d_lambda_old, int nx, int ny, int nz) {
    
    int ii = blockDim.x*blockIdx.x+threadIdx.x;
    
    if(ii < nz*ny*nx) {
        d_lambda_old[ii] = d_lambda[ii];
    }
    
}

__global__ void applyNeumannBC(double *d_lambda, int nx, int ny) {
    // Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
    int ii = blockDim.x*blockIdx.x+threadIdx.x;
    
    if(ii < nx*ny) {
      d_lambda[ii] = d_lambda[ii + 1*nx*ny];
    }
}

__global__ void calculateError(double *d_lambda, double *d_lambda_old, int nx, int ny, int nz, double *d_value, double *d_bvalue){


	int d_size = nx*ny*nz;
	int ii = blockDim.x*blockIdx.x+threadIdx.x;
	int numblocks = (d_size/BLOCKSIZE) +1;

	if (ii < d_size){
	    d_value[ii] = fabs(d_lambda[ii] - d_lambda_old[ii])/((nx-1)*(ny-1)*(nz-1));
	}
	__syncthreads();
        double sum = 0.0;
	if (threadIdx.x > 0){ 
	    return;
	}
	if (threadIdx.x == 0) {
	     for (int j=0; j<BLOCKSIZE; j++){
		int index = blockIdx.x*blockDim.x+j;
		if (index<d_size){
			sum += d_value[index]; 
		}
	     }
	}
	
	__syncthreads();
	d_bvalue[blockIdx.x] = sum;

	if (ii>0){
	    return;
	}

	error = 0.0;
	if (ii==0){
	    for (int k =0; k<numblocks; k++){
		error += d_bvalue[k];
	    }
	}

 }

// Euler Final Velocity kernel
__global__ void finalVelocity(double *uin, double *vin, double *win, double *lambdain, double *uf, double *vf,double *wf, int alpha1, float dx, float dy, float dz, int  nx, int  ny, int nz){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int ii = i + j*nx + k*nx*ny;    
    if((i<nx)&&(j<ny)&&(k<nz)){
        uf[ii] = uin[ii]+(1/(2*(alpha1^2)*dx))*(lambdain[ii]-lambdain[ii-1]);
        vf[ii] = vin[ii]+(1/(2*(alpha1^2)*dy))*(lambdain[ii]-lambdain[ii - nx]);
        wf[ii] = win[ii]+(1/(2*(alpha1^2)*dz))*(lambdain[ii]-lambdain[ii - nx*ny]);
    }
}

__global__ void SOR_iteration (double *d_lambda, double *d_lambda_old, int nx, int ny, int nz, float omega, float  A, float  B, float  dx, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, float *d_o, float *d_p, float *d_q, double *d_R, int itermax, double tol, double *d_value, double *d_bvalue, double *d_u0, double *d_v0, double *d_w0,int alpha1, float dy, float dz, double *d_u, double *d_v, double *d_w) {
    int iter = 0;
    error = 1.0;

    // Calculate divergence of initial velocity field
    dim3 dimGrid(ceil(nx/(double)Blocksize_x),ceil(ny/(double)Blocksize_y),ceil(nz/(double)Blocksize_z));
    dim3 dimBlock(Blocksize_x,Blocksize_y,Blocksize_z);
    // Invoke divergence kernel
    divergence<<<dimGrid,dimBlock>>>(d_u0,d_v0,d_w0,d_R,alpha1,nx,ny,nz,dx,dy,dz);

    // Iterate untill convergence is reached
    while ( (iter < itermax) && (error > tol)) {
        
        dim3 numberOfThreadsPerBlock(BLOCKSIZE,1,1);
        dim3 numberOfBlocks(ceil((nx*ny*nz)/(double) (BLOCKSIZE)),1,1);
	// Save previous iteration values for error calculation 
        assign_lambda_to_lambda_old<<<numberOfBlocks,numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, nx, ny, nz);
        // SOR part
        dim3 numberOfThreadsPerBlock1(BLOCKSIZE,1,1);
        dim3 numberOfBlocks1(ceil((nx*ny*(nz-2))/(double) (BLOCKSIZE)),1,1);
        int offset = 0;   // red nodes
	// Invoke red-black SOR kernel for red nodes
        SOR_RB<<<numberOfBlocks1,numberOfThreadsPerBlock1>>>(d_lambda, nx, ny, nz, omega, A, B, dx, d_e, d_f, d_g, d_h, d_m, d_n, d_o, d_p, d_q, d_R, offset);
        cudaDeviceSynchronize();
	offset = 1;    // black nodes
	// Invoke red-black SOR kernel for black nodes
        SOR_RB<<<numberOfBlocks1,numberOfThreadsPerBlock1>>>(d_lambda, nx, ny, nz, omega, A, B, dx, d_e, d_f, d_g, d_h, d_m, d_n, d_o, d_p, d_q, d_R,offset);
        cudaDeviceSynchronize();
	dim3 numberOfThreadsPerBlock2(BLOCKSIZE,1,1);
	dim3 numberOfBlocks2(ceil((nx*ny)/(double) (BLOCKSIZE)),1,1);
	// Invoke kernel to apply Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
        applyNeumannBC<<<numberOfBlocks2,numberOfThreadsPerBlock2>>>(d_lambda, nx, ny);
        
        // Error calculation
	calculateError<<<numberOfBlocks,numberOfThreadsPerBlock>>>(d_lambda,d_lambda_old, nx, ny, nz, d_value,d_bvalue);

        iter += 1;
        
    }
    printf("number of iteration = %d\n", iter);
    printf("error = %2.9f\n", error);
    // Invoke final velocity (Euler) kernel
    finalVelocity<<<dimGrid,dimBlock>>>(d_u0,d_v0,d_w0,d_lambda,d_u,d_v,d_w,alpha1,dx,dy,dz,nx,ny,nz);
}



int main(int argc, const char * argv[]) {
    

    auto start = std::chrono::high_resolution_clock::now(); // Start recording execution time
    int nx = 200;     // Number of cells in x-dir
    int ny = 200;     // Number of cells in y-dir
    int nz = 50;      // Number of cells in z-dir
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
    int itermax = 500;    // Maximum number of iterations
  
    float *d_e, *d_f, *d_g, *d_h, *d_m, *d_n, *d_o, *d_p, *d_q;
    double *d_R, *d_lambda, *d_lambda_old;
    
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

    // Declare Lagrange multipliers
    double *lambda, *lambda_old;
    lambda = new double [d_size];
    cudaMalloc((void **) &d_lambda, d_size * sizeof(double));
    cudaMalloc((void **) &d_lambda_old, d_size * sizeof(double));
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
    
    int numblocks = (d_size/BLOCKSIZE)+1;
    double *value, *bvalue;
    value = new double [d_size];
    bvalue = new double [numblocks];    
    
    
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
                
                //lambda[ii] = lambda_old[ii] = 0.0;
                
            }
        }
    }   

    //auto start = std::chrono::high_resolution_clock::now(); // Start recording execution time
    // Allocate GPU memory
    double *d_u0, *d_v0, *d_w0;
    cudaMalloc((void **) &d_u0,d_size*sizeof(double));
    cudaMalloc((void **) &d_v0,d_size*sizeof(double));
    cudaMalloc((void **) &d_w0,d_size*sizeof(double));
    // Initialize GPU input/output
    cudaMemcpy(d_u0,u0,d_size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_v0,v0,d_size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_w0,w0,d_size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_R,R,d_size*sizeof(double),cudaMemcpyHostToDevice);

        
    // Boundary condition at the surface (wall below)
    for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            int ii = i + j*nx;   // Lineralize the vectors (make it 1D)
            n[ii] = 0.0;
            q[ii] = 0.5;
        }
    }
	
	double *d_value,*d_bvalue;
	float *d_x,*d_y,*d_z;
	cudaMalloc((void **) &d_value,d_size*sizeof(double));
	cudaMalloc((void **) &d_bvalue,numblocks*sizeof(double));
	cudaMalloc((void **) &d_x,nx*sizeof(float));
	cudaMalloc((void **) &d_y,ny*sizeof(float));
	cudaMalloc((void **) &d_z,nz*sizeof(float));
	cudaMemcpy(d_value , value , d_size * sizeof(double) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_bvalue , bvalue , numblocks * sizeof(double) , cudaMemcpyHostToDevice);      
	cudaMemcpy(d_e , e , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_f , f , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_g , g , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_h , h , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_m , m , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_n , n , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_o , o , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_p , p , d_size * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_q , q , d_size * sizeof(float) , cudaMemcpyHostToDevice);
	cudaMemcpy(d_x , x , nx * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_y , y , ny * sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_z , z , nz * sizeof(float) , cudaMemcpyHostToDevice);
	
	double *d_u, *d_v, *d_w;
	cudaMalloc((void **) &d_u,d_size*sizeof(double));
	cudaMalloc((void **) &d_v,d_size*sizeof(double));
	cudaMalloc((void **) &d_w,d_size*sizeof(double));

    /////////////////////////////////////////////////
    //                 SOR solver              //////
    /////////////////////////////////////////////////
    
    cudaMemcpy(d_lambda , lambda , d_size * sizeof(double) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda_old , lambda_old , d_size * sizeof(double) , cudaMemcpyHostToDevice);
    // Invoke the main (mother) kernel
    SOR_iteration<<<1,1>>>(d_lambda,d_lambda_old, nx, ny, nz, omega, A, B, dx, d_e, d_f, d_g, d_h, d_m, d_n, d_o, d_p, d_q, d_R,itermax,tol,d_value,d_bvalue,d_u0,d_v0,d_w0,alpha1,dy,dz,d_u,d_v,d_w);
    cudaCheck(cudaGetLastError()); 
    
    cudaMemcpy (lambda , d_lambda , d_size * sizeof(double) , cudaMemcpyDeviceToHost);
    cudaMemcpy(u,d_u,d_size*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(v,d_v,d_size*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(w,d_w,d_size*sizeof(double),cudaMemcpyDeviceToHost);

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
    cudaFree (d_value);
    cudaFree (d_bvalue);
    cudaFree (d_u0);
    cudaFree (d_v0);
    cudaFree (d_w0);
    cudaFree (d_u);
    cudaFree (d_v);
    cudaFree (d_w);
    cudaFree (d_x);
    cudaFree (d_y);
    cudaFree (d_z);

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

