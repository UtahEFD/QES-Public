
#include <stdio.h>
#include "DynamicParallelism.h"
#include <cuda.h>
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

template<typename T>
void _cudaCheck(T e, const char* func, const char* call, const int line)
{
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