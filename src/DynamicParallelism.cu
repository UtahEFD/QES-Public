#include "DynamicParallelism.h"

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

#define BLOCKSIZE 1024
#define cudaCheck(x) _cudaCheck(x, #x ,__FILE__, __LINE__)

__device__ double error;


template<typename T>
void DynamicParallelism::_cudaCheck(T e, const char* func, const char* call, const int line){
    if(e != cudaSuccess){
        printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

/// Divergence CUDA Kernel.
/// The divergence kernel ...
///
__global__ void divergence(double *d_u0, double *d_v0, double *d_w0, double *d_R, float *d_e, float *d_f, float *d_g,
						float *d_h, float *d_m, float *d_n, int alpha1, int  nx, int  ny, int nz,float dx,float dy,float *d_dz_array)
{

    int icell_cent = blockDim.x*blockIdx.x+threadIdx.x;
    int k = icell_cent/((nx-1)*(ny-1));
    int j = (icell_cent - k*(nx-1)*(ny-1))/(nx-1);
    int i = icell_cent - k*(nx-1)*(ny-1) - j*(nx-1);
    int icell_face = i + j*nx + k*nx*ny;

    // Would be nice to figure out how to not have this branch check...
    if((i<nx-1)&&(j<ny-1)&&(k<nz-1)) {

        // Divergence equation
        d_R[icell_cent] = (-2*pow(alpha1, 2.0))*((( d_e[icell_cent] * d_u0[icell_face+1]       - d_f[icell_cent] * d_u0[icell_face]) * dx ) +
                                               (( d_g[icell_cent] * d_v0[icell_face + nx]    - d_h[icell_cent] * d_v0[icell_face]) * dy ) +
                                               ( d_m[icell_cent] * d_dz_array[k]*0.5*(d_dz_array[k]+d_dz_array[k+1]) * d_w0[icell_face + nx*ny]
                                                - d_n[icell_cent] * d_w0[icell_face] * d_dz_array[k]*0.5*(d_dz_array[k]+d_dz_array[k-1]) ));
    }
}


/// SOR RedBlack Kernel.
///
///
__global__ void SOR_RB(double *d_lambda, int nx, int ny, int nz, float omega, float  A, float  B, float  dx, float *d_e,
						float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, double *d_R, int offset)
{
    int icell_cent = blockDim.x*blockIdx.x+threadIdx.x;
    int k = icell_cent/((nx-1)*(ny-1));
    int j = (icell_cent - k*(nx-1)*(ny-1))/(nx-1);
    int i = icell_cent - k*(nx-1)*(ny-1) - j*(nx-1);

    if ( (i > 0) && (i < nx-2) && (j > 0) && (j < ny-2) && (k < nz-2) && (k > 0) && ((i+j+k)%2) == offset ){

        d_lambda[icell_cent] = (omega / ( d_e[icell_cent] + d_f[icell_cent] + d_g[icell_cent] +
                                          d_h[icell_cent] + d_m[icell_cent] + d_n[icell_cent])) *
            ( d_e[icell_cent] * d_lambda[icell_cent+1]               + d_f[icell_cent] * d_lambda[icell_cent-1] +
              d_g[icell_cent] * d_lambda[icell_cent + (nx-1)]        + d_h[icell_cent] * d_lambda[icell_cent - (nx-1)] +
              d_m[icell_cent] * d_lambda[icell_cent + (nx-1)*(ny-1)] +
              d_n[icell_cent] * d_lambda[icell_cent - (nx-1)*(ny-1)] - d_R[icell_cent] ) +
            (1.0 - omega) * d_lambda[icell_cent];    /// SOR formulation
    }
}

__global__ void assign_lambda_to_lambda_old(double *d_lambda, double *d_lambda_old, int nx, int ny, int nz)
{
    int ii = blockDim.x*blockIdx.x+threadIdx.x;

    if(ii < (nz-1)*(ny-1)*(nx-1)) {
        d_lambda_old[ii] = d_lambda[ii];
    }
}

__global__ void applyNeumannBC(double *d_lambda, int nx, int ny)
{
    // Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
    int ii = blockDim.x*blockIdx.x+threadIdx.x;

    if(ii < nx*ny) {
      d_lambda[ii] = d_lambda[ii + 1*(nx-1)*(ny-1)];
    }
}

__global__ void calculateError(double *d_lambda, double *d_lambda_old, int nx, int ny, int nz,
                               double *d_value,
                               double *d_bvalue)
{
    int d_size = (nx-1)*(ny-1)*(nz-1);
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
__global__ void finalVelocity(double *d_u0, double *d_v0, double *d_w0, double *d_lambda, double *d_u, double *d_v,
							 double *d_w, int *d_icellflag, float *d_f, float *d_h, float *d_n, int alpha1, int alpha2,
							 float dx, float dy, float dz, float *d_dz_array, int  nx, int  ny, int nz)
{

    int icell_face = blockDim.x*blockIdx.x+threadIdx.x;
    int k = icell_face/(nx*ny);
    int j = (icell_face - k*nx*ny)/nx;
    int i = icell_face - k*nx*ny - j*nx;
    int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values

    if((i>= 0) && (j>= 0) && (k >= 0) && (i<nx)&&(j<ny)&&(k<nz)){

        d_u[icell_face] = d_u0[icell_face];
        d_v[icell_face] = d_v0[icell_face];
        d_w[icell_face] = d_w0[icell_face];

    }


    if ((i > 0) && (i < nx-1) && (j > 0) && (j < ny-1) && (k < nz-1) && (k > 0)) {

        d_u[icell_face] = d_u0[icell_face]+(1/(2*pow(alpha1, 2.0)))*d_f[icell_cent]*dx*
						 (d_lambda[icell_cent]-d_lambda[icell_cent-1]);
        d_v[icell_face] = d_v0[icell_face]+(1/(2*pow(alpha1, 2.0)))*d_h[icell_cent]*dy*
						 (d_lambda[icell_cent]-d_lambda[icell_cent - (nx-1)]);
        d_w[icell_face] = d_w0[icell_face]+(1/(2*pow(alpha2, 2.0)))*d_n[icell_cent]*d_dz_array[k]*
						 (d_lambda[icell_cent]-d_lambda[icell_cent - (nx-1)*(ny-1)]);

    }


    if ((i >= 0) && (i < nx-1) && (j >= 0) && (j < ny-1) && (k < nz-1) && (k > 0) && (d_icellflag[icell_cent] == 0) ) {
        d_u[icell_face] = 0;
        d_u[icell_face+1] = 0;
        d_v[icell_face] = 0;
        d_v[icell_face+nx] = 0;
        d_w[icell_face] = 0;
        d_w[icell_face+nx*ny] = 0;

    }
}


/// SOR iteration kernel
///
__global__ void SOR_iteration (double *d_lambda, double *d_lambda_old, int nx, int ny, int nz, float omega, float  A,
								float  B, float  dx, float dy, float dz, float *d_dz_array, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n,
								double *d_R, int itermax, double tol, double *d_value, double *d_bvalue, double *d_u0,
								double *d_v0, double *d_w0,int alpha1, int alpha2, double *d_u,
								double *d_v, double *d_w, int *d_icellflag)
{
    int iter = 0;
    error = 1.0;

    // Calculate divergence of initial velocity field
    dim3 numberOfThreadsPerBlock(BLOCKSIZE,1,1);
    dim3 numberOfBlocks(ceil(((nx-1)*(ny-1)*(nz-1))/(double) (BLOCKSIZE)),1,1);

    // Invoke divergence kernel
    divergence<<<numberOfBlocks,numberOfThreadsPerBlock>>>(d_u0,d_v0,d_w0,d_R,d_e,d_f,d_g,d_h,d_m,d_n,alpha1,nx,ny,nz,dx,dy,
															d_dz_array);
    // Iterate untill convergence is reached
    while ( (iter < itermax) && (error > tol)) {

        // Save previous iteration values for error calculation
        assign_lambda_to_lambda_old<<<numberOfBlocks,numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, nx, ny, nz);
        cudaDeviceSynchronize();
        // SOR part
        int offset = 0;   // red nodes
        // Invoke red-black SOR kernel for red nodes
        SOR_RB<<<numberOfBlocks,numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, dx, d_e, d_f, d_g, d_h, d_m,
															d_n, d_R, offset);
        cudaDeviceSynchronize();
        offset = 1;    // black nodes
        // Invoke red-black SOR kernel for black nodes
        SOR_RB<<<numberOfBlocks,numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, dx, d_e, d_f, d_g, d_h, d_m,
															d_n, d_R,offset);
        cudaDeviceSynchronize();
        dim3 numberOfBlocks2(ceil(((nx-1)*(ny-1))/(double) (BLOCKSIZE)),1,1);
        // Invoke kernel to apply Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
        applyNeumannBC<<<numberOfBlocks2,numberOfThreadsPerBlock>>>(d_lambda, nx, ny);
        cudaDeviceSynchronize();
        // Error calculation
        calculateError<<<numberOfBlocks,numberOfThreadsPerBlock>>>(d_lambda,d_lambda_old, nx, ny, nz, d_value,d_bvalue);
        cudaDeviceSynchronize();

        iter += 1;

    }
    printf("number of iteration = %d\n", iter);
    printf("error = %2.9f\n", error);
    dim3 numberOfBlocks3(ceil((nx*ny*nz)/(double) (BLOCKSIZE)),1,1);
    // Invoke final velocity (Euler) kernel
    finalVelocity<<<numberOfBlocks3,numberOfThreadsPerBlock>>>(d_u0,d_v0,d_w0,d_lambda,d_u,d_v,d_w,d_icellflag,d_f,d_h,d_n,
																alpha1,alpha2,dx,dy,dz, d_dz_array,nx,ny,nz);

}



void DynamicParallelism::solve(bool solveWind)
{
    auto startTotal = std::chrono::high_resolution_clock::now(); // Start
                                                                 // recording
                                                                 // execution
                                                                 // time
    int numblocks = (numcell_cent/BLOCKSIZE)+1;

    std::vector<double> value(numcell_cent,0.0);
    std::vector<double> bvalue(numblocks,0.0);
    double *d_u0, *d_v0, *d_w0;
    double *d_value,*d_bvalue;
    float *d_x,*d_y,*d_z;
    double *d_u, *d_v, *d_w;
    int *d_icellflag;
    float *d_dz_array;

    auto start = std::chrono::high_resolution_clock::now(); // Start recording execution time

    cudaMalloc((void **) &d_e, numcell_cent * sizeof(float));
    cudaMalloc((void **) &d_f, numcell_cent * sizeof(float));
    cudaMalloc((void **) &d_g, numcell_cent * sizeof(float));
    cudaMalloc((void **) &d_h, numcell_cent * sizeof(float));
    cudaMalloc((void **) &d_m, numcell_cent * sizeof(float));
    cudaMalloc((void **) &d_n, numcell_cent * sizeof(float));
    cudaMalloc((void **) &d_R, numcell_cent * sizeof(double));
    cudaMalloc((void **) &d_lambda, numcell_cent * sizeof(double));
    cudaMalloc((void **) &d_lambda_old, numcell_cent * sizeof(double));
    cudaMalloc((void **) &d_icellflag, numcell_cent * sizeof(int));
    cudaMalloc((void **) &d_u0,numcell_face*sizeof(double));
    cudaMalloc((void **) &d_v0,numcell_face*sizeof(double));
    cudaMalloc((void **) &d_w0,numcell_face*sizeof(double));
    cudaMalloc((void **) &d_value,numcell_cent*sizeof(double));
    cudaMalloc((void **) &d_bvalue,numblocks*sizeof(double));
    cudaMalloc((void **) &d_x,nx*sizeof(float));
    cudaMalloc((void **) &d_y,ny*sizeof(float));
    cudaMalloc((void **) &d_z,nz*sizeof(float));
    cudaMalloc((void **) &d_dz_array,(nz-1)*sizeof(float));
    cudaMalloc((void **) &d_u,numcell_face*sizeof(double));
    cudaMalloc((void **) &d_v,numcell_face*sizeof(double));
    cudaMalloc((void **) &d_w,numcell_face*sizeof(double));


    cudaMemcpy(d_icellflag,icellflag.data(),numcell_cent*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_u0,u0.data(),numcell_face*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_v0,v0.data(),numcell_face*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_w0,w0.data(),numcell_face*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_R,R.data(),numcell_cent*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_value , value.data() , numcell_cent * sizeof(double) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_bvalue , bvalue.data() , numblocks * sizeof(double) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_e , e.data() , numcell_cent * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_f , f.data() , numcell_cent * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_g , g.data() , numcell_cent * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_h , h.data() , numcell_cent * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_m , m.data() , numcell_cent * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_n , n.data() , numcell_cent * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_x , x.data() , nx * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_y , y.data() , ny * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_z , z.data() , nz * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dz_array , dz_array.data() , (nz-1) * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda , lambda.data() , numcell_cent * sizeof(double) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda_old , lambda_old.data() , numcell_cent * sizeof(double) , cudaMemcpyHostToDevice);


    /////////////////////////////////////////////////
    //                 SOR solver              //////
    /////////////////////////////////////////////////


    // Invoke the main (mother) kernel
    SOR_iteration<<<1,1>>>(d_lambda,d_lambda_old, nx, ny, nz, omega, A, B, dx, dy,dz, d_dz_array, d_e, d_f, d_g, d_h, d_m, d_n, d_R,itermax,tol,d_value,d_bvalue,d_u0,d_v0,d_w0,alpha1,alpha2,d_u,d_v,d_w,d_icellflag);
    cudaCheck(cudaGetLastError());

    cudaMemcpy (lambda.data() , d_lambda , numcell_cent * sizeof(double) , cudaMemcpyDeviceToHost);
    cudaMemcpy(u.data(),d_u,numcell_face*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(),d_v,numcell_face*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(w.data(),d_w,numcell_face*sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree (d_lambda);
    cudaFree (d_e);
    cudaFree (d_f);
    cudaFree (d_g);
    cudaFree (d_h);
    cudaFree (d_m);
    cudaFree (d_n);
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
    cudaFree (d_dz_array);
    cudaFree (d_icellflag);

    auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";   // Print out elapsed execution time

}
