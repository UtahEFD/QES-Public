#include "Interface.h"

__global__ void matrixMul( const long* Md, const long* Nd, long* Pd)
{

	//the index of the element of the result matrix to be computed
	int threadNum = THREADS * (blockIdx.x + blockIdx.y * BX) +
					TX * (threadIdx.y + threadIdx.z * TY) + threadIdx.x;

	long sum = 0; //running sum of the dot product
	int row = (threadNum / S) * S; //truncates down to multiple of S
	int col = threadNum % S; //gets offset of column
	int k; //index of loops

	//run dot product
	for (k = 0; k < S; k++)
		sum += Md[row + k] * Nd[k * S + col];

	//place dot product into element spot
	Pd[threadNum] = sum;
}

/*
* This function prints out the values of a matrix
*/
void Interface::printMatrix(long* matrix)
{
	int i; //index of loops

	for (i = 0; i < S * S; i++)
		//prints number in matrix, if it is the last element in a row, new line
		printf("%ld%s", matrix[i], ( i % S == S - 1 )? "\n" : ", " );
}

void Interface::multiplyMatricies_Wrapper( const long* M, const long* N, long* P)
{
	int size = S * S * sizeof(long);
	long * Md = (long *)malloc(sizeof(long) * STOTAL); //copy of primary matrix on GPU
	long * Nd = (long *)malloc(sizeof(long) * STOTAL); //copy of secondary matrix on GPU
	long * Pd = (long *)malloc(sizeof(long) * STOTAL); //copy of result matrix on GPU

	//Set sizes of grid and blcoks
	dim3 dimGrid(BX, BY, 1);
	dim3 dimBlock(TX, TY, TZ);

	//Allocate space
	cudaMalloc( (void**)&Md, size);
	cudaMalloc( (void**)&Nd, size);
	cudaMalloc( (void**)&Pd, size);

	//Copy values to GPU memory
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

	//call threads
	matrixMul<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

	//Place copy of result matrix into result matrix
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

	//Free GPU space
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);



}
