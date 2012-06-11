#include "Cell.cuh"

#include <cstdlib>
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include "vector_functions.h"

// __device__ Cell*  dCells;
__device__ float*  gf;

//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
  numThreads = min(blockSize, n);
  numBlocks = iDivUp(n, numThreads);
}

__global__
void testGCell(Cell*  dCells, uint numParticles )
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;
    
//     volatile Cell cell = pos[index]; 
//     df[index] = gf[index];// - 10.f;
//     df[index] -= 10.f;
    dCells[index].wind += make_float3(1.f, 1.f, 1.f);
}
  
int main()
{
  uint m_numGridCells = 1000;
  Cell*  hCells = new Cell[m_numGridCells];////64*64*64
  float*  hf = new float[m_numGridCells];////64*64*64
  float*  df;
  Cell*  dCells;
//   memset(m_hCells, 0, m_numGridCells*sizeof(Cell));
  Cell cell;
  for(int i=0; i<m_numGridCells; i++)
  {
    cell.wind.x = 0.001f * i;
    cell.wind.y = 0.001f * i;
    cell.wind.z = 0.001f * i;
     hCells[i] = cell;
     hf[i] = i/10.f;
  }/*
  cudaMalloc((void**)&dCells, m_numGridCells*sizeof(Cell)); 
  cudaMemcpy(dCells,  hCells, m_numGridCells * sizeof(Cell), cudaMemcpyHostToDevice); *//*
  cudaMalloc((void**)&df, m_numGridCells*sizeof(float)); 
  cudaMalloc((void**)&gf, m_numGridCells*sizeof(float)); */
  cudaMalloc((void**)&dCells, m_numGridCells*sizeof(Cell)); 
//   cudaMalloc((void**)&gf, m_numGridCells*sizeof(float)); 
  cudaMemcpy(dCells,  hCells, m_numGridCells * sizeof(Cell), cudaMemcpyHostToDevice); 
  
  uint numThreads, numBlocks;
  computeGridSize(m_numGridCells, 256, numBlocks, numThreads);
  printf("%d, %d\n", numBlocks, numThreads);

    // execute the kernel
  testGCell<<< numBlocks, numThreads >>>(dCells, m_numGridCells);
//   cudaMemcpy(hCells,  dCells, m_numGridCells * sizeof(Cell), cudaMemcpyDeviceToHost); 
  cudaMemcpy(hCells,  dCells, m_numGridCells * sizeof(Cell), cudaMemcpyDeviceToHost); 
//   
  
//   for(uint i=0; i<m_numGridCells; i++) 
//   {
// //     printf("this is:%d x=%f, y=%f, z=%f\n", i, hCells[i].wind.x, hCells[i].wind.y, hCells[i].wind.z);      
//   }
  for(uint i=0; i<m_numGridCells; i++) 
  {
    printf("this is:%d x=%f, y=%f, z=%f\n", i, hCells[i].wind.x, hCells[i].wind.y, hCells[i].wind.z);    
//     printf("this is:%d x=%f, y=%f, z=%f\n", i, df[i] , df[i] , df[i] );          
//     printf("this is:%d x=%f, y=%f, z=%f\n", i, hf[i] , hf[i] , hf[i] );      
  }
   
}