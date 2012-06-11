/*
* kernel_interface.cu
* This file is part of CUDAPLUME
*
* Copyright (C) 2012 - Alex Geng
*
* CUDAPLUME is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* CUDAPLUME is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with CUDAPLUME. If not, see <http://www.gnu.org/licenses/>.
*/

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h 
#include <cstdlib>
#include <cstdio>
#include <string.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cuda_gl_interop.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "Particle/particles_kernel.cu"

extern "C"
{

cudaArray *cellArray;
  // create 3D texture for cells
void createCellTexture(int w, int h, int d, float4* &cellData)
{
    cudaExtent size = make_cudaExtent(w, h, d); 
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cutilSafeCall( cudaMalloc3DArray(&cellArray, &channelDesc, size) );

    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)cellData, size.width*sizeof(float4), size.width, size.height);
    copyParams.dstArray = cellArray;
    copyParams.extent   = size;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cutilSafeCall( cudaMemcpy3D(&copyParams) );

    free(cellData);

    // set texture parameters
    cellTex.normalized = false;                      // access with nonnormalized texture coordinates 
    cellTex.filterMode = cudaFilterModePoint;      // 
    cellTex.addressMode[0] = cudaAddressModeClamp;   // Clamp texture coordinates
    cellTex.addressMode[1] = cudaAddressModeClamp;
    cellTex.addressMode[2] = cudaAddressModeClamp;

    // bind array to 3D texture
    cutilSafeCall(cudaBindTextureToArray(cellTex, cellArray, channelDesc)); 
}

  
void cudaInit(int argc, char **argv)
{   
  int devID;
  // use command-line specified CUDA device, otherwise use device with highest Gflops/s
  if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
      devID = cutilDeviceInit(argc, argv);
      if (devID < 0) {
	  printf("No CUDA Capable devices found, exiting...\n"); 
      }
  } else {
      devID = cutGetMaxGflopsDeviceId();
      cudaSetDevice( devID );
  }
}

void cudaGLInit(int argc, char **argv)
{   
  // use command-line specified CUDA device, otherwise use device with highest Gflops/s
//   if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
//       cutilDeviceInit(argc, argv);
//       printf("123123\n");
//   } else {
      cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
//   }
}

void allocateArray(void **devPtr, size_t size)
{
  cutilSafeCall(cudaMalloc(devPtr, size));
}

void freeArray(void *devPtr)
{
  cutilSafeCall(cudaFree(devPtr));
}

void threadSync()
{
  cutilSafeCall(cutilDeviceSynchronize());
}

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
  cutilSafeCall(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}

void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
{
  cutilSafeCall(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, 
					      cudaGraphicsMapFlagsNone));
}

void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
  cutilSafeCall(cudaGraphicsUnregisterResource(cuda_vbo_resource));	
}

void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
  void *ptr;
  cutilSafeCall(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
  size_t num_bytes; 
  cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,  
						      *cuda_vbo_resource));
  return ptr;
}

void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
  cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

void copyArrayFromDevice(void* host, const void* device, 
			struct cudaGraphicsResource **cuda_vbo_resource, int size)
{   
  if (cuda_vbo_resource)
      device = mapGLBufferObject(cuda_vbo_resource);

  cutilSafeCall(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
  
  if (cuda_vbo_resource)
      unmapGLBufferObject(*cuda_vbo_resource);
}

void setParameters(ConstParams *hostParams)
{
  // copy parameters to constant memory
  cutilSafeCall( cudaMemcpyToSymbol(g_params, hostParams, sizeof(ConstParams)) );
}

// void setDeviceCells(const Cell* hCells, int count)
// {
// // //   copy Cells to device
// //   if(g_dCells)
// //     freeArray(g_dCells);
// //   allocateArray((void**)&g_dCells, count * sizeof(Cell));  
// //   cutilSafeCall( cudaMemcpy(g_dCells, hCells, count * sizeof(Cell), cudaMemcpyHostToDevice) );
// }

// void freeDeviceCells()
// { 
//   // copy Cells to device
//   if(g_dCells)
//     freeArray(g_dCells);
// }

// compute grid and thread block size for a given number of elements
//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
  numThreads = min(blockSize, n);
  numBlocks = iDivUp(n, numThreads);
}

//debugging, check for each cell info;
void dumpCells( int count)
{  
  int total = 40*25*26;
  float4 *hCell = (float4 *)malloc(total*sizeof(float4));
  float4 *dCell;
  allocateArray((void**)&dCell, total*sizeof(float4));  
  
  uint numThreads, numBlocks;
  computeGridSize(total, 256, numBlocks, numThreads);
  getCell<<< numBlocks, numThreads >>>(dCell);
  cutilSafeCall(cudaMemcpy(hCell, dCell, total * sizeof(float4), cudaMemcpyDeviceToHost));  
  for(uint i=0; i<total; i++) 
  {
    printf("this is:%d x=%f, y=%f, z=%f\n", i, hCell[i].x, hCell[i].y, hCell[i].z);      
  }
}


void integrateSystem(float *pos,
		    float *vel,
		    float deltaTime,
		    uint numParticles)
{
  thrust::device_ptr<float4> d_pos4((float4 *)pos);
//   thrust::device_ptr<float4> d_vel4((float4 *)vel);

  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(d_pos4)),//(d_pos4, d_vel4)),
      thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles)),//(d_pos4+numParticles, d_vel4+numParticles)),
      advect_functor(deltaTime));
}
 






}   // extern "C"
