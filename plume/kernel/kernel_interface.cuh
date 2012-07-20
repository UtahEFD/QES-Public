/*
 * kernel_interface.cuh
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
 

 extern "C"
{  
void cudaInit(int argc, char **argv);  
// void bindTexture(cudaArray *cellArray, cudaChannelFormatDesc channelDesc, CellTextureType textureType);
void bindTexture(cudaArray *&cellArray, const cudaChannelFormatDesc& channelDesc, const char* texname);
 
//   template<typename T>
// void createCellTexture(float w, float h, float d, 
// 		       thrust::host_vector<T> windData,
// 		       texture<T, 3, cudaReadModeElementType> cellTex); 
// void createCellTexture(int w, int h, int d, float4* &cellData); 

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


void setParameters(ConstParams *hostParams);
// void setDeviceCells(const Cell* hCells, int count);
void freeDeviceCells();
//debugging, check for each cell info;
uint iDivUp(uint a, uint b);
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads);
void dumpCells(int count);//(Cell* hCells, int count);

//   void copyDeviceData(uint size);
void compareHstDev(const thrust::host_vector<float4> &hData, const uint &size, const int &texname);
void randDevTest(thrust::host_vector<float4> &hData, const uint &size);
void initialWind(uint size);

void copyTurbsToDevice(const thrust::host_vector<turbulence> &hData);

////kernels/////////////////////////////////////////
void global_kernel(float *pos, float *winP, bool* seeds_flag, const uint &numParticles,
		   turbulence* d_turbs_ptr);
void global_kernel_debug(float *pos, float *winP, bool* seeds_flag, const uint &numParticles,
		   const thrust::host_vector<turbulence> &hData);
void advectPar_with_textureMemory(float *pos, float *winP, bool* seed_flag,
		     uint* concens, float deltaTime,  uint numParticles); 
void cal_concentration(float *pos, uint* concens, uint numParticles);
// 		     float deltaTime,  uint numParticles); 

void calcHash(int    numParticles);
  
}  