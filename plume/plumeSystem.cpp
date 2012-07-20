/*
 * particleSystem.cpp
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

#include "plumeSystem.h" 
// #include "Kernel/Texture_knl.cu"  

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>  

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>



#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

PlumeSystem::PlumeSystem(uint numParticles, bool bUseOpenGL, bool bUseGlobal, Building building, 
			 uint3 domain, float3 origin, Source source) :
    m_bInitialized(false), m_bUseOpenGL(bUseOpenGL),m_bUseGlobal(bUseGlobal),
    m_bGlobal_set(false), m_bTexture_set(false),
    m_numParticles(numParticles),
    m_hPos(0),m_hWinP(0), m_hSeeds(0), m_dPos(0), m_dWinP(0), m_dSeeds(0),//m_hCells(0),
//     m_gridSize(gridSize),
    m_timer(0) 
{
  m_numGridCells = domain.x * domain.y * domain.z;//m_gridSize.x*m_gridSize.y*m_gridSize.z; 
//     m_params.isFirsttime = true;

  m_params.building = building;//.lowCorner = make_float3(1.f, -1.f, 1.f); 
  m_params.particleRadius = 4.0f / 64.0f ;//particleRadius 
  m_params.domain = domain;//make_float3(40.f, 25.f, 25.f); 
  m_params.source = source;
  _initialize(); 
} 
 
  // create 3D texture for cells
template<class T>
// void createCellTexture(float w, float h, float d, T* &cellData, CellTextureType textureType)
// void createCellTexture(float w, float h, float d, thrust::host_vector<T> windData, CellTextureType textureType) 
// void PlumeSystem::createCellTexture(thrust::host_vector<T> &cellData, CellTextureType textureType) 
void PlumeSystem::createCellTexture(thrust::host_vector<T> &cellData, const char* texname) 
{
  cudaArray *cellArray; 
  int w = m_params.domain.x, h = m_params.domain.y, d = m_params.domain.z;
  cudaExtent size = make_cudaExtent(w, h, d);  
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
  cutilSafeCall( cudaMalloc3DArray(&cellArray, &channelDesc, size) );

  cudaMemcpy3DParms copyParams = { 0 };
  T *cellDatapt =  thrust::raw_pointer_cast(&cellData[0]);
  copyParams.srcPtr   = make_cudaPitchedPtr((void*)(cellDatapt), size.width*sizeof(T), size.width, size.height);
//   copyParams.srcPtr   = make_cudaPitchedPtr((void*)windData, size.width*sizeof(T), size.width, size.height);
  copyParams.dstArray = cellArray;
  copyParams.extent   = size;
  copyParams.kind     = cudaMemcpyHostToDevice;
  cutilSafeCall( cudaMemcpy3D(&copyParams) );

  cellData.clear();
  
  const textureReference* texPt=NULL; 
  cudaGetTextureReference(&texPt, texname);  
  ((textureReference *)texPt)->normalized = false;                      // access with nonnormalized texture coordinates 
  ((textureReference *)texPt)->filterMode = cudaFilterModePoint;      // 
  ((textureReference *)texPt)->addressMode[0] = cudaAddressModeClamp;   // Clamp texture coordinates
  ((textureReference *)texPt)->addressMode[1] = cudaAddressModeClamp;
  ((textureReference *)texPt)->addressMode[2] = cudaAddressModeClamp;
  cutilSafeCall(cudaBindTextureToArray(texPt, cellArray, &channelDesc)); 
   
 } //*/

PlumeSystem::~PlumeSystem()
{cudaArray *cellArray;
    _finalize();
    m_numParticles = 0;
}

uint
PlumeSystem::createVBO(uint size)
{
  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  return vbo;
}

// inline float lerp(float a, float b, float t)
// {
//   return a + t*(b-a);
// }

// create a color ramp
void colorRamp(float t, float *r)
{
  const int ncolors = 7;
  float c[ncolors][3] = {
      { 1.0, 0.0, 0.0, },
      { 1.0, 0.5, 0.0, },
      { 1.0, 1.0, 0.0, },
      { 0.0, 1.0, 0.0, },
      { 0.0, 1.0, 1.0, },
      { 0.0, 0.0, 1.0, },
      { 1.0, 0.0, 1.0, },
  };
  t = t * (ncolors-1);
  int i = (int) t;
  float u = t - floor(t);
  r[0] = lerp(c[i][0], c[i+1][0], u);
  r[1] = lerp(c[i][1], c[i+1][1], u);
  r[2] = lerp(c[i][2], c[i+1][2], u);
}

void PlumeSystem::_initialize()
{
  assert(!m_bInitialized);  
  assert(m_numParticles>0);  
  assert(m_numGridCells>0);  
  // allocate host storage
  m_hPos = new float[m_numParticles*4];
  m_hWinP = new float[m_numParticles*4];
  m_hSeeds = new bool[m_numParticles];
  m_hConcetration = new uint[m_numGridCells];
  memset(m_hPos, 0, m_numParticles*4*sizeof(float));//16384
  memset(m_hWinP, 0, m_numParticles*4*sizeof(float));//16384  

  for(int i=0; i<m_numParticles*4; i=i+4)
  {
    m_hWinP[i] = 0;
    m_hWinP[i+1] = 0;
    m_hWinP[i+2] = 0;
    m_hWinP[i+3] = 0;
  }
  // allocate GPU data
  unsigned int memSize = sizeof(float) * 4 * m_numParticles;

  if (m_bUseOpenGL) 
  {
    m_posVbo = createVBO(memSize);    // vertex buffer object for particle positions
    registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);// handles OpenGL-CUDA exchange
  } else {
    allocateArray((void**)&m_dPos, memSize); 
  }
//   memSize = sizeof(float) * 4 * m_numParticles;
  allocateArray((void**)&m_dWinP, memSize); 
  
  memSize = sizeof(uint) * m_numGridCells;
  allocateArray((void**)&m_dConcetration, memSize); 
  
  memSize = sizeof(bool) * m_numParticles;
  allocateArray((void**)&m_dSeeds, memSize); 
//   allocateArray((void**)&m_dWinP, memSize); 

  if(m_bUseGlobal) 
  { 
    memSize = sizeof(turbulence) * m_numGridCells;
    allocateArray((void**)&m_dGlobal_turbData, memSize); 
  }
  

  if (m_bUseOpenGL) 
  {
    m_colorVBO = createVBO(m_numParticles*4*sizeof(float));//16384
    registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

    // fill color buffer
    glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
    float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    float *ptr = data;
    for(uint i=0; i<m_numParticles; i++) 
    {//16384
      float t = i / (float) m_numParticles;//16384
    #if 1
	*ptr++ = drand48();//rand() / (float) RAND_MAX;
	*ptr++ = drand48();//rand() / (float) RAND_MAX;
	*ptr++ = drand48();//rand() / (float) RAND_MAX;
    #else
	colorRamp(t, ptr);
	ptr+=3;//////////////////////////???????????????????
    #endif
	*ptr++ = 1.0f;
      }
    glUnmapBufferARB(GL_ARRAY_BUFFER);
  } else 
  {
//     cutilSafeCall( cudaMalloc( (void **)&m_cudaColorVBO, sizeof(float)*numParticles*4) );
  }

  cutilCheckError(cutCreateTimer(&m_timer));

  setParameters(&m_params);

  m_bInitialized = true;
}

void
PlumeSystem::_initDeviceTexture(
	     thrust::host_vector<float> &CoEps,
	     thrust::host_vector<int> &cellType,
	     thrust::host_vector<float4> &windData, 
	     thrust::host_vector<float4> &eigVal, 
	     thrust::host_vector<float4> &ka0, 
	     thrust::host_vector<float4> &g2nd, 
      ////////////////  matrix 9////////////////
	     thrust::host_vector<float4> &eigVec1,
	     thrust::host_vector<float4> &eigVec2,
	     thrust::host_vector<float4> &eigVec3,
	     thrust::host_vector<float4> &eigVecInv1,
	     thrust::host_vector<float4> &eigVecInv2,
	     thrust::host_vector<float4> &eigVecInv3,
	     thrust::host_vector<float4> &lam1,
	     thrust::host_vector<float4> &lam2,
	     thrust::host_vector<float4> &lam3,
      //////////////// matrix6 ////////////////
	     thrust::host_vector<float4> &sig1,
	     thrust::host_vector<float4> &sig2,
	     thrust::host_vector<float4> &taudx1,
	     thrust::host_vector<float4> &taudx2, 
	     thrust::host_vector<float4> &taudy1,
	     thrust::host_vector<float4> &taudy2, 
	     thrust::host_vector<float4> &taudz1,
	     thrust::host_vector<float4> &taudz2) 
{
  assert(!m_bTexture_set);  
//   bool isUseTexture = false;
  if (m_bUseOpenGL && !m_bUseGlobal) 
  {
    createCellTexture(CoEps, "CoEpsTex");
    createCellTexture(cellType, "cellTypeTex");
    createCellTexture(windData, "windFieldTex"); 
    createCellTexture(eigVal, "eigValTex"); 
    createCellTexture(ka0, "ka0Tex");  
    createCellTexture(g2nd, "g2ndTex"); 
    
    createCellTexture(eigVec1, "eigVec1Tex"); 
    createCellTexture(eigVec2, "eigVec2Tex"); 
    createCellTexture(eigVec3, "eigVec3Tex"); 
    createCellTexture(eigVecInv1, "eigVecInv1Tex"); 
    createCellTexture(eigVecInv2, "eigVecInv2Tex"); 
    createCellTexture(eigVecInv3, "eigVecInv3Tex"); 
    createCellTexture(lam1, "lam1Tex"); 
    createCellTexture(lam2, "lam2Tex"); 
    createCellTexture(lam3, "lam3Tex"); 
    //////////////// matrix6 ////////////////
    createCellTexture(sig1, "sig1Tex"); 
    createCellTexture(sig2, "sig2Tex"); 
    createCellTexture(taudx1, "taudx1Tex"); 
    createCellTexture(taudx2, "taudx2Tex"); 
    createCellTexture(taudy1, "taudy1Tex"); 
    createCellTexture(taudy2, "taudy2Tex"); 
    createCellTexture(taudz1, "taudz1Tex"); 
    createCellTexture(taudz2, "taudz2Tex"); 
    m_bTexture_set = true;
  }//else 
//   _initialize();
  
}

void PlumeSystem::_finalize()
{
  assert(m_bInitialized);

  delete [] m_hPos;
  delete [] m_hWinP;
//     delete [] m_hCells; 
  
//   freeArray(m_dVel); 
  
  if (m_bUseOpenGL) {
    unregisterGLBufferObject(m_cuda_posvbo_resource);
    glDeleteBuffers(1, (const GLuint*)&m_posVbo);
    glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
  } else {
//     cutilSafeCall( cudaFree(m_cudaPosVBO) );
//     cutilSafeCall( cudaFree(m_cudaColorVBO) );
  }
}

// step the simulation
void PlumeSystem::update(const float &deltaTime, bool &b_print_concentration)
{
  assert(m_bInitialized);
  if(m_bUseGlobal)
    assert(m_bGlobal_set);
  
  static uint numeject=1000;
  float *dPos;

  if(m_bUseOpenGL)
  {
    dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
  } else {
//     dPos = (float *) m_cudaPosVBO;
  }

  // update constants
  static bool isFirstime = false; 
  if(!isFirstime)
  {
    isFirstime= true;
    setParameters(&m_params); 
  }  
  
  // integrate
  if(m_bUseOpenGL && !m_bUseGlobal)  
    advectPar_with_textureMemory(dPos, m_dWinP, m_dSeeds, m_dConcetration, 
				 deltaTime, numeject);
  else if(m_bUseOpenGL && m_bUseGlobal) 
  { 
    global_kernel(dPos, m_dWinP, m_dSeeds, numeject, m_dGlobal_turbData); 
  }
  else if(!m_bUseOpenGL && m_bUseGlobal) 
  { 
    global_kernel(m_dPos, m_dWinP, m_dSeeds, numeject, m_dGlobal_turbData); 
  } 
  
  if(b_print_concentration)
  {
    cal_concentration(dPos, m_dConcetration, numeject);   
    b_print_concentration = false;
  }
    
  if(numeject < m_numParticles)
  {
    numeject += 100;//speed of emitting particles  
  }

  // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
  if (m_bUseOpenGL) {
    unmapGLBufferObject(m_cuda_posvbo_resource);
  }
}

void PlumeSystem::dumpGrid()
{
//   dumpCells(m_hCells, 100);// its on Kernel/particleSystem.cu 
//   dumpCells( 100);// its on Kernel/particleSystem.cu 
}

void PlumeSystem::dumpParticles(uint start, uint count)
{
    // debug
   copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float)*4*count); 
   copyArrayFromDevice(m_hWinP, m_dWinP, 0, sizeof(float)*4*count);

  for(uint i=start; i<start+count; i++) { 
    printf("pos %d: (%f, %f, %f, %f)\n", i, m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]); 
    printf("prime %d: (%f, %f, %f, %f)\n", i, m_hWinP[i*4+0], m_hWinP[i*4+1], m_hWinP[i*4+2], m_hWinP[i*4+3]); 
  }
}

float* PlumeSystem::getArray(DeviceArray array)
{
  assert(m_bInitialized);

  float* hdata = 0;
  float* ddata = 0;
  struct cudaGraphicsResource *cuda_vbo_resource = 0;

  switch (array)
  {
  default:
  case POSITION:
      hdata = m_hPos;
      ddata = m_dPos;
      cuda_vbo_resource = m_cuda_posvbo_resource;
//   copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
      break;
  case WINDPRIME:
      hdata = m_hWinP;
      ddata = m_dWinP;
//   copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numGridCells*4*sizeof(float));
      break;
  }

  copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
  return hdata;
}

void PlumeSystem::setArray(DeviceArray array, const float* data, int start, int count)
{
  assert(m_bInitialized);

  switch (array)
  {
  default:
  case POSITION:
    {
      if (m_bUseOpenGL) {
	unregisterGLBufferObject(m_cuda_posvbo_resource);
	glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
	glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
      } else
	 copyArrayToDevice(m_dPos, data, start*4*sizeof(float), 4 * m_numParticles*sizeof(float)); 
    }
    break;
  case WINDPRIME:
    copyArrayToDevice(m_dWinP, data, start*3*sizeof(float), 3 * m_numParticles*sizeof(float));
    break;  
  }       
}  

void PlumeSystem::reset(const float3 &pos, thrust::host_vector<float3> &prime)//ParticleSystem::(ParticleConfig config)
{ 
  int p = 0, v = 0;
  for(uint i=0; i < m_numParticles; i++) 
  {  
    m_hPos[p++] = pos.x;// (.1*(frand()/2.f) - 2.5f);
    m_hPos[p++] = pos.y;//  (.0f+ .1*frand() - 1.5f);
    m_hPos[p++] = pos.z;//  2 * (0.2f+ .1*frand() - 0.5f);
    m_hPos[p++] = i;//drand48();
//     m_hPos[p++] = 1000000.0f; // radius 
  } 	
  
  for(uint i=0; i < m_numParticles; i++) 
  {  
    m_hSeeds[i] = false;
  } 	
  
  for(uint i=0; i < m_numGridCells; i++) 
  {  
    m_hConcetration[i] = 0;
//     m_hConcetration[i] = i;
  } 	
  for(int i=0; i<m_numParticles*4; i=i+4)
  {
    m_hWinP[i] = prime[i].x;
    m_hWinP[i+1] = prime[i].y;
    m_hWinP[i+2] = prime[i].z;
    m_hWinP[i+3] = 0;
  }
  
 //copy data from host to device  
  setArray(POSITION, m_hPos, 0, m_numParticles);
  setArray(WINDPRIME, m_hWinP, 0, m_numParticles);
  
  copyArrayToDevice(m_dSeeds, m_hSeeds, 0, m_numParticles*sizeof(bool));
  copyArrayToDevice(m_dConcetration, m_hConcetration, 0, m_numGridCells*sizeof(uint));  
}

void PlumeSystem::copy_turbs_2_deviceGlobal(const thrust::host_vector<turbulence> &turbData)//ParticleSystem::(ParticleConfig config)
{   
   if(m_bUseGlobal)  
  {
    m_bGlobal_set = true; 
    copyArrayToDevice(m_dGlobal_turbData, thrust::raw_pointer_cast(&turbData[0]),
		      0, m_numGridCells*sizeof(turbulence));  
  } 
}

void PlumeSystem::dev_par_concentration()
{ 
  cal_concentration(m_dPos, m_dConcetration, 2000);
}
 
