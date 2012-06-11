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
#include "Kernel/kernel_interface.cuh" 

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h

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

PlumeSystem::PlumeSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL, Building building,
				float3 domain, float3 origin, Source source, float4* &cellData) :
    m_bInitialized(false), m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
    m_hPos(0),m_hVel(0), m_dPos(0), m_dVel(0),//m_hCells(0),
    m_gridSize(gridSize),
    m_timer(0) 
{
  m_numGridCells = gridSize.x * gridSize.y * gridSize.z;//m_gridSize.x*m_gridSize.y*m_gridSize.z; 
//     m_params.isFirsttime = true;

  m_params.building = building;//.lowCorner = make_float3(1.f, -1.f, 1.f); 
  m_params.particleRadius = 4.0f / 64.0f ;//particleRadius 
  m_params.domain = domain;//make_float3(40.f, 25.f, 25.f);
//     m_params.origin = origin;//make_float3(.0f, .0f, .0f); 
//     m_params.sourceOrigin = sourceOrigin;//make_float3(10.0f, 0.5f, 12.5f); 
  m_params.source = source;

  _initialize(numParticles, cellData);
}
/*
PlumeSystem::setSource()
{
  if( g_params.source.type == SPHERESOURCE)
  {
    assert(g_params.source.type == SPHERESOURCE);
    source.info.sph.ori = sourceOrigin;
    source.info.sph.rad = .5f;
    source.speed = 0.5f;
  }
  if( g_params.source.type == LINESOURCE)
  {
    assert(g_params.source.type == LINESOURCE);
  }
  
}*/

PlumeSystem::~PlumeSystem()
{
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

inline float lerp(float a, float b, float t)
{
  return a + t*(b-a);
}

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

void
PlumeSystem::_initialize(int numParticles, float4* &cellData)
{
  assert(!m_bInitialized);
  
  createCellTexture(40, 25, 26, cellData);

  m_numParticles = numParticles;//16384

  // allocate host storage
  m_hPos = new float[m_numParticles*4];
//   m_hVel = new float[m_numParticles*4];
  memset(m_hPos, 0, m_numParticles*4*sizeof(float));//16384
//   memset(m_hVel, 0, m_numParticles*4*sizeof(float));//16384 

  // allocate GPU data
  unsigned int memSize = sizeof(float) * 4 * m_numParticles;

  if (m_bUseOpenGL) {
      m_posVbo = createVBO(memSize);    // vertex buffer object for particle positions
      registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);// handles OpenGL-CUDA exchange
  } else {
      cutilSafeCall( cudaMalloc( (void **)&m_cudaPosVBO, memSize )) ;// vertex buffer object for particle positions
  }

//   allocateArray((void**)&m_dVel, memSize); 


  if (m_bUseOpenGL) {
    m_colorVBO = createVBO(m_numParticles*4*sizeof(float));//16384
    registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

    // fill color buffer
    glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
    float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    float *ptr = data;
    for(uint i=0; i<m_numParticles; i++) {//16384
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
  } else {
    cutilSafeCall( cudaMalloc( (void **)&m_cudaColorVBO, sizeof(float)*numParticles*4) );
  }

  cutilCheckError(cutCreateTimer(&m_timer));

  setParameters(&m_params);

  m_bInitialized = true;
}

void
PlumeSystem::_finalize()
{
  assert(m_bInitialized);

  delete [] m_hPos;
//   delete [] m_hVel;
//     delete [] m_hCells; 
  
//   freeArray(m_dVel); 
  
  if (m_bUseOpenGL) {
    unregisterGLBufferObject(m_cuda_posvbo_resource);
    glDeleteBuffers(1, (const GLuint*)&m_posVbo);
    glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
  } else {
    cutilSafeCall( cudaFree(m_cudaPosVBO) );
    cutilSafeCall( cudaFree(m_cudaColorVBO) );
  }
}

// step the simulation
void 
PlumeSystem::update(float deltaTime)
{
  assert(m_bInitialized);
  static uint numeject=1;
  float *dPos;

  if (m_bUseOpenGL) {
    dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
  } else {
    dPos = (float *) m_cudaPosVBO;
  }

  // update constants
  static bool isFirstime = false;
  if(!isFirstime)
  {
    isFirstime= true;
    setParameters(&m_params); 
  } 
  
  // integrate
  integrateSystem(dPos, m_dVel, deltaTime, numeject);
      // m_numParticles-60); 
    
  if(numeject < m_numParticles)
  {
    numeject += 10;//speed of emitting particles 
  }

  // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
  if (m_bUseOpenGL) {
    unmapGLBufferObject(m_cuda_posvbo_resource);
  }
}

void
PlumeSystem::dumpGrid()
{
//   dumpCells(m_hCells, 100);// its on Kernel/particleSystem.cu 
  dumpCells( 100);// its on Kernel/particleSystem.cu 
}

void
PlumeSystem::dumpParticles(uint start, uint count)
{
    // debug
  copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float)*4*count);
//   copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float)*4*count);

  for(uint i=start; i<start+count; i++) {
//        printf("%d: ", i);
    if(m_hPos[i*4+0]==0.f)
      printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
      // printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
  }
}

float* 
PlumeSystem::getArray(DeviceArray array)
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
      break;
  case VELOCITY:
//       hdata = m_hVel;
//       ddata = m_dVel;
      break;
  }

  copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
  return hdata;
}

void
PlumeSystem::setArray(DeviceArray array, const float* data, int start, int count)
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
      }
    }
    break;
  case VELOCITY:
//     copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
    break; 
  }       
}  
void
PlumeSystem::reset()//ParticleSystem::(ParticleConfig config)
{ 
  int p = 0, v = 0;
  for(uint i=0; i < m_numParticles; i++) 
  {
    float point[3]; 
    m_hPos[p++] =  100000.f;// (.1*(frand()/2.f) - 2.5f);
    m_hPos[p++] =  100000.f;//  (.0f+ .1*frand() - 1.5f);
    m_hPos[p++] =  100000.f;//  2 * (0.2f+ .1*frand() - 0.5f);
    m_hPos[p++] = 1000000.0f; // radius 
  } 	
      
  setArray(POSITION, m_hPos, 0, m_numParticles);
//   setArray(VELOCITY, m_hVel, 0, m_numParticles);
//   setDeviceCells(m_hCells, m_numGridCells);
//   copyArrayToDevice(m_hCell, g_dCells, 0, m_numGridCells*4*sizeof(Cell)); 
}

 
