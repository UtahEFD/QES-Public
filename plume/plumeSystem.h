/*
 * particleSystem.h
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

#ifndef __PLUMESYSTEM_H__
#define __PLUMESYSTEM_H__
 

// #include "vector_functions.h"
#include <thrust/host_vector.h>
#include "kernel/kernel_global/CellTextureType.cuh" 
#include "kernel/kernel_global/turbulence.cuh" 
#include "kernel/kernel_global/ConstParams.cuh" 
#include "kernel/kernel_interface.cuh"   

//set global memory of cells 
// extern Cell* g_dCells; 
// Plume system class
class PlumeSystem
{
public:
    PlumeSystem(uint numParticles, bool bUseOpenGL, bool bUseGlobal, Building building,
		uint3 domain, float3 origin, Source source
// 		float4* &cellData);
// 		thrust::host_vector<float4> cellData);
		);
    ~PlumeSystem();
    
    enum DeviceArray
    {
        POSITION,
        WINDPRIME, 
        SEEDS
    };

//     void copy_turbs_2_device(const thrust::host_vector<turbulence> &turbData);
    void update(const float &deltaTime, bool &b_print_concentration); 
    void reset(const float3 &pos, thrust::host_vector<float3> &prime);

    float* getArray(DeviceArray array);
    void   setArray(DeviceArray array, const float* data, int start, int count);
//     void   setCell(const Cell* cellData, int start, int count);

    int    getNumParticles() const { return m_numParticles; }

    unsigned int getCurrentReadBuffer() const { return m_posVbo; }
    unsigned int getColorBuffer()       const { return m_colorVBO; }

    void * getCudaPosVBO()   const { return (void *)m_cudaPosVBO; }
    void * getCudaColorVBO() const { return (void *)m_cudaColorVBO; }

    void dumpGrid();
    void dumpParticles(uint start, uint count);

//     void setIterations(int i) { m_solverIterations = i; }
    
    
    void copy_turbs_2_deviceGlobal(const thrust::host_vector<turbulence> &turbData);
    void dev_par_concentration();
//     void setDamping(float x) { m_params.globalDamping = x; }
//     void setGravity(float x) { m_params.gravity = make_float3(0.00001f, x*.1f, 0.00015f); } 

    float getParticleRadius() { return m_params.particleRadius; } 
    /*
    template<class T> 
    void createCellTexture(cudaArray *cellArray, thrust::host_vector<T> &cellData, CellTextureType textureType);*/
//     template<class T> 
//     void createCellTexture(thrust::host_vector<T> &cellData, CellTextureType textureType);
    template<class T> 
    void createCellTexture(thrust::host_vector<T> &cellData, const char* texname);
    void _initDeviceTexture(
	     thrust::host_vector<float> &CoEps,
	     thrust::host_vector<int> &cellType,
	     thrust::host_vector<float4> &cellData, 
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
	     thrust::host_vector<float4> &taudz2) ;
    void _initialize();

protected: // methods
    PlumeSystem() {}
    uint createVBO(uint size);

//     void _initialize(int numParticles, float4* &cellData);
    void _finalize();

    void initGrid(uint *size, float spacing, float jitter, uint numParticles);

protected: // data
    bool m_bInitialized, m_bUseOpenGL, m_bUseGlobal, m_bGlobal_set, m_bTexture_set;
    uint m_numParticles;

    // CPU data
    float* m_hPos;              // particle positions
    float* m_hWinP;              // particle velocities
    uint * m_hConcetration;
    bool* m_hSeeds;              // particle velocities
//     Cell*  m_hCells;
 

    // GPU data
    float* m_dPos;
    float* m_dWinP;
    uint * m_dConcetration;
    turbulence* m_dGlobal_turbData;
    bool*  m_dSeeds;              // particle velocities
    uint   m_posVbo;            // vertex buffer object for particle positions
    uint   m_colorVBO;          // vertex buffer object for colors
//     Cell*  m_dCell; //we use g_dCell instead
    
    //if not using OPENGL
    float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
    float *m_cudaColorVBO;      // these are the CUDA deviceMem Color 
//     Cell*  m_cudaCell;//

    struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *m_cuda_Cell_resource; // handles OpenGL-CUDA exchange

    // params
    ConstParams m_params;
//     uint3 m_gridSize;
    uint m_numGridCells;

    uint m_timer;

//     uint m_solverIterations;
 

};

#endif // __PLUMESYSTEM_H__
