/*
 *particles_kernel.cuh
 * This file is part of CUDAPLUME
 *
 * Copyright (C) 2012 - Alex Geng
 *
 * CUDAPLUME is free softwareTex; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software FoundationTex; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * CUDAPLUME is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTYTex; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with CUDAPLUME. If not, see <http://www.gnu.org/licenses/>.
 */
/* 
 * CUDA particle kernel code.
 */

#ifndef _PARTICLES_KERNEL_CUH_
#define _PARTICLES_KERNEL_CUH_ 

#include "CellTextureType.cuh" 
  texture<float, 3, cudaReadModeElementType> CoEpsTex;
  texture<int, 3, cudaReadModeElementType> cellTypeTex;
  texture<float4, 3, cudaReadModeElementType> windFieldTex;   
  texture<float4, 3, cudaReadModeElementType> eigValTex; 
  texture<float4, 3, cudaReadModeElementType> ka0Tex; 
  texture<float4, 3, cudaReadModeElementType> g2ndTex; 
////////////////  matrix 9////////////////
  texture<float4, 3, cudaReadModeElementType> eigVec1Tex;
  texture<float4, 3, cudaReadModeElementType> eigVec2Tex;
  texture<float4, 3, cudaReadModeElementType> eigVec3Tex;
  texture<float4, 3, cudaReadModeElementType> eigVecInv1Tex;
  texture<float4, 3, cudaReadModeElementType> eigVecInv2Tex;
  texture<float4, 3, cudaReadModeElementType> eigVecInv3Tex;
  texture<float4, 3, cudaReadModeElementType> lam1Tex;
  texture<float4, 3, cudaReadModeElementType> lam2Tex;
  texture<float4, 3, cudaReadModeElementType> lam3Tex;
//////////////// matrix6 ////////////////
  texture<float4, 3, cudaReadModeElementType> sig1Tex;
  texture<float4, 3, cudaReadModeElementType> sig2Tex;
  texture<float4, 3, cudaReadModeElementType> taudx1Tex;
  texture<float4, 3, cudaReadModeElementType> taudx2Tex; 
  texture<float4, 3, cudaReadModeElementType> taudy1Tex;
  texture<float4, 3, cudaReadModeElementType> taudy2Tex; 
  texture<float4, 3, cudaReadModeElementType> taudz1Tex;
  texture<float4, 3, cudaReadModeElementType> taudz2Tex;

#endif
