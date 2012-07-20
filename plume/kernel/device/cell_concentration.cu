/*
 * concentration.cu
 * This file is part of GPUPLUME
 *
 * Copyright (C) 2012 - Alex
 *
 * GPUPLUME is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * GPUPLUME is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GPUPLUME. If not, see <http://www.gnu.org/licenses/>.
 */

 #ifndef __CONCENTRATION_CU_H__
 #define __CONCENTRATION_CU_H__
 
__global__ void concentration_kernel
                (float4* posPtr, uint* device_cons) 
{ 
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  uint offset = x + y * blockDim.x * gridDim.x; 
  
  float3 posf3 = make_float3(posPtr[offset]);//make_float3(43.5,55,0.66);//
  uint cellIndex = (int)(posf3.z+1)*g_params.domain.x*g_params.domain.y + 
		   (int)(posf3.y)*g_params.domain.x + (int)posf3.x;
  if(cellIndex > g_params.domain.x*g_params.domain.y*g_params.domain.z)
    return;
  atomicAdd(&device_cons[cellIndex], 1);		  
}

 #endif /* __CONCENTRATION_CU_H__ */
 
