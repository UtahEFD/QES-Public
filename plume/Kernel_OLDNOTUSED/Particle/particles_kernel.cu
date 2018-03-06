/*
 *particles_kernel.cu
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
/* 
 * CUDA particle kernel code.
 */

#ifndef _PARTICLES_KERNEL_CU_
#define _PARTICLES_KERNEL_CU_

#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "../../Domain/ConstParams.cuh" 
#include "../random.h"	
#include "particles_kernel.cuh"
 

// texture<float4, 3, cudaReadModeElementType> windTex;   
__constant__ ConstParams g_params;  
 

__device__ void resetParticles(float3 &pos)//resetParticles(float3 &pos, float3 &vel)
{  
  int thnum = __umul24(blockIdx.x,blockDim.x) + threadIdx.x; 
////generate random value based on threadID and drand48()
  unsigned int seed = tea<16>((pos.x *1000 /pos.z), thnum);  
  float3 jitter = make_float3(rnd(seed) ,  rnd(seed) ,  rnd(seed));
  
  if(g_params.source.type == SPHERESOURCE)
  {
    //sphere emitter// sphere sampling
    jitter = jitter*2.0f - 1.f;  //(-1.f, 1.f)
    jitter = jitter / sqrt(jitter.x*jitter.x + jitter.y*jitter.y + jitter.z*jitter.z) * g_params.source.info.sph.rad;   
    pos = jitter + g_params.source.info.sph.ori;
  }else if(g_params.source.type == LINESOURCE)
  {
    //pos = linestart + drand48() * (lineend - linestart)
    pos = (1.f - jitter.x) * g_params.source.info.ln.start + jitter.x * g_params.source.info.ln.end;
    pos.z += thnum%25;
//     
  }else if(g_params.source.type == POINTSOURCE)
  {
    //pos = linestart + drand48() * (lineend - linestart)
//     pos = (1.f - jitter.x) * g_params.source.info.ln.start + jitter.x * g_params.source.info.ln.end;
    
  }
//   float4 n =  tex3D(windTex, jitter.x, jitter.y, jitter.z);//tex3D(windTex, jitter.x*40, jitter.y*25, jitter.z*25);
//   pos = make_float3(n.x, n.y, n.z);
//   pos.x =  jitter.x/4.f-6.f;//- 3.f;//.1 *(u01(rng));
//   pos.y =  jitter.y/2.f;// + 2.f ;//- 0.f;//*u01(rng);
//   pos.z = jitter.z/4.f + 5.f;// 5.f;//jitter.z/2.f + 8.f; ;//- 3.f;//*u01(rng); */ 
//   vel = make_float3(0.f, 0.f, 0.f);
//   volatile float4 wind = tex3D(windTex, pos.x, pos.y, pos.z);
//   vel = make_float3(wind.x, wind.y, wind.z);//tex3D(windTex, pos.x, pos.y, pos.z);
//   pos = make_float3(cell4.x*8, cell4.y*50, cell4.z*50);//tex3D(windTex, pos.x, pos.y, pos.z);
//   unsigned int seed1 = tea<16>((pos.z*1000),thnum); 
//   float3 jitter1 = make_float3(rnd(seed),  rnd(seed),  rnd(seed));
//   vel.x = (jitter.z * 2.f)/100.f;//.01f *(u01(rng));
//   vel.y = (jitter.y * 2.f)/100.f;//.01f *(u01(rng));
//   vel.z = (jitter.x * 2.f - 1.0f )/100.f;//.01f *(u01(rng));
}


struct advect_functor
{
  float deltaTime; 
  __host__ __device__
  advect_functor(float delta_time) : deltaTime(delta_time){}//,domain(make_float3(29,1,29)) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t)
  {   
    volatile float4 posData = thrust::get<0>(t);
//     volatile float4 velData = thrust::get<1>(t);
    float3 pos = make_float3(posData.x, posData.y, posData.z);
//     float3 vel = make_float3(velData.x, velData.y, velData.z);

/*/////////////////////
reset the particles that are out of boundary       
//////////////////////*/
    if(  (pos.x > g_params.domain.x - g_params.particleRadius) 
      || (pos.x < g_params.origin.x + g_params.particleRadius)
      || (pos.y > g_params.domain.y - g_params.particleRadius) 
      || (pos.y < g_params.origin.y + g_params.particleRadius)  
      || (pos.z > g_params.domain.z - g_params.particleRadius) 
      || (pos.z < g_params.origin.z + g_params.particleRadius)) 
    { 
      resetParticles(pos); //resetParticles(pos, vel); 
    } 
    else 
// /*/////////////////////
// if particle hits buildings then it bounces    
// //////////////////////*/
    {  
//       if((pos.z > g_params.domain.z  - g_params.particleRadius))
//       { 
// 	float3 N = make_float3(0.f, 0.f, -1.f);
// 	vel = vel - (2 * dot(vel, N) *N); 
// // 	  vel.x += vel.x>0 ? -0.05f :0.05f;// + 0.05ff * jitter.x; 
// // 	  vel.y += vel.y>0 ? -0.05f :0.05f;// -vel.y + 0.05ff * jitter.x; 
// // 	  vel.z += vel.z>0 ? -0.05f :0.05f;// -vel.z + 0.05ff * jitter.x; 
// 	//vel /= 1.3f; 
//       }
//       else if((pos.z < -g_params.domain.z  + g_params.particleRadius))
//       {
// 	float3 N = make_float3(0.f, 0.f, 1.f);
// 	vel = (vel - (2 * dot(vel, N) *N));//1.3f;  
//       }
//       if((pos.x > g_params.building.lowCorner.x - g_params.particleRadius) 
// 	  && (pos.x <  g_params.building.highCorner.x + g_params.particleRadius)
// 	  && (pos.y >  g_params.building.lowCorner.y - g_params.particleRadius)
// 	  && (pos.y < g_params.building.highCorner.y  + g_params.particleRadius)   
// 	  && (pos.z >  g_params.building.lowCorner.z - g_params.particleRadius) 
// 	  && (pos.z < g_params.building.highCorner.z + g_params.particleRadius))
//       {  
// 	float3 N = make_float3(0.f, 0.f, 0.f);
// 	if(pos.x < g_params.building.lowCorner.x)
// 	  N = make_float3(-1.f, 0.f, 0.f);
// 	if(pos.x > g_params.building.highCorner.x)
// 	  N = make_float3(1.f, 0.f, 0.f);
// 	else if(pos.y > g_params.building.highCorner.y)
// 	  N = make_float3(0.f, 1.f, 0.f);
// 	else if(pos.y < g_params.building.lowCorner.y)
// 	  N = make_float3(0.f, -1.f, 0.f);
// 	else if(pos.z > g_params.building.highCorner.z)
// 	  N = make_float3(0.f, 0.f, 1.f);
// 	else if(pos.z < g_params.building.lowCorner.z)
// 	  N = make_float3(0.f, 0.f, -1.f);
// 	vel = vel - (2 * dot(vel, N) *N); 
// 	// vel = vel / 1.1f; 
// //   int thnum = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
// //   unsigned int seed = tea<16>((pos.x*1000/pos.z), thnum); 
// //   float3 jitter = make_float3(rnd(seed),  rnd(seed),  rnd(seed));
// 	vel.x += vel.x>0 ? -0.05f :0.05f;// + 0.05ff * jitter.x; 
// 	vel.y += vel.y>0 ? -0.05f :0.05f;// -vel.y + 0.05ff * jitter.x; 
// 	vel.z += vel.z>0 ? -0.05f :0.05f;// -vel.z + 0.05ff * jitter.x; 
//       }
//       else //{
//       {
// 	int thnum = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
// 	unsigned int seed = tea<16>((pos.y*1000), thnum+200);  
// 	float3 jitter = make_float3((rnd(seed)/2.f)/4000.f, (rnd(seed)/2.f -.5f)/4000.f, (rnd(seed)/2.f -.5f)/4000.f);
// 	vel += jitter;
// 	vel += g_params.gravity * deltaTime;
// 	vel *= g_params.globalDamping;
//       }

////////////////////////////////////////////////////////
// new position = old position + velocity * deltaTime 
//////////////////////////////////////////////////////

////read winddata from texture memory
	volatile float4 wind = tex3D(windTex, pos.x, pos.y, pos.z);
	float3 vel = make_float3(wind.x, wind.y, wind.z);// = tex3D(windTex, pos.x, pos.y, pos.z);

////generate random value based on threadID and drand48()
	int thnum = __umul24(blockIdx.x,blockDim.x) + threadIdx.x; 
	unsigned int seed = tea<16>((pos.x*1000/pos.z), thnum); 
	float3 jitter = make_float3(rnd(seed),  rnd(seed),  rnd(seed));
	jitter = jitter*0.1f;
  // 	vel.x += vel.x>0 ? -0.05f :0.05f;// + 0.05ff * jitter.x; 
  // 	vel.y += vel.y>0 ? -0.05f :0.05f;// -vel.y + 0.05ff * jitter.x; 
  // 	vel.z += vel.z>0 ? -0.05f :0.05f;// -vel.z + 0.05ff * jitter.x; 
	pos += (vel+jitter) * deltaTime/10.f;
    } 

    // store new position and velocity
    thrust::get<0>(t) = make_float4(pos, posData.w);
/////////////dont need to store vel any more
/////////////(read from files to texture memory)
//     thrust::get<1>(t) = make_float4(vel, velData.w); 
  }
};
 
//debugging the winddata stored in texture memory
__global__
void getCell(float4* output)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
/////////hard coding here, needs to fix
    if (index >= 40*25*26) return;
    
//     volatile Cell cell = pos[index]; 
    output[index] = tex3D(windTex, threadIdx.x%40, threadIdx.x%25, threadIdx.x%25);
}
//   
  
#endif
