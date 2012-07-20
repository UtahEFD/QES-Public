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
#include "texture_mem.cuh" 
#include "reflection.cu"
    
std::ofstream random_file;
__constant__ ConstParams g_params;  
// thrust::device_vector<float4> devVec;
__constant__ int MAXCOUNT = 10;//which could be moved to g_params(const value) 
__constant__ float terFacU = 2.5;
__constant__ float terFacV = 2.;
__constant__ float terFacW = 2.;

__device__ void resetParticles(float3 &pos, uint &seed)//resetParticles(float3 &pos, float3 &vel)
{  
//   int thnum = __umul24(blockIdx.x,blockDim.x) + threadIdx.x; 
// //--------------generate random value based on threadID and drand48()
//   unsigned int seed = tea<16>((pos.x *1000 /pos.z), thnum);  
  float3 jitter = make_float3(0,0,0);//rnd(seed) ,  rnd(seed) ,  rnd(seed));
  
  if(g_params.source.type == SPHERESOURCE)
  {
//----------sphere emitter// sphere sampling
    jitter = jitter*2.0f - 1.f;  //(-1.f, 1.f)
    jitter = jitter / sqrtf(jitter.x*jitter.x + jitter.y*jitter.y + jitter.z*jitter.z) * g_params.source.info.sph.rad;   
    pos = jitter + g_params.source.info.sph.ori;
  }else if(g_params.source.type == LINESOURCE)
  {
    //pos = linestart + drand48() * (lineend - linestart)
    pos = (1.f - jitter.x) * g_params.source.info.ln.start + jitter.x * g_params.source.info.ln.end;
//     pos.z += thnum%11;
    pos.z += seed%11;
//     
  }else if(g_params.source.type == POINTSOURCE)
  {
    pos = g_params.source.info.sph.ori; 
  }
}

// __device__ void readTex(float3 wind, float3 ka0, float3 g2nd, float3 lam1, float3 lam2, float3 lam3,)

struct advect_functor
{
  float deltaTime; 
  __host__ __device__
  advect_functor(float delta_time) : deltaTime(delta_time){}//,domain(make_float3(29,1,29)) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t)
  {    
    float3 pos = make_float3(thrust::get<0>(t)); 
    float3 windPrime = make_float3(thrust::get<1>(t)); 
//     unsigned int seed = thrust::get<2>(t); 
    unsigned int seed = tea<16>(__umul24(blockIdx.x,blockDim.x) + threadIdx.x, thrust::get<2>(t));  
    float4 debugf4 = thrust::get<3>(t); 
    int countPrmMax=10;//1000;
//--------------reset the particles that are out of boundary        
    debugf4 = make_float4(pos, thrust::get<2>(t));;
//     if(!isANum(pos))
//     {
//       resetParticles(pos);
//     }else 
    if(  (pos.x > g_params.domain.x-1 - g_params.particleRadius) 
      || (pos.x < g_params.origin.x + g_params.particleRadius)
      || (pos.y > g_params.domain.y-1 - g_params.particleRadius) 
      || (pos.y < g_params.origin.y + g_params.particleRadius)  
      || (pos.z+1 > g_params.domain.z - g_params.particleRadius) 
      || (pos.z < g_params.origin.z-1 + g_params.particleRadius)) 
    {    
// 	debugf4 = make_float4(pos, seed);
      resetParticles(pos, seed); //resetParticles(pos, vel);          
    } 
    else 
//------ if particle hits buildings then it bounces     
    {  
#if 0
      volatile int cellType = tex3D(cellTypeTex, pos.x, pos.y, pos.z+1);
      if(cellType == 0)
      { 
	debugf4 = make_float4(pos, 1.01f);
      thrust::get<3>(t) = debugf4; 
	return ;
      }
/////////////////calculate dt
      float tStepInp = 1.f;//read from input file
      float tStepRem = tStepInp;
      float dt = tStepRem;
//       float tStepMin = 1.0f; 
//       int count = 0;
//       int countprimeloop = 0;
      
	volatile float CoEps = tex3D(CoEpsTex, pos.x, pos.y, pos.z+1);
	volatile float3 eigVal = make_float3(tex3D(eigValTex, pos.x, pos.y, pos.z+1));
	float tFac = 0.5f;
// 	volatile float3 sig222333 = make_float3(tex3D(sig2Tex, pos.x, pos.y, pos.z));
	volatile float3 sig = make_float3(tex3D(sig1Tex, pos.x, pos.y, pos.z+1).x,
					  tex3D(sig2Tex, pos.x, pos.y, pos.z+1).x,  
					  tex3D(sig2Tex, pos.x, pos.y, pos.z+1).z
	                                 );
	
	float tStepSigW = 2.f * sig.z * sig.z / CoEps;
	
	float tStepCal = tFac * minf4(fabs(-1.f/eigVal), tStepSigW);
	 dt = minf4(tStepInp, tStepCal, tStepRem,dt);
	//dt = 1.f;
	
// 	
	volatile float3 wind = make_float3(tex3D(windFieldTex, pos.x, pos.y, pos.z+1));
	volatile float3 ka0  = make_float3(tex3D(ka0Tex, pos.x, pos.y, pos.z+1));
	         float3 g2nd = make_float3(tex3D(g2ndTex, pos.x, pos.y, pos.z+1));
		 
	volatile float3 lam1 = make_float3(tex3D(lam1Tex, pos.x, pos.y, pos.z+1));
	volatile float3 lam2 = make_float3(tex3D(lam2Tex, pos.x, pos.y, pos.z+1));
	volatile float3 lam3 = make_float3(tex3D(lam3Tex, pos.x, pos.y, pos.z+1));
	
	volatile float3 taudx1 = make_float3(tex3D(taudx1Tex, pos.x, pos.y, pos.z+1)); 
	volatile float3 taudx2 = make_float3(tex3D(taudx2Tex, pos.x, pos.y, pos.z+1));  
	volatile float3 taudy1 = make_float3(tex3D(taudy1Tex, pos.x, pos.y, pos.z+1));
	volatile float3 taudy2 = make_float3(tex3D(taudy2Tex, pos.x, pos.y, pos.z+1)); 
	volatile float3 taudz1 = make_float3(tex3D(taudz1Tex, pos.x, pos.y, pos.z+1));
	volatile float3 taudz2 = make_float3(tex3D(taudz2Tex, pos.x, pos.y, pos.z+1)); 
        
      bool is_first_NAN = false;
      bool isDone = false;
      while(!isDone && cellType!=0){
    //--------------windPrimeRotation = eigenvectorInverseMatrix * windPrime
	float3 windPrimeRot = make_matrix3X3(
			    make_float3(tex3D(eigVecInv1Tex, pos.x, pos.y, pos.z+1)),
			    make_float3(tex3D(eigVecInv2Tex, pos.x, pos.y, pos.z+1)),
			    make_float3(tex3D(eigVecInv3Tex, pos.x, pos.y, pos.z+1))) * windPrime ;
	if(!is_first_NAN && !isANum(windPrimeRot))
	{
	  debugf4 = make_float4(pos, 1.1f);
	  is_first_NAN = true;
	}
    //--------set UVWRotation to windPrimeRotation    
	float3 UVWRot = windPrimeRot;
    //----- URot_1st first step????????????????????????????????
	float3 exp_eigVal = exp(eigVal * dt);//make_float3(exp(eigVal.x * dt), exp(eigVal.y * dt), exp(eigVal.z * dt));
      //float3 randxyz = make_float3(box_muller(seed));//box_muller random number
	float3 randxyz = sqrtf( (CoEps/(2.f*eigVal)) * ( exp(2.f*eigVal*dt)- make_float3(1.f, 1.f, 1.f)) )  * box_muller(seed);//box_muller random number
	
	if(!is_first_NAN && !isANum(randxyz))
	{
	  debugf4 = make_float4(pos, 1.2f);
	  is_first_NAN = true;
	}
	float3 UVWRot_1st = UVWRot*exp_eigVal - ka0/eigVal*(make_float3(1.f, 1.f, 1.f) - exp_eigVal) + randxyz; 
	
	if(!is_first_NAN && !isANum(UVWRot_1st))
	{
	  debugf4 = make_float4(UVWRot_1st, 1.3f);
	  is_first_NAN = true;
	}
	float3 UVW_1st = make_matrix3X3(
		      make_float3(tex3D(eigVec1Tex, pos.x, pos.y, pos.z+1)),
		      make_float3(tex3D(eigVec2Tex, pos.x, pos.y, pos.z+1)),
		      make_float3(tex3D(eigVec3Tex, pos.x, pos.y, pos.z+1))) * UVWRot_1st; 
// 	windPrime = make_float3(1.f, 1.f, 1.f) - g2nd*U_1st*dt; 
	float3 UVW_2nd = UVW_1st / (make_float3(1.f, 1.f, 1.f) - g2nd * UVW_1st * dt); 
	
	if(!is_first_NAN && !isANum(UVW_2nd))
	{
	  debugf4 = make_float4(windPrime, 1.4f);
	  is_first_NAN = true;
	}
	float U_2nd = UVW_2nd.x, V_2nd = UVW_2nd.y, W_2nd = UVW_2nd.z;
	float du_3rd = 0.5f*(  lam1.x*(                      taudy1.x*U_2nd*V_2nd + taudz1.x*U_2nd*W_2nd) 
				    + lam1.y*(taudx1.x*V_2nd*U_2nd + taudy1.x*V_2nd*V_2nd + taudz1.x*V_2nd*W_2nd) 
				    + lam1.z*(taudx1.x*W_2nd*U_2nd + taudy1.x*W_2nd*V_2nd + taudz1.x*W_2nd*W_2nd) 
				    + lam2.x*(                      taudy1.y*U_2nd*V_2nd + taudz1.y*U_2nd*W_2nd)
				    + lam2.y*(taudx1.y*V_2nd*U_2nd + taudy1.y*V_2nd*V_2nd + taudz1.y*V_2nd*W_2nd) 
				    + lam2.z*(taudx1.y*W_2nd*U_2nd + taudy1.y*W_2nd*V_2nd + taudz1.y*W_2nd*W_2nd) 
				    + lam3.x*(                      taudy1.z*U_2nd*V_2nd + taudz1.z*U_2nd*W_2nd)
				    + lam3.y*(taudx1.z*V_2nd*U_2nd + taudy1.z*V_2nd*V_2nd + taudz1.z*V_2nd*W_2nd) 
				    + lam3.z*(taudx1.z*W_2nd*U_2nd + taudy1.z*W_2nd*V_2nd + taudz1.z*W_2nd*W_2nd) 
				    )*dt;
	float dv_3rd = 0.5f * (  lam1.x*(taudx1.y*U_2nd*U_2nd + taudy1.y*U_2nd*V_2nd + taudz1.y*U_2nd*W_2nd)
				    + lam1.y*(taudx1.y*V_2nd*U_2nd +                       taudz1.y*V_2nd*W_2nd) 
				    + lam1.z*(taudx1.y*W_2nd*U_2nd + taudy1.y*W_2nd*V_2nd + taudz1.y*W_2nd*W_2nd) 
				    + lam2.x*(taudx2.x*U_2nd*U_2nd + taudy2.x*U_2nd*V_2nd + taudz2.x*U_2nd*W_2nd)
				    + lam2.y*(taudx2.x*V_2nd*U_2nd +                       taudz2.x*V_2nd*W_2nd) 
				    + lam2.z*(taudx2.x*W_2nd*U_2nd + taudy2.x*W_2nd*V_2nd + taudz2.x*W_2nd*W_2nd) 
				    + lam3.x*(taudx2.y*U_2nd*U_2nd + taudy2.y*U_2nd*V_2nd + taudz2.y*U_2nd*W_2nd)
				    + lam3.y*(taudx2.y*V_2nd*U_2nd +                       taudz2.y*V_2nd*W_2nd) 
				    + lam3.z*(taudx2.y*W_2nd*U_2nd + taudy2.y*W_2nd*V_2nd + taudz2.y*W_2nd*W_2nd) 
				    )*dt;
	float dw_3rd = 0.5f * (  lam1.x*(taudx1.z*U_2nd*U_2nd + taudy1.z*U_2nd*V_2nd + taudz1.z*U_2nd*W_2nd) 
				    + lam1.y*(taudx1.z*V_2nd*U_2nd + taudy1.z*V_2nd*V_2nd + taudz1.z*V_2nd*W_2nd) 
				    + lam1.z*(taudx1.z*W_2nd*U_2nd + taudy1.z*W_2nd*V_2nd                      ) 
				    + lam2.x*(taudx2.y*U_2nd*U_2nd + taudy2.y*U_2nd*V_2nd + taudz2.y*U_2nd*W_2nd)
				    + lam2.y*(taudx2.y*V_2nd*U_2nd + taudy2.y*V_2nd*V_2nd + taudz2.y*V_2nd*W_2nd) 
				    + lam2.z*(taudx2.y*W_2nd*U_2nd + taudy2.y*W_2nd*V_2nd                      ) 
				    + lam3.x*(taudx2.z*U_2nd*U_2nd + taudy2.z*U_2nd*V_2nd + taudz2.z*U_2nd*W_2nd)
				    + lam3.y*(taudx2.z*V_2nd*U_2nd + taudy2.z*V_2nd*V_2nd + taudz2.z*V_2nd*W_2nd) 
				    + lam3.z*(taudx2.z*W_2nd*U_2nd + taudy2.z*W_2nd*V_2nd                      ) 
				    )*dt;
        windPrime = UVW_2nd + make_float3(du_3rd, dv_3rd, dw_3rd); 
	float3 newPos;
	if(!is_first_NAN && !isANum(windPrime))
	{  
	  pos = pos;
// 	  if(!is_first_NAN)
	    debugf4 = make_float4(make_float3(du_3rd, dv_3rd, dw_3rd), 1.5f);
	}else
	{
	  float3 distxyz = (wind + windPrime) * dt;
	  if(distxyz.x > 1.f)
	  {
	    distxyz.x = 1.f;//continue;
	  } else
	  if(distxyz.y > 1.f)
	  {
	   distxyz.y = 1.f;//continue;//distxyz.y = 1.f;
	  } else
	  if(distxyz.z > 1.f)
	  {
	    distxyz.z = 1.f;//continue;//distxyz.z = 1.f;
	  } //else
	  //pos += distxyz;
	  newPos = pos + distxyz;
	}
     // cellType = tex3D(cellTypeTex, newPos.x, newPos.y, newPos.z+1);
      if(tex3D(cellTypeTex, newPos.x, newPos.y, newPos.z+1) == 0)
      {
        pos = newPos;
	newPos = getReflectPos(pos, newPos);
      }
      else
	pos = newPos;
       
      
	isDone = true;
// 	///////////////////////////////////////////////////////////////////
// 	pos += distxyz;
// //         pos += (wind + windPrime) * deltaTime/10.f + jitter;
//   ////////////////////////////////////////////////////////
//   // new position = old position + velocity * deltaTime 
//   ////////////////////////////////////////////////////// 
//   
// //     	pos += (wind) * dt/10.f;
      }  
#endif
      {
      int countPrm=0;
      volatile int cellType = tex3D(cellTypeTex, pos.x, pos.y, pos.z+1);
      debugf4 = make_float4(cellType,tex3D(CoEpsTex, pos.x, pos.y, pos.z+1), tex3D(sig2Tex, pos.x, pos.y, pos.z+1).x,tex3D(sig2Tex, pos.x, pos.y, pos.z+1).z);
      if(cellType == 0)
      { 
	
// 	debugf4 = make_float4(pos, 1.01f);
// 	thrust::get<3>(t) = debugf4; 
	return ;
      }
	  /////////////////calculate dt
	float tStepInp = 1.f;//read from input file
	float tStepRem = tStepInp;
	float dt = .01f;//tStepRem;
    //       float tStepMin = 1.0f; 
    //       int count = 0;
    //       int countprimeloop = 0;
      
	volatile float CoEps = tex3D(CoEpsTex, pos.x, pos.y, pos.z+1);
	volatile float3 eigVal = make_float3(tex3D(eigValTex, pos.x, pos.y, pos.z+1));
	float tFac = 0.5f; 
	volatile float3 sig = make_float3(tex3D(sig1Tex, pos.x, pos.y, pos.z+1).x,
					  tex3D(sig2Tex, pos.x, pos.y, pos.z+1).x,  
					  tex3D(sig2Tex, pos.x, pos.y, pos.z+1).z
	                                 );
	
	volatile float3 wind = make_float3(tex3D(windFieldTex, pos.x, pos.y, pos.z+1));
	volatile float3 ka0  = make_float3(tex3D(ka0Tex, pos.x, pos.y, pos.z+1));
	         float3 g2nd = make_float3(tex3D(g2ndTex, pos.x, pos.y, pos.z+1));
		 
	volatile float3 lam1 = make_float3(tex3D(lam1Tex, pos.x, pos.y, pos.z+1));
	volatile float3 lam2 = make_float3(tex3D(lam2Tex, pos.x, pos.y, pos.z+1));
	volatile float3 lam3 = make_float3(tex3D(lam3Tex, pos.x, pos.y, pos.z+1));
	
	volatile float3 taudx1 = make_float3(tex3D(taudx1Tex, pos.x, pos.y, pos.z+1)); 
	volatile float3 taudx2 = make_float3(tex3D(taudx2Tex, pos.x, pos.y, pos.z+1));  
	volatile float3 taudy1 = make_float3(tex3D(taudy1Tex, pos.x, pos.y, pos.z+1));
	volatile float3 taudy2 = make_float3(tex3D(taudy2Tex, pos.x, pos.y, pos.z+1)); 
	volatile float3 taudz1 = make_float3(tex3D(taudz1Tex, pos.x, pos.y, pos.z+1));
	volatile float3 taudz2 = make_float3(tex3D(taudz2Tex, pos.x, pos.y, pos.z+1)); 
	
	matrix3X3 eigMatrix = make_matrix3X3(
		      make_float3(tex3D(eigVec1Tex, pos.x, pos.y, pos.z+1)),
		      make_float3(tex3D(eigVec2Tex, pos.x, pos.y, pos.z+1)),
		      make_float3(tex3D(eigVec3Tex, pos.x, pos.y, pos.z+1)));
	
	matrix3X3 eigMatrixInv = make_matrix3X3(
			    make_float3(tex3D(eigVecInv1Tex, pos.x, pos.y, pos.z+1)),
			    make_float3(tex3D(eigVecInv2Tex, pos.x, pos.y, pos.z+1)),
			    make_float3(tex3D(eigVecInv3Tex, pos.x, pos.y, pos.z+1)));
	
	float tStepSigW = 2.f * sig.z * sig.z / CoEps;
	
	float tStepCal = tFac * minf4(fabs(-1.f/eigVal), tStepSigW);
	bool isDone = false;
// 	bool isSec = false;
// 	int count_loop = 0;
	while(!isDone)
	{
	  dt = minf4(tStepInp, tStepCal, tStepRem,dt);
	  
	  float3 windPrimeRot = eigMatrixInv * windPrime ; 
      //--------set UVWRotation to windPrimeRotation    
	  float3 UVWRot = windPrimeRot;
      //----- URot_1st first step????????????????????????????????
	  float3 exp_eigVal = exp(eigVal * dt);//make_float3(exp(eigVal.x * dt), exp(eigVal.y * dt), exp(eigVal.z * dt));
	  float3 randxyz_rand3 = box_muller(seed);//box_muller random number 
	  float3 randxyz = sqrtf( (CoEps/(2.f*eigVal)) * ( exp(2.f*eigVal*dt)- make_float3(1.f, 1.f, 1.f)) )  * randxyz_rand3;//box_muller(seed);//box_muller random number
	  randxyz_rand3.z = thrust::get<2>(t);
// 	  debugf4 = make_float4(randxyz_rand3, 999999);
	  
	  float3 UVWRot_1st = UVWRot*exp_eigVal - ka0/eigVal*(make_float3(1.f, 1.f, 1.f) - exp_eigVal) + randxyz; 
	  
	  
	  float3 UVW_1st = eigMatrix * UVWRot_1st; 
	  
// 	  debugf4 = make_float4(count_loop,dt,0,2);
	  
// 	  if(isHasSameSign(UVW_1st, g2nd*1.f) && !isSec)
// 	  {
// 	    dt = minf3(1.f/(g2nd * UVW_1st)) * rnd(seed);
// 	    dt = dt > 0 ? dt : -dt;
// 	    isSec = true;
// 	    count_loop++;
// 	    debugf4 = make_float4(UVW_1st, 1);
// 	    continue;
// 	  }
  // 	windPrime = make_float3(1.f, 1.f, 1.f) - g2nd*U_1st*dt; 
	  float3 UVW_2nd = UVW_1st / (make_float3(1.f, 1.f, 1.f) - g2nd * UVW_1st * dt); 
	  
	  float U_2nd = UVW_2nd.x, V_2nd = UVW_2nd.y, W_2nd = UVW_2nd.z;
	  float du_3rd = 0.5f*(  lam1.x*(                      taudy1.x*U_2nd*V_2nd + taudz1.x*U_2nd*W_2nd) 
				    + lam1.y*(taudx1.x*V_2nd*U_2nd + taudy1.x*V_2nd*V_2nd + taudz1.x*V_2nd*W_2nd) 
				    + lam1.z*(taudx1.x*W_2nd*U_2nd + taudy1.x*W_2nd*V_2nd + taudz1.x*W_2nd*W_2nd) 
				    + lam2.x*(                      taudy1.y*U_2nd*V_2nd + taudz1.y*U_2nd*W_2nd)
				    + lam2.y*(taudx1.y*V_2nd*U_2nd + taudy1.y*V_2nd*V_2nd + taudz1.y*V_2nd*W_2nd) 
				    + lam2.z*(taudx1.y*W_2nd*U_2nd + taudy1.y*W_2nd*V_2nd + taudz1.y*W_2nd*W_2nd) 
				    + lam3.x*(                      taudy1.z*U_2nd*V_2nd + taudz1.z*U_2nd*W_2nd)
				    + lam3.y*(taudx1.z*V_2nd*U_2nd + taudy1.z*V_2nd*V_2nd + taudz1.z*V_2nd*W_2nd) 
				    + lam3.z*(taudx1.z*W_2nd*U_2nd + taudy1.z*W_2nd*V_2nd + taudz1.z*W_2nd*W_2nd) 
				    )*dt;
	float dv_3rd = 0.5f * (  lam1.x*(taudx1.y*U_2nd*U_2nd + taudy1.y*U_2nd*V_2nd + taudz1.y*U_2nd*W_2nd)
				    + lam1.y*(taudx1.y*V_2nd*U_2nd +                       taudz1.y*V_2nd*W_2nd) 
				    + lam1.z*(taudx1.y*W_2nd*U_2nd + taudy1.y*W_2nd*V_2nd + taudz1.y*W_2nd*W_2nd) 
				    + lam2.x*(taudx2.x*U_2nd*U_2nd + taudy2.x*U_2nd*V_2nd + taudz2.x*U_2nd*W_2nd)
				    + lam2.y*(taudx2.x*V_2nd*U_2nd +                       taudz2.x*V_2nd*W_2nd) 
				    + lam2.z*(taudx2.x*W_2nd*U_2nd + taudy2.x*W_2nd*V_2nd + taudz2.x*W_2nd*W_2nd) 
				    + lam3.x*(taudx2.y*U_2nd*U_2nd + taudy2.y*U_2nd*V_2nd + taudz2.y*U_2nd*W_2nd)
				    + lam3.y*(taudx2.y*V_2nd*U_2nd +                       taudz2.y*V_2nd*W_2nd) 
				    + lam3.z*(taudx2.y*W_2nd*U_2nd + taudy2.y*W_2nd*V_2nd + taudz2.y*W_2nd*W_2nd) 
				    )*dt;
	float dw_3rd = 0.5f * (  lam1.x*(taudx1.z*U_2nd*U_2nd + taudy1.z*U_2nd*V_2nd + taudz1.z*U_2nd*W_2nd) 
				    + lam1.y*(taudx1.z*V_2nd*U_2nd + taudy1.z*V_2nd*V_2nd + taudz1.z*V_2nd*W_2nd) 
				    + lam1.z*(taudx1.z*W_2nd*U_2nd + taudy1.z*W_2nd*V_2nd                      ) 
				    + lam2.x*(taudx2.y*U_2nd*U_2nd + taudy2.y*U_2nd*V_2nd + taudz2.y*U_2nd*W_2nd)
				    + lam2.y*(taudx2.y*V_2nd*U_2nd + taudy2.y*V_2nd*V_2nd + taudz2.y*V_2nd*W_2nd) 
				    + lam2.z*(taudx2.y*W_2nd*U_2nd + taudy2.y*W_2nd*V_2nd                      ) 
				    + lam3.x*(taudx2.z*U_2nd*U_2nd + taudy2.z*U_2nd*V_2nd + taudz2.z*U_2nd*W_2nd)
				    + lam3.y*(taudx2.z*V_2nd*U_2nd + taudy2.z*V_2nd*V_2nd + taudz2.z*V_2nd*W_2nd) 
				    + lam3.z*(taudx2.z*W_2nd*U_2nd + taudy2.z*W_2nd*V_2nd                      ) 
				    )*dt;
        windPrime = UVW_2nd + make_float3(du_3rd, dv_3rd, dw_3rd);
	
// 	float terFacU = 2.5;
//         float terFacV = 2.0;
//         float terFacW = 2.0;
        int flagPrime=0;
	if(fabs(windPrime.x)>terFacU*fabs(sig.x) && countPrm<countPrmMax){ 
	      debugf4.w = box_muller_1(seed);
	      windPrime.x=sig.x * debugf4.w;  
// 	      windPrime.x=sig.x * box_muller_1(seed);  
	      countPrm++; 
	      flagPrime=1;
// 	      filestream<< " uPrime "; 
	    }
	    if(fabs(windPrime.y)>terFacV*fabs(sig.y) && countPrm<countPrmMax){
	      debugf4.w = box_muller_1(seed);
	      windPrime.y=sig.y * debugf4.w;  
// 	      windPrime.y=sig.y * box_muller_1(seed);   
	      countPrm++; 
	      flagPrime=1;
// 	      filestream<< " vPrime "; 
	    }
	    if(fabs(windPrime.z)>terFacW*fabs(sig.z) && countPrm<countPrmMax){
	      debugf4.w = box_muller_1(seed);
	      windPrime.z=sig.z * debugf4.w;  
// 	      windPrime.z=sig.z * box_muller_1(seed); 
	      countPrm++; 
	      flagPrime=1;
// 	      filestream<< " wPrime "; 
	    }
	    
	    if(flagPrime==1){
	      flagPrime=0;  
// 	      filestream<< countPrm<<"   contiune\n"; 
	      continue;
	    }
	
	isDone = true;
	}
	float3 newPos = pos + (wind + windPrime) * dt;
	if(tex3D(cellTypeTex, newPos.x, newPos.y, newPos.z+1) == 0)
	{
	  pos = newPos;
	  newPos = getReflectPos(pos, newPos);
	}
// 	else
	  pos = newPos;
	
      }
    }
      // store new position and windPrime
    thrust::get<0>(t) = make_float4(pos, 0.f); 
    thrust::get<1>(t) = make_float4(windPrime, 0.f);//1*110*153 + floor(pos.y)*153 + floor(pos.x) ); 
    thrust::get<3>(t) = debugf4; //make_float4(2.22044604925031e-45/2.f,2.22044604925031e-46,2.22044604925031e-45/8.f, 2.22044604925031e-45/4.f); // debugf4; 
  }
};

 
//debugging the winddata stored in texture memory
struct copyDeviceData_functor
{   
  int texname; 
  __host__ __device__
  copyDeviceData_functor(int texname1) : texname(texname1){}

  template <typename Tuple>
  __device__
  void operator()(Tuple t)
  {    
    int index = thrust::get<1>(t); 
    int k = index/((int)g_params.domain.x * (int)g_params.domain.y); //int k = index/(110*153);
    int j = (index - k * (int)g_params.domain.x*(int)g_params.domain.y)/ (int)g_params.domain.x;// int j = (index - k*110*153)/153;
    int i = (index - k*(int)g_params.domain.y*(int)g_params.domain.x - j* (int)g_params.domain.x); //int i = (index - k*110*153 - j*153);
    switch(texname)
    {  
      case 1:  thrust::get<0>(t) = tex3D(windFieldTex, i, j, k);  break;
      case 2:  thrust::get<0>(t) = tex3D(eigValTex, i, j, k);     break;
      case 3:  thrust::get<0>(t) = tex3D(ka0Tex, i, j, k);        break;
      case 4:  thrust::get<0>(t) = tex3D(g2ndTex, i, j, k);       break;
      case 5:  thrust::get<0>(t) = tex3D(eigVec1Tex, i, j, k);    break;
      case 6:  thrust::get<0>(t) = tex3D(eigVec2Tex, i, j, k);    break;
      case 7:  thrust::get<0>(t) = tex3D(eigVec3Tex, i, j, k);    break;
      case 8:  thrust::get<0>(t) = tex3D(eigVecInv1Tex, i, j, k); break;
      case 9:  thrust::get<0>(t) = tex3D(eigVecInv2Tex, i, j, k); break;
      case 10: thrust::get<0>(t) = tex3D(eigVecInv3Tex, i, j, k); break;
      case 11: thrust::get<0>(t) = tex3D(lam1Tex, i, j, k);       break;
      case 12: thrust::get<0>(t) = tex3D(lam2Tex, i, j, k);       break;
      case 13: thrust::get<0>(t) = tex3D(lam3Tex, i, j, k); 	  break;
      case 14: thrust::get<0>(t) = tex3D(sig1Tex, i, j, k);	  break;
      case 15: thrust::get<0>(t) = tex3D(sig2Tex, i, j, k); 	  break;
      case 16: thrust::get<0>(t) = tex3D(taudx1Tex, i, j, k);	  break;
      case 17: thrust::get<0>(t) = tex3D(taudx2Tex, i, j, k); 	  break;
      case 18: thrust::get<0>(t) = tex3D(taudy1Tex, i, j, k);	  break;
      case 19: thrust::get<0>(t) = tex3D(taudy2Tex, i, j, k); 	  break;
      case 20: thrust::get<0>(t) = tex3D(taudz1Tex, i, j, k);	  break;
      case 21: thrust::get<0>(t) = tex3D(taudz2Tex, i, j, k); 	  break; 
//       case 22: thrust::get<0>(t) = tex3D(cellTypeTex, i, j, k); 	  break; 
    } 
  } 
};

struct rand_box_muller_f4 
{     
  template <typename Tuple>
  __device__
  void operator()(Tuple t)
  {      
//     unsigned int seed = tea<16>(thrust::get<1>(t), thrust::get<1>(t));   
//     thrust::get<0>(t) = box_muller(seed); 
//     thrust::get<0>(t) = make_float4(exp(1.2f), exp(88.9f),exp(-8.9f),exp(0.00009f));
  } 
};
 
  
#endif
