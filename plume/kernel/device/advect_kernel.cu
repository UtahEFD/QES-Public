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
#include <curand_kernel.h> 
#include "cutil_math.h"
#include "math_constants.h"
#include "../kernel_global/ConstParams.cuh" 
#include "../kernel_global/texture_mem.cuh" 
#include "particles_reflection.cu" 
#include "../kernel_global/turbulence.cuh"

__constant__ int MAXCOUNT = 10;//which could be moved to g_params(const value) 
__constant__ float terFacU = 2.5;
__constant__ float terFacV = 2.;
__constant__ float terFacW = 2.;
std::ofstream random_file;



// __device__ void resetParticles(float3 &pos, float &seed_para, uint &seed)//resetParticles(float3 &pos, float3 &vel)
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
//     seed_para = pos.x + pos.y + pos.z;
  }
}

// __device__ void readTex(float3 wind, float3 ka0, float3 g2nd, float3 lam1, float3 lam2, float3 lam3,)
// template <typename InputIterator>
struct advect_functor
{
  unsigned long deltaTime; 
  __host__ __device__
  advect_functor(unsigned long delta_time) : deltaTime(delta_time){}//,domain(make_float3(29,1,29)) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t)
  {    
    float3 pos = make_float3(thrust::get<0>(t)); 
    int seed_first_para = thrust::get<0>(t).w;
    float3 windPrime = make_float3(thrust::get<1>(t));  
//     bool is_seed_flag = thrust::get<2>(t); 
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    uint offset = x + y * blockDim.x * gridDim.x; 
    unsigned int seed = offset + deltaTime;//tea<16>(pos.x*100, offset);   
//     unsigned int seed = tea<16>(pos.x*100, seed_first_para);  
    float3 seed_record = make_float3(seed, pos.x*100, offset);
//     unsigned int seed = offset;
    
    float4 debugf4 = thrust::get<3>(t); 
    int countPrmMax=10;//1000;
//--------------reset the particles that are out of boundary        
//     debugf4 = make_float4(pos, thrust::get<2>(t));;
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
      debugf4 = make_float4(0, 0, 0, seed);
//       resetParticles(pos,seed_first_para, seed);        
    } 
    else 
//------ if particle hits buildings then it bounces     
    {  
      {
      int countPrm=0;
      volatile int cellType = tex3D(cellTypeTex, pos.x, pos.y, pos.z+1);
      debugf4 = make_float4(cellType,tex3D(CoEpsTex, pos.x, pos.y, pos.z+1), tex3D(sig2Tex, pos.x, pos.y, pos.z+1).x,tex3D(sig2Tex, pos.x, pos.y, pos.z+1).z);
      if(cellType == 0)
      { 
	resetParticles(pos, seed); //resetParticles(pos, vel);   
        debugf4 = make_float4(0, 0, 0, seed);
// 	debugf4 = make_float4(pos, 1.01f);
// 	thrust::get<3>(t) = debugf4; 
	return ;
      }
	  /////////////////calculate dt
	float tStepInp = 1.f;//read from input file
	float tStepRem = tStepInp;
	float dt = .03f;//tStepRem;
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
// 	curandState s;
// 	curand_init(seed , 0, 0, &s) ; 
	while(!isDone)
	{
	  dt = minf4(tStepInp, tStepCal, tStepRem,dt);
	  
	  float3 windPrimeRot = eigMatrixInv * windPrime ; 
      //--------set UVWRotation to windPrimeRotation    
	  float3 UVWRot = windPrimeRot;
      //----- URot_1st first step????????????????????????????????
	  float3 exp_eigVal = exp(eigVal * dt);//make_float3(exp(eigVal.x * dt), exp(eigVal.y * dt), exp(eigVal.z * dt));
// 	  float3 randxyz_rand3 = box_muller(seed, is_seed_flag);//box_muller random number 
// 	  seed *= 10000;

	  curandState s;
	  curand_init(seed, 0, 0, &s) ; 
	  float randx = curand_normal(&s);// curand_log_normal(&s, 0, 1.f);
// 	  curand(&s);
	  float randy = curand_normal(&s);// curand_log_normal(&s, 0, 1.f);
// 	  curand(&s);
	  float randz = curand_normal(&s);// curand_log_normal(&s, 0, 1.f); 
	  float3 randxyz_rand3 = make_float3(randx, randy, randz);
// 	  float3 randxyz_rand3 = make_float3(curand_uniform(&s), curand_uniform(&s), curand_uniform(&s));
	  float3 randxyz = sqrtf( (CoEps/(2.f*eigVal)) * ( exp(2.f*eigVal*dt)- make_float3(1.f, 1.f, 1.f)) ) 
			  * randxyz_rand3;//box_muller(seed, is_seed_flag);//box_muller random number
// 	  randxyz_rand3.z = thrust::get<2>(t);
	  debugf4 = make_float4(randxyz_rand3, offset);
	  
	  float3 UVWRot_1st = UVWRot*exp_eigVal - ka0/eigVal*(make_float3(1.f, 1.f, 1.f) - exp_eigVal) + randxyz; 
// 	  s	  
	  float3 UVW_1st = eigMatrix * UVWRot_1st; 
	   
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
	 
        int flagPrime=0;
	if(fabs(windPrime.x)>terFacU*fabs(sig.x) && countPrm<countPrmMax){ 
	      debugf4.w = curand_normal(&s);//box_muller_1(seed, is_seed_flag);
	      windPrime.x=sig.x * debugf4.w;  
// 	      windPrime.x=sig.x * box_muller_1(seed);  
	      countPrm++; 
	      flagPrime=1;
// 	      filestream<< " uPrime "; 
	    }
	    if(fabs(windPrime.y)>terFacV*fabs(sig.y) && countPrm<countPrmMax){
	      debugf4.w = curand_normal(&s);//box_muller_1(seed, is_seed_flag);
	      windPrime.y=sig.y * debugf4.w;  
// 	      windPrime.y=sig.y * box_muller_1(seed);   
	      countPrm++; 
	      flagPrime=1;
// 	      filestream<< " vPrime "; 
	    }
	    if(fabs(windPrime.z)>terFacW*fabs(sig.z) && countPrm<countPrmMax){
	      debugf4.w = curand_normal(&s);//box_muller_1(seed, is_seed_flag);
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
//     pos += randxyz_rand3*dt;
	}
	float3 newPos = pos + (wind + windPrime) * dt;
	if(tex3D(cellTypeTex, newPos.x, newPos.y, newPos.z+1) == 0/* ||
	   tex3D(cellTypeTex, newPos.x-g_params.particleRadius, newPos.y-g_params.particleRadius, newPos.z+1-g_params.particleRadius) == 0 ||
	   tex3D(cellTypeTex, newPos.x-g_params.particleRadius, newPos.y-g_params.particleRadius, newPos.z+1+g_params.particleRadius) == 0 ||
	   tex3D(cellTypeTex, newPos.x-g_params.particleRadius, newPos.y+g_params.particleRadius, newPos.z+1+g_params.particleRadius) == 0 ||
	   tex3D(cellTypeTex, newPos.x+g_params.particleRadius, newPos.y-g_params.particleRadius, newPos.z+1-g_params.particleRadius) == 0 ||
	   tex3D(cellTypeTex, newPos.x+g_params.particleRadius, newPos.y-g_params.particleRadius, newPos.z+1+g_params.particleRadius) == 0 ||
	   tex3D(cellTypeTex, newPos.x+g_params.particleRadius, newPos.y+g_params.particleRadius, newPos.z+1+g_params.particleRadius) == 0 ||
	   tex3D(cellTypeTex, newPos.x+g_params.particleRadius, newPos.y+g_params.particleRadius, newPos.z+1-g_params.particleRadius) == 0 
	*/)
	{
// 	  pos = newPos;
	  newPos = getReflectPos(pos, newPos, g_params.particleRadius);
	}
// 	else
	  pos = newPos;
	
      }
    }
    debugf4 = make_float4(0, seed_record);
    // store new position and windPrime
    thrust::get<0>(t) = make_float4(pos, thrust::get<0>(t).w); 
    thrust::get<1>(t) = make_float4(windPrime, 0.f);//1*110*153 + floor(pos.y)*153 + floor(pos.x) ); 
//     thrust::get<2>(t) = is_seed_flag;
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
    i=j+k;i++;
//     switch(texname)
//     {  
//       case 1:  thrust::get<0>(t) = tex3D(windFieldTex, i, j, k);  break;
//       case 2:  thrust::get<0>(t) = tex3D(eigValTex, i, j, k);     break;
//       case 3:  thrust::get<0>(t) = tex3D(ka0Tex, i, j, k);        break;
//       case 4:  thrust::get<0>(t) = tex3D(g2ndTex, i, j, k);       break;
//       case 5:  thrust::get<0>(t) = tex3D(eigVec1Tex, i, j, k);    break;
//       case 6:  thrust::get<0>(t) = tex3D(eigVec2Tex, i, j, k);    break;
//       case 7:  thrust::get<0>(t) = tex3D(eigVec3Tex, i, j, k);    break;
//       case 8:  thrust::get<0>(t) = tex3D(eigVecInv1Tex, i, j, k); break;
//       case 9:  thrust::get<0>(t) = tex3D(eigVecInv2Tex, i, j, k); break;
//       case 10: thrust::get<0>(t) = tex3D(eigVecInv3Tex, i, j, k); break;
//       case 11: thrust::get<0>(t) = tex3D(lam1Tex, i, j, k);       break;
//       case 12: thrust::get<0>(t) = tex3D(lam2Tex, i, j, k);       break;
//       case 13: thrust::get<0>(t) = tex3D(lam3Tex, i, j, k); 	  break;
//       case 14: thrust::get<0>(t) = tex3D(sig1Tex, i, j, k);	  break;
//       case 15: thrust::get<0>(t) = tex3D(sig2Tex, i, j, k); 	  break;
//       case 16: thrust::get<0>(t) = tex3D(taudx1Tex, i, j, k);	  break;
//       case 17: thrust::get<0>(t) = tex3D(taudx2Tex, i, j, k); 	  break;
//       case 18: thrust::get<0>(t) = tex3D(taudy1Tex, i, j, k);	  break;
//       case 19: thrust::get<0>(t) = tex3D(taudy2Tex, i, j, k); 	  break;
//       case 20: thrust::get<0>(t) = tex3D(taudz1Tex, i, j, k);	  break;
//       case 21: thrust::get<0>(t) = tex3D(taudz2Tex, i, j, k); 	  break; 
// //       case 22: thrust::get<0>(t) = tex3D(cellTypeTex, i, j, k); 	  break; 
//     } 
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
 
__global__ void test_kernel(float4* posPtr, float4* primePtr, bool* is_seed_flagPtr,
			    float4* debugPtr, turbulence* device_turbs_test)//turbulence* device_turbs_test)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  uint offset = x + y * blockDim.x * gridDim.x; 
  
  
  float3 pos = make_float3(posPtr[offset]);
  unsigned int seed = offset;// = tea<16>(pos.x*100, offset); 
  float3 windPrime = make_float3(primePtr[offset]);
//   bool is_seed_flag = is_seed_flagPtr[offset];
  float4 debugf4; 
   
  int cellIndex = (int)(pos.z+1)*g_params.domain.y*g_params.domain.x + (int)(pos.y)*g_params.domain.x + (int)pos.x; 
  turbulence turb = device_turbs_test[cellIndex];//*in;//(int)(pos.z+1)*110*153 + (int)(pos.y)*153 + (int)pos.x];
//   float3 wind = turb.windData;  
  float dt = 0.3;//tStepRem;
  int countPrmMax=10;//1000;
  int countPrm=0;

  int cellType = turb.cellType; 
  debugf4 = make_float4(turb.eigVal, cellType);

  if(cellType == 0)
  { //734910
    posPtr[offset]   = make_float4(pos, cellType); 
    debugf4 = make_float4(pos.x, pos.y, offset, cellType);
    debugPtr[offset] = debugf4;
    return ;
  }
	  ///////////////calculate dt
  float tStepInp = 1.0f;//read from input file
  float tStepRem = tStepInp; 

  float CoEps = turb.CoEps;
  float3 eigVal = turb.eigVal;
  float tFac = 0.5f; 
  float3 sig = make_float3(turb.sig1.x,turb.sig2.x,turb.sig2.z);

  float3 wind = turb.windData; 
  float3 ka0  = turb.ka0;
  float3 g2nd = turb.g2nd;
	  
  float3 lam1 = turb.lam1;
  float3 lam2 = turb.lam2;
  float3 lam3 = turb.lam3;

  float3 taudx1 = turb.taudx1; 
  float3 taudx2 = turb.taudx2;  
  float3 taudy1 = turb.taudy1;
  float3 taudy2 = turb.taudy2; 
  float3 taudz1 = turb.taudz1;
  float3 taudz2 = turb.taudz2; 
  
  matrix3X3 eigMatrix = make_matrix3X3(turb.eigVec1,turb.eigVec2,turb.eigVec3);
  
  matrix3X3 eigMatrixInv = make_matrix3X3(turb.eigVecInv1,turb.eigVecInv2,turb.eigVecInv3);
  
  float tStepSigW = 2.0f * sig.z * sig.z / CoEps;
  
// 	float tStepCal = tFac * minf4(fabs(-1.0f/eigVal.x), fabs(-1.0f/eigVal.y), fabs(-1.0f/eigVal.z), tStepSigW);
  float tStepCal = tFac * minf4(fabs(-1.0/eigVal), tStepSigW);
  bool isDone = false; 
  while(!isDone)
  {
     dt = minf4(tStepInp, tStepCal, tStepRem,dt);
	  
    float3 windPrimeRot = eigMatrixInv * windPrime ; 
//--------set UVWRotation to windPrimeRotation    
    float3 UVWRot = windPrimeRot;
//----- URot_1st first step????????????????????????????????
    float3 exp_eigVal = exp(eigVal * dt);//make_float3(exp(eigVal.x * dt), exp(eigVal.y * dt), exp(eigVal.z * dt));
// 	  float3 randxyz_rand3 = box_muller(seed, is_seed_flag);//box_muller random number 
// 	  seed *= 10000;

    curandState s;
    curand_init(seed, 0, 0, &s) ; 
    float randx = curand_normal(&s);// curand_log_normal(&s, 0, 1.f);
// 	  curand(&s);
    float randy = curand_normal(&s);// curand_log_normal(&s, 0, 1.f);
// 	  curand(&s);
    float randz = curand_normal(&s);// curand_log_normal(&s, 0, 1.f); 
    float3 randxyz_rand3 = make_float3(randx, randy, randz);
// 	  float3 randxyz_rand3 = make_float3(curand_uniform(&s), curand_uniform(&s), curand_uniform(&s));
    float3 randxyz = sqrtf( (CoEps/(2.f*eigVal)) * ( exp(2.f*eigVal*dt)- make_float3(1.f, 1.f, 1.f)) ) 
		    * randxyz_rand3;//box_muller(seed, is_seed_flag);//box_muller random number 
    debugf4 = make_float4(randxyz_rand3, offset);
    
    float3 UVWRot_1st = UVWRot*exp_eigVal - ka0/eigVal*(make_float3(1.f, 1.f, 1.f) - exp_eigVal) + randxyz; 
// // 	  s	  
    float3 UVW_1st = eigMatrix * UVWRot_1st; 
      
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
//       debugf4 = make_float4(UVW_2nd, 0);
      windPrime = UVW_2nd + make_float3(du_3rd, dv_3rd, dw_3rd);
      if(!isANum(windPrime))
      {
	windPrime = make_float3(2,2,2);
      }
      int flagPrime=0;
      if(fabs(windPrime.x)>terFacU*fabs(sig.x) && countPrm<countPrmMax){ 
	debugf4.w = curand_normal(&s); 
	windPrime.x=sig.x * debugf4.w;   
	countPrm++; 
	flagPrime=1;
  // 	      filestream<< " uPrime "; 
      }
      if(fabs(windPrime.y)>terFacV*fabs(sig.y) && countPrm<countPrmMax){
	debugf4.w = curand_normal(&s);//box_muller_1(seed, is_seed_flag);
	windPrime.y=sig.y * debugf4.w;   
	countPrm++; 
	flagPrime=1;
  // 	      filestream<< " vPrime "; 
      }
      if(fabs(windPrime.z)>terFacW*fabs(sig.z) && countPrm<countPrmMax){
	debugf4.w = curand_normal(&s);//box_muller_1(seed, is_seed_flag);
	windPrime.z=sig.z * debugf4.w;   
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
 // int newCellType = (int)(newPos.z+1)*110*153 + (int)(newPos.y)*153 + (int)newPos.x;//tex3D(cellTypeTex, newPos.x, newPos.y, newPos.z+1);
  int newIndex = (int)(newPos.z+1)*g_params.domain.y*g_params.domain.x + (int)(newPos.y)*g_params.domain.x + (int)newPos.x;//tex3D(cellTypeTex, newPos.x, newPos.y, newPos.z+1);
  cellType = device_turbs_test[newIndex].cellType;
//   if(newCellType==0)//(tex3D(cellTypeTex, newPos.x, newPos.y, newPos.z+1) == 0)
  if(cellType==0)
  {  
    newPos = getReflectPos(pos, newPos, g_params.particleRadius);
  }
//  else
    pos = newPos; 
  newIndex = (int)(pos.z+1)*g_params.domain.y*g_params.domain.x + (int)(pos.y)*g_params.domain.x + (int)pos.x;
  posPtr[offset]   = make_float4(pos, newIndex); //make_float4(pos, posPtr[offset].w); 
  primePtr[offset] = make_float4(windPrime, offset); 
//   debugPtr[offset] = debugf4;
//   is_seed_flagPtr[offset] = is_seed_flag;
}
  
#endif
