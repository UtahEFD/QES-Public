#include "Cell.cuh" 
#include <cstdlib>
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
 
 struct functor
{
    //float deltaTime;
    //float3 domain;
    //float3 ground;
    __host__ __device__
    functor(){}// : deltaTime(delta_time){}//,domain(make_float3(29,1,29)) {}

   // t - 
    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {   
      volatile float posData = thrust::get<0>(t);
     //volatile float velData = thrust::get<1>(t);
//       float3 pos = make_float3(posData.x, posData.y, posData.z);
//       float3 vel = make_float3(velData.x, velData.y, velData.z); 
      // store new position and velocity
      thrust::get<0>(t) = posData+0.02f;//make_float4(pos, posData.w);
     // thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};
 
 
 
 
 
 
 int main()
{
  int numParticle = 100;
  float* h_pos = new float(numParticle);
  
  for (int i = 0 ; i < numParticle; ++i) 
  {
    h_pos[i] = i/10.f;
  }
  float* d_pos1;	 
  cudaMalloc((void**)&d_pos1, numParticle*sizeof(float)); 
  cudaMemcpy(d_pos1,  h_pos, numParticle * sizeof(float), cudaMemcpyHostToDevice); 
  thrust::device_ptr<float> d_pos(d_pos1); 

  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(d_pos)),
      thrust::make_zip_iterator(thrust::make_tuple(d_pos + numParticle)),
      functor());
      
  cudaMemcpy(h_pos,  d_pos1, numParticle * sizeof(float), cudaMemcpyDeviceToHost); 
  for(uint i=0; i<numParticle	; i++) 
  {
//     printf("this is:%d x=%f, y=%f, z=%f\n", i, dCells[i].wind.x, dCells[i].wind.y, dCells[i].wind.z);    
//     printf("this is:%d x=%f, y=%f, z=%f\n", i, df[i] , df[i] , df[i] );          
    printf("this is:%d x=%f, y=%f, z=%f\n", i, h_pos[i] , h_pos[i] , h_pos[i] );      
  }
}