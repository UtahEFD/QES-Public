
//Contains all the functions to be used in OptixRayTrace


#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <vector>
#include <time.h>
#include <cmath>

#include <curand.h>
#include <curand_kernel.h>

#include "OptixRayTrace.h"

extern "C" {
__constant__ Params params; //should match var name in initPipeline
}


/*
__forceinline__ __device__ float random(unsigned int seed){
       curandState_t state;
  //     curand_init(seed, 0, 0, &state);
       curand_init(129, 0, 0, &state);
       //return curand_uniform(&state);

       

       return (curand_uniform(&state)*2.0f) - 1.0;
}
*/

/*
__forceinline__ __device__ float3 randDir(){
   float x, y, z;

   unsigned int seed = clock();
   x = random(seed);
   seed = clock()*100 + seed;
   y = random(seed);
   seed = clock()*50;
   z = random(seed);
 float magnitude = std::sqrt((x*x) + (y*y) + (z*z));
  
  return make_float3( x/magnitude, y/magnitude, z/magnitude);
 

 //  return make_float3(x,y,z);
}
*/


/*
__forceinline__ __device__ float randBound(float lower, float upper){
      curandState_t state;
      curand_init(129, 0, 0, &state);
      return (curand_uniform(&state)*(upper-lower))+lower;          
}
*/

/*

__forceinline__ __device__ float3 randDir(){
  float theta = std::asin(randBound(-1.0f, 1.0f));
  float phi = randBound(0.0f, 2.0f*3.14f);     //change to PI value

  float x = std::cos(theta)*std::cos(phi);
  float y = std::sin(phi);
  float z = std::cos(theta)*sin(phi);

  float magnitude = std::sqrt((x*x) + (y*y) + (z*z));
  
  return make_float3( x/magnitude, y/magnitude, z/magnitude);
  
}
*/


__global__ void randBound(float *num, float upper, float lower){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  curandState state;
  curand_init(clock64(), i, 0, &state);
  *num = (curand_uniform(&state)*(upper - lower))+lower;
}


extern "C" __global__ void __raygen__from_cell(){

  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  
  const uint32_t linear_idx = idx.x + idx.y*(dim.x-1) + idx.z*(dim.y-1)*(dim.x-1);


  uint32_t t;

  if(params.icellflagArray[linear_idx] == 1){

     float lowestLen = FLT_MAX; //current lowest length


     float3 cardinal[5]{
         make_float3(0,0,-1),
         make_float3(1,0, 0),
         make_float3(-1,0,0),
         make_float3(0,1,0),
         make_float3(0,-1,0)
     };

   
     float3 origin = make_float3((idx.x+0.5)*params.dx, (idx.y+0.5)*params.dy, (idx.z+0.5)*params.dz);
     float3 dir;


     curandState_t state;
     curand_init(129, 0, 0, &state);



     for(int i = 0; i < params.numSamples; i++){
   
         if(i < 5){
            dir = cardinal[i];
         }else{
            //dir = randDir();
            
            float theta = std::asin((curand_uniform(&state)*2.0) - 1.0);
            float phi = (curand_uniform(&state)*2*M_PI);     

            float x = std::cos(theta)*std::cos(phi);
            float y = std::sin(phi);
            float z = std::cos(theta)*sin(phi);

            float magnitude = std::sqrt((x*x) + (y*y) + (z*z));
  
            dir = make_float3( x/magnitude, y/magnitude, z/magnitude);

            

            /*

            float x = (curand_uniform(&state) * 2.0) - 1.0;
            float y = (curand_uniform(&state) * 2.0) - 1.0;
            float z = (curand_uniform(&state) * 2.0) - 1.0;
            
            float magnitude = std::sqrt((x*x) + (y*y) + (z*z));
            dir = make_float3( x/magnitude, y/magnitude, z/magnitude);
            */          
         }
     

         //printf("In .cu, dir = <%f, %f, %f>\n", dir.x, dir.y, dir.z);

         optixTrace(params.handle,
                    origin,
                    dir,
                    0.0f,
                    1e16f,
                    0.0f,
                    //OptixVisibilityMask(1),
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_NONE,
                    RAY_TYPE_RADIENCE,
                    RAY_TYPE_COUNT,
                    RAY_TYPE_RADIENCE,
                    t
                    );


         if(int_as_float(t) < lowestLen){

            lowestLen = int_as_float(t);
         }

     } //end of for loop

 
     Hit hit;
     hit.t = lowestLen;
      
     params.hits[linear_idx] = hit;

      
  } //end of if for icell

} //end of raygen function



extern "C" __global__ void __miss__miss(){

   optixSetPayload_0(float_as_int(FLT_MAX)); //set to a large number 

}

extern "C" __global__ void __closesthit__mixlength(){

  
  const uint32_t t = optixGetRayTmax(); //get t value from OptiX function 

  optixSetPayload_0(float_as_int(t));   //assign payload

}


