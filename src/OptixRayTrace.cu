//Contains all the functions to be used in OptixRayTrace


#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <vector>
#include <time.h>

#include "OptixRayTrace.h"

extern "C" {
__constant__ Params params;
}

/*
static __forceinline__ __device__ uint32_t thomas_wang_hash(uint32_t seed){
       seed = (seed ^ 61) ^ (seed >> 16);
       seed = seed + (seed << 3);
       seed = seed ^ (seed >> 4);
       seed *= 0x27d4eb2d;
       seed = seed ^ (seed >>15);
       return seed;
}

static __forceinline__ __device__ uint32_t genSeed(uint32_t val){
       time_t gmTime;
       struct tm *gmTimeInfo;
       time(&gmTime);
       gmTimeInfo = gmtime(&gmTime);

       uint32_t seed = gmTimeInfo->tm_hour + gmTimeInfo->tm_sec/gmTimeInfo->tm_year*val;

       //for(int i = 0; i < 3; i++){
         seed = thomas_wang_hash(seed);
       //}
       return seed;
}

static __forceinline__ __device__ float rndNum(){
   return 1.0;
}
*/

extern "C" __global__ void __raygen__from_cell(){
  const uint3 idx = optixGetLaunchIndex();
   const uint3 dim = optixGetLaunchDimensions();
  const uint32_t linear_idx = (idx.z*dim.x*dim.y) + (idx.y*dim.x) + idx.x;

  uint32_t t;


  //single direction version start 
  optixTrace(params.handle,
             params.rays[linear_idx].origin,
             params.rays[linear_idx].dir,
             params.rays[linear_idx].tmin,
             params.rays[linear_idx].tmax,
             0.0f,
             OptixVisibilityMask(1),
             OPTIX_RAY_FLAG_NONE,
             RAY_TYPE_RADIENCE,
             RAY_TYPE_COUNT,
             RAY_TYPE_RADIENCE,
             t);

      Hit hit;
      hit.t = int_as_float(t);

      params.hits[linear_idx] = hit;

    //single direction version end 
   

   //multiple samples version start

   // Hit tempHits[state.samples_per_cell];   
   // for(int i = 0; i < state.samples_per_cell; i++){
   //    float dx, dy, dx;
   //    //add random sphere direction functionality
         //uint32_t seed = genSeed(i+idx.x);
      //Vertex vdir;
       // vdir.x = dx;
     // vdir.y = dy;
     // vdir.z = dz;
   //    params.rays[linear_idx].dir = vdir;

   //    optixTrace(params.handle,
   //           params.rays[linear_idx].origin,
   //           params.rays[linear_idx].dir,
   //           params.rays[linear_idx].tmin,
   //           params.rays[linear_idx].tmax,
   //           0.0f,
   //           OptixTraceVisibilityMask(1),
   //           OPTIX_RAY_FLAG_NONE,
   //           RAY_TYPE_RADIANCE,
   //           RAY_TYPE_COUNT,
   //           RAY_TYPE_RADIANCE,
   //           t);

   //   Hit hit;
   //   hit.t = int_as_float(t);
   //   tempHits[i] = hit;
   // }

   // int minIndex = 0;
   // for(int i = 1; i < state.samples_per_cell; i++){
   //    if(tempHit[i].t < tempHits[minIndex].t){
   //      minIndex = i; 
   //    }
   // }

   // params.hits[linear_idx] = tempHits[minIndex];

   //multiple samples version end
}

extern "C" __global__ void __miss__miss(){
   optixSetPayload_0(float_as_int(FLT_MAX)); //need to set to a large number 
}

extern "C" __global__ void __closesthit__mixlength(){
HitGroupData *rt_data = (HitGroupData *)optixGetSbtDataPointer();
const uint32_t t = optixGetRayTmax();

optixSetPayload_0(float_as_int(t));
}


