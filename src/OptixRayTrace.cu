//Contains all the functions to be used in OptixRayTrace

#include <optix.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>

#include "OptixRayTrace.h"


extern "C" {
__constant__ Params params;
}

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
             OptixTraceVisibilityMask(1),
             OPTIX_RAY_FLAG_NONE,
             RAY_TYPE_RADIANCE,
             RAY_TYPE_COUNT,
             RAY_TYPE_RADIANCE,
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
    //  Vertex vdir;
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
  optixSetPayload_0(); //need to set to a large number 
}

extern "C" __global__ void __closesthit__mixlength(){
HitGroupData rt_data = (HitGroupData *)optixGetSbtDataPointer();
const uint32_t t = optixGetRayTmax();

optixSetPayload_0(float_as_int(t));
}


