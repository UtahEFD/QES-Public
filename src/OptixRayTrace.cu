
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
__constant__ Params params; //should match var name in initPipeline
}

extern "C" __global__ void __raygen__from_cell(){


  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();

  const uint32_t linear_idx = idx.x + idx.y*(dim.x-1) + idx.z*(dim.y-1)*(dim.x-1);

  uint32_t t;
  //float t;
  
  params.flag = 40; //test to see if device-> host is updated

  if(params.icellflagArray[linear_idx] == 1){
     float3 origin = make_float3((dim.x+0.5)*params.dx, (dim.y+0.5)*params.dy, (dim.z+0.5)*params.dz);
     float3 dir = make_float3(0.0,0.0,-1.0);

     optixTrace(params.handle,
                origin,
                dir,
                0.0f,
                1e34f,
                0.0f,
                OptixVisibilityMask(1),
                OPTIX_RAY_FLAG_NONE,
                RAY_TYPE_RADIENCE,
                RAY_TYPE_COUNT,
                RAY_TYPE_RADIENCE,
                t
               );

      Hit hit;
      hit.t = int_as_float(t);
      //hit.t = t;
      
      printf("In .cu, t = %d\n", hit.t);
      params.hits[linear_idx] = hit;

      
  } //end of if for icell


/*
//single direction version start 
  if(params.rays[linear_idx].isRay){   //non-air cells will have a flag = 0

    printf("in .cu, in if condition, meaning this is an air cell\n");



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

} //end of if 


*/ 

} //end of raygen function

extern "C" __global__ void __miss__miss(){
   //printf("miss");

   optixSetPayload_0(float_as_int(FLT_MAX)); //need to set to a large number 

}

extern "C" __global__ void __closesthit__mixlength(){
  HitGroupData *rt_data = (HitGroupData *)optixGetSbtDataPointer();
  const uint32_t t = optixGetRayTmax();


  printf("In .cu, closet hit called \n");

  optixSetPayload_0(float_as_int(t));

}


