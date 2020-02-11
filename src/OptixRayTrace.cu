//Contains all the functions to be used in OptixRayTrace

#include <optix.h>
#include <sutil/vec_math>
#include <stdio.h>
#include <vector>

#include "OptixRayTrace.h"


extern "C" {
__constant__ Params params;
}

extern "C" __global__void__raygen__from_cell(){
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDumensions();
  const unit32_t linear_idx = (idx.z*dim.x*dim.y) + (idx.y*dim.x) + idx.x;

  uint32_t t, nx, ny, nz;


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
             t,
             nx,
             ny,
             nz);

      Hit hit;
      hit.t = int_as_float(t);
      hit.geom_normal.x = int_as_float(nx);
      hit.geom_normal.y = int_as_float(ny);
      hit.geom_normal.z = int_as_float(nz);


      params.hits[linear_idx] = hit;

    //single direction version end 
   

   //multiple samples version start

   Hit tempHits[state.samples_per_cell];   
   for(int i = 0; i < state.samples_per_cell; i++){
      float dx, dy, dx;

      //add random sphere direction functionality
      
      params.rays[linear_idx].dir = make_float3(dx, dy, dz);

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
             t,
             nx,
             ny,
             nz);

     Hit hit;
     hit.t = int_as_float(t);
     hit.geom_normal.x = int_as_float(nx);
     hit.geom_normal.y = int_as_float(ny);
     hit.geom_normal.z = int_as_float(nz);

     tempHits[i] = hit;
   }

   int minIndex = 0;
   for(int i = 1; i < state.samples_per_cell; i++){
      if(tempHit[i].t < tempHits[minIndex].t){
        minIndex = i; 
      }
   }

   params.hits[linear_idx] = tempHits[minIndex];
   //multiple samples version end

  
  
  
}

extern "C" __global__void__miss__miss(){
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDumensions();
  const unit32_t linear_idx = (idx.z*dim.x*dim.y) + (idx.y*dim.x) + idx.x;


  Hit hit;
  hit.t = int_as_float(1e34f);  //check to see if this is really the max 
  params.hits[linear_idx] = hit; 
  
}

extern "C" __global__void__closesthit__mixlength(){
//handles what should happen in the closest hit program

//maybe later this will handle the comparison of multiple rays per
//cell 
}


