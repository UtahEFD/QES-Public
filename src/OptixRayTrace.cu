
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
//printf("In .cu, enters raygen\n");

  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  
  const uint32_t linear_idx = idx.x + idx.y*(dim.x-1) + idx.z*(dim.y-1)*(dim.x-1);



  uint32_t t;
  //float t;



 // params.flag = 40; //test to see if device-> host is updated

  //printf("icellflag at idx %i: %i\n", linear_idx, params.icellflagArray[linear_idx]);

  if(params.icellflagArray[linear_idx] == 1){

//printf("In .cu, it enters if conditional. In other words, it is an air cell.\n");

     //printf("params.dx = %f, params.dy = %f, params.dz = %f\n", params.dx, params.dy, params.dz);

    float3 origin = make_float3((idx.x+0.5)*params.dx, (idx.y+0.5)*params.dy, (idx.z+0.5)*params.dz);


     //float3 origin = make_float3(100,100,100); //default
     
     float3 dir = make_float3(0.0,0.0,-1.0);

     

//printf("Origin: (%f, %f, %f)      Dir: (%f, %f, %f)\n", origin.x,  origin.y, origin.z, dir.x, dir.y, dir.z);

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


      float temp_t;
      temp_t = int_as_float(t);
      printf("In .cu, t = %f, %i\n", temp_t, t);

      //Hit hit;
      //hit.t = int_as_float(t);
      //hit.t = t;
      
      //printf("In .cu, t = %d\n", hit.t);
      //printf("In .cu, t = %i, hit.t = %f\n", t, hit.t);
      //params.hits[linear_idx] = hit;

      
  } //end of if for icell



} //end of raygen function

extern "C" __global__ void __miss__miss(){
   printf("In .cu, miss\n");

//   optixSetPayload_0(float_as_int(FLT_MAX)); //need to set to a large number 


    optixSetPayload_0(float_as_int(5)); //test value 

}

extern "C" __global__ void __closesthit__mixlength(){
  printf("In .cu, closet hit called \n");


  //HitGroupData *rt_data = (HitGroupData *)optixGetSbtDataPointer();
  const uint32_t t = optixGetRayTmax();



  //optixSetPayload_0(float_as_int(t));

  optixSetPayload_0(float_as_int(10));  //test value

}


