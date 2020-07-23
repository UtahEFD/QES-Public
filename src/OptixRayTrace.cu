
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


extern "C" __global__ void __raygen__from_cell(){

  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  
  const uint32_t linear_idx = idx.x + idx.y*(dim.x-1) + idx.z*(dim.y-1)*(dim.x-1);

  uint32_t t;

  //if not building or terrain cell
 if(params.icellflagArray[linear_idx] != 0 && params.icellflagArray[linear_idx] != 2){


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

            //Trignometric-Polar Method
            
            float theta = (curand_uniform(&state)*M_PI);
            float phi = (curand_uniform(&state)*2*M_PI);
            
            float x = std::cos(phi)*std::sin(theta);
            float y = std::sin(theta)*std::sin(phi);
            float z = std::cos(theta);


            float magnitude = std::sqrt((x*x) + (y*y) + (z*z));  
            dir = make_float3( x/magnitude, y/magnitude, z/magnitude);
             
         }
     
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

 
  
   params.hits[linear_idx].t = lowestLen;

      
  } //end of if for icell

} //end of raygen function



extern "C" __global__ void __miss__miss(){

   optixSetPayload_0(float_as_int(FLT_MAX)); //set to a large number 

}

extern "C" __global__ void __closesthit__mixlength(){
  
    const float t = optixGetRayTmax(); //get t value from OptiX function 
 
    optixSetPayload_0(float_as_int(t));   //assign payload
}


