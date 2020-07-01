
//Contains all the functions to be used in OptixRayTrace


#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <vector>
#include <time.h>


#include <curand.h>
#include <curand_kernel.h>

#include "OptixRayTrace.h"

extern "C" {
__constant__ Params params; //should match var name in initPipeline
}


__forceinline__ __device__ float random(unsigned int seed){
       curandState_t state;
  //     curand_init(seed, 0, 0, &state);
       curand_init(129, 0, 0, &state);
       //return curand_uniform(&state);

       

       return (curand_uniform(&state)*2.0f) - 1.0;
}

__forceinline__ __device__ float3 randDir(){
   float x, y, z;

   unsigned int seed = clock();
   x = random(seed);
   seed = clock()*100 + seed;
   y = random(seed);
   seed = clock()*50;
   z = random(seed);

   return make_float3(x,y,z);
}


extern "C" __global__ void __raygen__from_cell(){

  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  
  const uint32_t linear_idx = idx.x + idx.y*(dim.x-1) + idx.z*(dim.y-1)*(dim.x-1);




  uint32_t t;

   
/*
for(int i = 0; i < params.numSamples; i++){
        float3 dir = randDir();
        printf("Random dir in .cu = <%f,%f,%f>\n",dir.x, dir.y, dir.z );
}
*/


  if(params.icellflagArray[linear_idx] == 1){

   float temp = FLT_MAX; //starting point


float3 cardinal[5]{
       make_float3(0,0,-1),
       make_float3(1,0, 0),
       make_float3(-1,0,0),
       make_float3(0,1,0),
       make_float3(0,-1,0)
};

   
 float3 origin = make_float3((idx.x+0.5)*params.dx, (idx.y+0.5)*params.dy, (idx.z+0.5)*params.dz);
float3 dir;

   for(int i = 0; i < params.numSamples; i++){


   
     if(i < 5){
        dir = cardinal[i];
     }else{
        dir = randDir();
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


if(int_as_float(t) < temp){

temp = int_as_float(t);
}


} //end of for loop

 //     float temp_t;
   //   temp_t = int_as_float(t);
      // printf("In .cu, t = %f, %i\n", temp_t, t);

      Hit hit;
      //hit.t = int_as_float(t);
hit.t = temp;
//hit.t = t;
      
      params.hits[linear_idx] = hit;

      
  } //end of if for icell

} //end of raygen function

extern "C" __global__ void __miss__miss(){



   optixSetPayload_0(float_as_int(FLT_MAX)); //need to set to a large number 


//MissData* m_data = reinterpret_cast<MissData *>(optixGetSbtDataPointer());
//optixSetPayload_0(m_data->missNum); //need to set to a large number 

    //optixSetPayload_0(float_as_int(5)); //test value 

}

extern "C" __global__ void __closesthit__mixlength(){

  //HitGroupData *rt_data = (HitGroupData *)optixGetSbtDataPointer();
  const uint32_t t = optixGetRayTmax();

  //optixSetPayload_0(float_as_int(t));

  optixSetPayload_0(float_as_int(t));  //test value

}


