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

//printf("########### Testing from inside raygen in .cu file #################\n");



  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  //const uint32_t linear_idx = (idx.z*dim.x*dim.y) + (idx.y*dim.x) + idx.x;


   const uint32_t linear_idx = idx.x + idx.y*(dim.x-1) + idx.z*(dim.y-1)*(dim.x-1);

  uint32_t t;

/* printf("testOptixRay origin = <%d,%d,%d>\n",
params.testOptixRay.origin.x, params.testOptixRay.origin.y, params.testOptixRay.origin.z );
*/

//printf("testOptixRay.flag = %d\n", params.testOptixRay.flag);

//   params.count++;  //temp val

  //printf("I");
 // printf("#### In raygen.cu: ray: \n");
 
//  printf("params.rays[linear_idx].isRay = %i\n", params.rays[linear_idx].isRay);

  //printf("params.rays[linear_idx].tmin.x %d\n", params.rays[linear_idx].tmin);
  //printf("params.rays[linear_idx].origin.x %d\n", params.rays[linear_idx].origin.x);
  //printf("/n");

  //printf("linear_idx: ");
  //printf("%i\n", linear_idx);

//float3 test = make_float3(0,0,-1);
//if(params.rays[linear_idx].dir.z < -1){
  //printf("The rays is initialized correctly in .cu");
//}

//printf("in .cu, params flag = %i\n", params.flag);


//printf("testRays @ index %i dir z val = %d\n", linear_idx, params.testRays[linear_idx].dir.z);
/*printf("testRays @ index %i dir z val = <%d,%d,%d>\n", linear_idx,
                 params.testRays[linear_idx].dir.x,
                 params.testRays[linear_idx].dir.y,
                 params.testRays[linear_idx].dir.z);

*/
/*
printf("testRays @ index 0 dir z val = <%d,%d,%d>\n",
                 params.testRays[0].dir.x,
                 params.testRays[0].dir.y,
                 params.testRays[0].dir.z);
*/




params.hits[linear_idx].t = 400;  //test val for hits
//params.flag = 40;



/*

//single direction version start 
  if(params.rays[linear_idx].isRay){   //non-air cells will have a flag = 0

  printf("isRay\n");



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


      //Hit hit;
      //hit.t = int_as_float(t);
      //hit.t = 45; //test val

      //float temp = optixGetPayload_0();
//printf("t");
//printf("%d\n",t);
//printf("Get t %d\n", t);
       
       //params.hits[0].t = 45; //test

  //    params.hits[linear_idx] = hit;
} //end of if 

printf("I");

*/

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
printf("miss");

   optixSetPayload_0(float_as_int(FLT_MAX)); //need to set to a large number 

}

extern "C" __global__ void __closesthit__mixlength(){
HitGroupData *rt_data = (HitGroupData *)optixGetSbtDataPointer();
const uint32_t t = optixGetRayTmax();


printf("hit");

optixSetPayload_0(float_as_int(t));

 
}


