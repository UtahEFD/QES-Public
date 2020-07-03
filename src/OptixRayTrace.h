/**
 *OptiX version of mixinglength
 *OptiX Version: 7.0
 *
 */


#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>

#include <optix_stubs.h>

#include <stdexcept>
#include <sstream>
#include <vector>
#include <iomanip>
#include <iostream>
#include <string>
#include <limits>
#include <assert.h>
#include <typeinfo>

#include "Triangle.h"

#define RAY_TYPE_COUNT 2
#define RAY_TYPE_RADIENCE 0



#define CUDA_CHECK(call)                                        \
   do{                                                          \
      cudaError_t error = call;                                 \
      if(error != cudaSuccess){                                 \
         std::stringstream strStream;                           \
         strStream <<"CUDA call ( "<<#call                      \
                   << " ) failed with error: '"                 \
                   <<cudaGetErrorString(error)                  \
                   <<"' (" <<__FILE__<< ":"                     \
                   <<__LINE__<<")\n";                           \
         throw std::runtime_error(strStream.str().c_str());     \
      }                                                         \
   }while (0);


#define CUDA_SYNC_CHECK()                                       \
   do{                                                          \
      cudaDeviceSynchronize();                                  \
      cudaError_t error = cudaGetLastError();                   \
      if(error != cudaSuccess) {                                \
         std::stringstream strStream;                           \
         strStream << "CUDA error on synchronize with error "   \
                   << cudaGetErrorString(error)                 \
                   <<" ("__FILE__<<":"                          \
                   <<__LINE__<<")\n";                           \
         throw std::runtime_error(strStream.str().c_str());     \
      }                                                         \
   }while(0)

#define OPTIX_CHECK(call)                                       \
   do {                                                         \
      OptixResult res = call;                                   \
      if(res != OPTIX_SUCCESS){                                 \
         std::stringstream strStream;                           \
         strStream << optixGetErrorName(res) <<":"              \
                   <<"Optix call ( "<<#call                     \
                   <<" ) failed: " __FILE__":"                  \
                   <<__LINE__<<"\n";                            \
         throw std::runtime_error(strStream.str().c_str());     \
      }                                                         \
   }while(0)






template <typename T>
struct Record{
   __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
   T data;
};


struct Vertex{
   float x;
   float y;
   float z;

   friend std::ostream& operator<<(std::ostream& os, const Vertex& v){
      os<<"<"<<v.x<<", "<<v.y<<", "<<v.z<<">";
      return os;
   };
};


struct Hit{
   float t; 
};

struct Params{
   OptixTraversableHandle handle;
   Hit* hits;
   int* icellflagArray; //change to this later
   float dx, dy, dz;
   int numSamples;
};

struct RayGenData{
   //no data needed
};

struct MissData{
   //no data needed
};
   
struct HitGroupData{
   //no hit data needed
};



struct RayTracingState{
   OptixDeviceContext context = 0;
   CUstream stream = 0;
   
   OptixModule ptx_module  = 0;
  
   OptixPipeline pipeline = 0;
   OptixPipelineCompileOptions pipeline_compile_options = {};

   
   OptixTraversableHandle gas_handle = 0;
   CUdeviceptr d_gas_output_buffer = 0;
   CUdeviceptr d_tris;  //converted mesh list 

   OptixProgramGroup raygen_prog_group = 0;
   OptixProgramGroup miss_prog_group = 0;
   OptixProgramGroup hit_prog_group = 0;

   Params params = {};
   
   CUdeviceptr icellflagArray_d; //device memory 

   CUdeviceptr outputBuffer = 0; //buffer to read and write btw device and host 
   CUdeviceptr paramsBuffer = 0; //buffer to read parms info btw device and host

   OptixShaderBindingTable sbt = {};
   
   int nx, ny, nz;  //dim var passed in as params to mixLength
   float dx, dy, dz; 
};


class OptixRayTrace{
  public:

   OptixRayTrace(std::vector<Triangle*> tris);
   ~OptixRayTrace();
   
   void calculateMixingLength(int numSamples, int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag, std::vector<double> &mixingLengths);
   
  private:
   
   OptixRayTrace();  //cannot have an empty constructor (have to pass in a mesh to build)

    
   RayTracingState state; //needs to be accessable though out program


   //Cuda device context and & properties that pipeline will run on
   //(only used in createContext()
   CUcontext cudaContext;
   cudaDeviceProp deviceProps;
   
   typedef Record<RayGenData> RayGenRecord;
   typedef Record<MissData> MissRecord;
   typedef Record<HitGroupData> HitGroupRecord;
 
   void initOptix(); 
   void buildAS(); 
   void buildAS(std::vector<Triangle*> tris);
   void createContext();
   void createModule();
   void createProgramGroups();
   void createPipeline();
   void createSBT();
   void launch();
   void cleanState();

   /**
    *Helper function to convert vector<Triangle*> to array<float, 3>
    */
   void convertVecMeshType(std::vector<Triangle*> &tris, std::vector<Vertex> &trisArray);
   
   void initParams(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag);

   /**Helper function for rounding up
    */
   size_t roundUp(size_t x, size_t y);
   
   std::string ptx; 
};
