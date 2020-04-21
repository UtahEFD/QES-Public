/*
 *This class uses OptiX 7.0
 */

//#include <cuda_gl_interlop.h>

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




/*! simple wrapper for creating, and managing a device-side CUDA
  buffer */
struct CUDABuffer {
   inline CUdeviceptr d_pointer() const
   { return (CUdeviceptr)d_ptr; }

   //! re-size buffer to given number of bytes
   void resize(size_t size)
   {
      if (d_ptr) free();
      alloc(size);
   }

   //! allocate to given number of bytes
   void alloc(size_t size)
   {
      assert(d_ptr == nullptr);
      this->sizeInBytes = size;
      CUDA_CHECK(cudaMalloc( (void**)&d_ptr, sizeInBytes));
   }

   //! free allocated memory
   void free()
   {
      CUDA_CHECK(cudaFree(d_ptr));
      d_ptr = nullptr;
      sizeInBytes = 0;
   }

   template<typename T>
   void alloc_and_upload(const std::vector<T> &vt)
   {
      alloc(vt.size()*sizeof(T));
      upload((const T*)vt.data(),vt.size());
   }

   template<typename T>
   void upload(const T *t, size_t count)
   {
      std::cout<<"sizeInBytes in upload = "<<sizeInBytes<<std::endl;
      std::cout<<"size of T ( "<<typeid(T).name()<<" ) in upload = "<<sizeof(T)<<std::endl;
      assert(d_ptr != nullptr);
      assert(sizeInBytes == count*sizeof(T));
      CUDA_CHECK(cudaMemcpy(d_ptr, (void *)t,
                            count*sizeof(T), cudaMemcpyHostToDevice));
   }

   template<typename T>
   void download(T *t, size_t count)
   {
      assert(d_ptr != nullptr);
      assert(sizeInBytes == count*sizeof(T));
      CUDA_CHECK(cudaMemcpy((void *)t, d_ptr,
                            count*sizeof(T), cudaMemcpyDeviceToHost));
   }

   size_t sizeInBytes { 0 };
   void  *d_ptr { nullptr };
};


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

//struct OptixRay;
struct OptixRay{
   int flag; //test val
   //Vertex origin;
   float3 origin;
   float tmin;
   //Vertex dir;
   float3 dir;
   float tmax;
   bool isRay; //flag value; non-air cells = 0; air cells = 1;

   friend std::ostream& operator<<(std::ostream& os, const OptixRay& ray){
      os<<"Origin = <"
        <<ray.origin.x<<", "<<ray.origin.y<<", "<<ray.origin.z<<">"
        <<"\tDir = <"
        <<ray.dir.x<<", "<<ray.dir.y<<", "<<ray.dir.z<<">";
      return os;
   };
};
 
struct Hit{
   float t; 
};

struct Params{
   OptixTraversableHandle handle;
   OptixRay* rays;
   Hit* hits;
   int flag; //test value
   OptixRay* testRays;
   int sizeRays;
   int sizeIcell;
   OptixRay testOptixRay; 
//   int count; //temp val

//int * icellflagArray; //change to this later
};

struct RayGenData{
   //Vertex origin;
   //Vertex dir;
   float3 origin;
   float3 dir;
   float tmin = 0.0;
   float tmax;
//float dx, dy, dz; //change to Vertex type?

   
};

struct MissData{
   float largeNum; //should set value to large number 
};

struct HitGroupData{
   float t; //mixlength 
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
//   CUdeviceptr d_rays;   //holds params.rays
   CUDABuffer d_hits;    //holds params.hits
   CUDABuffer paramsBuffer;
   CUDABuffer testRays_d; //test array

   OptixShaderBindingTable sbt = {};

   int samples_per_cell;  //can change to bigger type value if needed 
   int num_cells;         //number of air cells 


   float nx, ny, nz;  //dim var passed in as params to mixLength
};


class OptixRayTrace{
  public:

   OptixRayTrace(std::vector<Triangle*> tris);
   void buildAS(std::vector<Triangle*> tris);
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
 
   void initOptix(); //checks for errors 
   void createContext();
   void createModule();
   void createProgramGroups();
   void createPipeline();
   void createSBT();
   void launch();
   void cleanState();

    /*
    *Helper function to convert vector<Triangle*> to array<float, 3>
    */
   void convertVecMeshType(std::vector<Triangle*> &tris, std::vector<Vertex> &trisArray);
   

   
   /*
    *Initializes Ray* rays based on the given cell data 
    */
   void initParams(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag);

   size_t roundUp(size_t x, size_t y);
   

   //ptx string definition (this is a hardcoded version, since there
   //is only one .cu file
   //std::string ptx = (std::string) PTX_DIR + "/target_name_generated_OptixRayTrace.cu.ptx";
   //std::string ptx = (std::string) PTX_DIR + "/cuda_compile_ptx_generated_OptixRayTrace.cu.ptx";
   std::string ptx; 
};
