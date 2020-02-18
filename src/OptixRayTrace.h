/*
 *This class uses OptiX 7.0
 */

//#include <cuda_gl_interlop.h>

#pragma once
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <vector>
#include <iomanip>
#include <iostream>
#include <string>
#include <limits>

#include "Triangle.cpp"



template <typename T>
struct Record{
   __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
   T data;
};

struct Vertex{
   float x, y, z;
};

struct OptixRay{
   Vertex origin;
   float tmin;
   Vertex dir;
   float tmax;
};

struct Hit{
   float t; 
};

struct Params{
   OptixTraversableHandle handle;
   OptixRay* rays;
   Hit* hits;
};

struct RayGenData{
   Vertex origin;
   float dx, dy, dz; //change to Vertex type? 
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
   OptixPipelineCompileOptions pipeline_compile_options = {};

   OptixPipeline pipeline = 0;

   OptixTraversableHandle gas_handle = 0;
   CUdeviceptr d_gas_output_buffer = 0;
   CUdeviceptr d_tris;  //converted mesh list 

   OptixProgramGroup raygen_prog_group = 0;
   OptixProgramGroup miss_prog_group = 0;
   OptixProgramGroup hit_prog_group = 0;

   Params params = {};
   OptixShaderBindingTable sbt = {};

   int samples_per_cell;  //can change to bigger type value if needed 
   int num_cells;
};


class OptixRayTrace{
  public:
   OptixRayTrace(std::vector<Triangle*> tris);
   void buildAS(std::vector<Triangle*> tris);
   void calculateMixingLength(int numSamples, int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag, std::vector<double> &mixingLengths);
  private:
   OptixRayTrace();  //cannot have an empty constructor (have to pass in a mesh to build)

   RayTracingState state; //needs to be accessable though out program

   typedef Record<RayGenData> RayGenRecord;
   typedef Record<MissData> MissRecord;
   typedef Record<HitGroupData> HitGroupRecord;

   
   void createContext();
   void createModule();
   void createProgramGroups();
   void createPipeline();
   void createSBT();
   void initLaunchParams();
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
   std::string ptx = PTX_DIR + "OptixRayTrace.cu.ptx";
};
