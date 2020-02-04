/*
 *This class uses OptiX 7.0
 */
#include <cuda_gl_interlop.h>
#include <cuda_runtime.h>


#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include <vector>
#include <iomanip>
#include <iostream>
#include <string>

#include "RayTraceInterface.h"
#include "Vec3D.h"
#include "Vector3D.h"
#include "Triangle.cpp"

struct RayTracingState{
   OptixDeviceContext context = 0;
   //CUDA stream needed?
   
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
   //another var to params needed?
   OptixShaderBindingTable sbt = {};
};

template <typename T>
struct Record{
   __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct Params{
   OptixTraversableHandle handle;
   Ray* rays;
   Hit* hits;
};

struct Vertex{
   float x, y, z;
};

class OptixRayTrace : public RayTraceInterface{
  public:
   void buildAS(RayTracingState& state, std::vector<Triangle*> tris);
   void calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const vector<int> &icellflag, std::vector<double> &mixingLengths);

  private:
//TODO: add typedef Record<...> recordName;
   void createModule(RayTracingState& state);
   void createProgramGroups(RayTracingState& state);
   void createPipeline(RayTraceingState& state);
   void createSBT(RayTracingState& state);
   void initLaunchParams(RayTracingState& state);
   void cleanState(RayTracingState& state);

   /*
    *Helper function to convert vector<Triangle*> to array<float, 3>
    */
   void convertVecMeshType(std::vector<Triangle*> &tris, std::vector<Vertex> &trisArray);
   
};
