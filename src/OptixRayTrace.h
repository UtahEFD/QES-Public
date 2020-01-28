/*
 *This class uses OptiX 7.0
 */
#include <cuda_runtime.h>
//TODO: other Cuda files?

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "RayTraceInterface.h"
#include "Ray.h"

struct RayTracingState{
   OptixDeviceContext context = 0;
   OptixPipelineCompileOptions pipeline_compile_options = {};
   Params params; //or = {}
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

class OptixRayTrace : public RayTraceInterface{
  public:
   void buildAS(vector<Triangle*> tris);
   void calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const vector<int> &icellflag, vector<double> &mixingLengths);

  private:
//TODO: add typedef Record<...> recordName;
   void createModule(RayTracingState& state);
   void createProgramGroups(RayTracingState& state);
   void createPipeline(RayTraceingState& state);
   void createSBT(RayTracingState& state);
   void initLaunchParams(RayTracingState& state);
   void cleanState(RayTracingState& state);
};
