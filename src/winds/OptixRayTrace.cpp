/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file OptixRayTrace.cpp
 * @brief :document this:
 */

#include "OptixRayTrace.h"
#include <optix_function_table_definition.h>

#define TEST 0// Set to true for ground-only AS
#define GEN_FILE 1// Generate mixing length output file for testing

OptixRayTrace::OptixRayTrace(std::vector<Triangle *> tris)
{
  initOptix();

  createContext();


  if (!TEST) {
    buildAS(tris);
  }
}


OptixRayTrace::~OptixRayTrace()
{
  OPTIX_CHECK(optixDeviceContextDestroy(state.context));
}


static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata*/)
{


  std::cerr << "[level, tag, message] = ["
            << level << ", " << tag << ", " << message << "]"
            << "\n";
}


void OptixRayTrace::createContext()
{

  OptixDeviceContext context;
  CUcontext cuCtx = 0;// 0 = current context
  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &context_log_cb;
  options.logCallbackLevel = 4;

  OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

  state.context = context;
}


void OptixRayTrace::initOptix()
{

  CUDA_CHECK(cudaFree(0));

  // check for Optix 7 compatible devices
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0) {
    throw std::runtime_error("No OptiX 7.0 compatible devices!");
  }

  // initialize OptiX
  OPTIX_CHECK(optixInit());
}


size_t OptixRayTrace::roundUp(size_t x, size_t y)
{
  return ((x + y - 1) / y) * y;
}


void OptixRayTrace::buildAS(std::vector<Triangle *> tris)
{

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  // need to translate tris --> to a vector array of measurable size type
  std::vector<Vertex> trisArray(tris.size() * 3);// each triangle 3 Vertex-es
  convertVecMeshType(tris, trisArray);

  const size_t tris_size = sizeof(Vertex) * trisArray.size();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_tris), tris_size));

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_tris),
    trisArray.data(),
    tris_size,
    cudaMemcpyHostToDevice));

  // set OptiX AS input types
  const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

  OptixBuildInput triangle_input = {};
  triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexBuffers = &state.d_tris;
  triangle_input.triangleArray.numVertices = static_cast<uint32_t>(trisArray.size());
  triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
  triangle_input.triangleArray.flags = triangle_input_flags;
  triangle_input.triangleArray.numSbtRecords = 1;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context,
    &accel_options,
    &triangle_input,
    1,
    &gas_buffer_sizes));

  CUdeviceptr d_temp_buffer_gas;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas),
    gas_buffer_sizes.tempSizeInBytes));

  CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
  size_t compactedSizeOffset = roundUp(gas_buffer_sizes.outputSizeInBytes, 8ull);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
    compactedSizeOffset + 8));

  OptixAccelEmitDesc emit_property = {};
  emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit_property.result =
    (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size
                  + compactedSizeOffset);

  OPTIX_CHECK(optixAccelBuild(state.context,
    0,
    &accel_options,
    &triangle_input,
    1,
    d_temp_buffer_gas,
    gas_buffer_sizes.tempSizeInBytes,
    d_buffer_temp_output_gas_and_compacted_size,
    gas_buffer_sizes.outputSizeInBytes,
    &state.gas_handle,
    &emit_property,
    1));

  CUDA_SYNC_CHECK();

  CUDA_CHECK(cudaFree((void *)d_temp_buffer_gas));
  CUDA_CHECK(cudaFree((void *)state.d_tris));


  // Compact the accelerated structure for efficiency
  size_t compacted_gas_size;
  CUDA_CHECK(cudaMemcpy(&compacted_gas_size,
    (void *)emit_property.result,
    sizeof(size_t),
    cudaMemcpyDeviceToHost));

  if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_output_buffer),
      compacted_gas_size));

    OPTIX_CHECK(optixAccelCompact(state.context,
      0,
      state.gas_handle,
      state.d_gas_output_buffer,
      compacted_gas_size,
      &state.gas_handle));

    CUDA_SYNC_CHECK();

    CUDA_CHECK(cudaFree((void *)d_buffer_temp_output_gas_and_compacted_size));
  } else {
    state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
  }
}


void OptixRayTrace::buildAS()
{

  std::cout << "\033[1;31m You are using testing acceleration structure in OptiX \033[0m" << std::endl;

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  // form 2 triangles representing the ground of the given space
  const std::array<float3, 6> trisArray = {
    { { 0, 0, 0 },
      { state.nx * state.dx, 0, 0 },
      { state.nx * state.dx, state.ny * state.dy, 0 },

      { 0, 0, 0 },
      { state.nx * state.dx, state.ny * state.dy, 0 },
      { 0, state.ny * state.dy, 0 } }
  };


  const size_t tris_size = sizeof(float3) * trisArray.size();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_tris), tris_size));

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_tris),
    trisArray.data(),
    tris_size,
    cudaMemcpyHostToDevice));


  // intialize OptiX AS values
  const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

  OptixBuildInput triangle_input = {};
  triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexBuffers = &state.d_tris;
  triangle_input.triangleArray.numVertices = static_cast<uint32_t>(trisArray.size());
  triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;

  triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
  triangle_input.triangleArray.flags = triangle_input_flags;
  triangle_input.triangleArray.numSbtRecords = 1;


  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context,
    &accel_options,
    &triangle_input,
    1,
    &gas_buffer_sizes));

  CUdeviceptr d_temp_buffer_gas;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas),
    gas_buffer_sizes.tempSizeInBytes));

  CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
  size_t compactedSizeOffset = roundUp(gas_buffer_sizes.outputSizeInBytes, 8ull);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
    compactedSizeOffset + 8));

  OptixAccelEmitDesc emit_property = {};
  emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit_property.result =
    (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size
                  + compactedSizeOffset);

  OPTIX_CHECK(optixAccelBuild(state.context,
    0,
    &accel_options,
    &triangle_input,
    1,
    d_temp_buffer_gas,
    gas_buffer_sizes.tempSizeInBytes,
    d_buffer_temp_output_gas_and_compacted_size,
    gas_buffer_sizes.outputSizeInBytes,
    &state.gas_handle,
    &emit_property,
    1));

  CUDA_SYNC_CHECK();

  CUDA_CHECK(cudaFree((void *)d_temp_buffer_gas));
  CUDA_CHECK(cudaFree((void *)state.d_tris));


  // Compact AS for efficiency
  size_t compacted_gas_size;
  CUDA_CHECK(cudaMemcpy(&compacted_gas_size,
    (void *)emit_property.result,
    sizeof(size_t),
    cudaMemcpyDeviceToHost));

  if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_output_buffer),
      compacted_gas_size));

    OPTIX_CHECK(optixAccelCompact(state.context,
      0,
      state.gas_handle,
      state.d_gas_output_buffer,
      compacted_gas_size,
      &state.gas_handle));
    CUDA_SYNC_CHECK();

    CUDA_CHECK(cudaFree((void *)d_buffer_temp_output_gas_and_compacted_size));
  } else {
    state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
  }
}


void OptixRayTrace::convertVecMeshType(std::vector<Triangle *> &tris, std::vector<Vertex> &trisArray)
{
  int tempIdx = 0;
  for (int i = 0; i < tris.size(); i++) {// get access to the Triangle at index


    trisArray[tempIdx] = { (*(tris[i]->a))[0], (*(tris[i]->a))[1], (*(tris[i]->a))[2] };
    tempIdx++;
    trisArray[tempIdx] = { (*(tris[i]->b))[0], (*(tris[i]->b))[1], (*(tris[i]->b))[2] };
    tempIdx++;
    trisArray[tempIdx] = { (*(tris[i]->c))[0], (*(tris[i]->c))[1], (*(tris[i]->c))[2] };
    tempIdx++;
  }
}


void OptixRayTrace::calculateMixingLength(int numSamples, int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag, std::vector<double> &mixingLengths)
{

  // Initialize variables used in called functions below
  state.params.numSamples = numSamples;

  state.nx = dimX;
  state.ny = dimY;
  state.nz = dimZ;

  state.dx = dx;
  state.dy = dy;
  state.dz = dz;


  // Check to see if building with test acceleration structure
  if (TEST) {
    buildAS();
  }


  // Create related OptiX structures
  createModule();

  createProgramGroups();

  createPipeline();

  createSBT();


  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.paramsBuffer), sizeof(state.params)));

  initParams(dimX, dimY, dimZ, dx, dy, dz, icellflag);


  // Launch OptiX
  launch();


  // Initialize mixingLengths array with t values returned from OptiX
  std::vector<Hit> hitList(icellflag.size());
  CUDA_CHECK(cudaMemcpy(hitList.data(),
    reinterpret_cast<void *>(state.outputBuffer),
    sizeof(Hit) * icellflag.size(),
    cudaMemcpyDeviceToHost));

  for (int i = 0; i < icellflag.size(); i++) {
    mixingLengths[i] = hitList[i].t;
  }

  if (GEN_FILE) {

    std::ofstream mixingLenOutputFile;
    if (mixingLenOutputFile.is_open()) {
      mixingLenOutputFile.close();
    } else {
      std::stringstream ss;
      ss << "mixingLength_s" << numSamples << ".csv";
      mixingLenOutputFile.open(ss.str());
    }


    for (int k = 0; k < dimZ - 1; k++) {
      for (int j = 0; j < dimY - 1; j++) {
        for (int i = 0; i < dimX - 1; i++) {

          int icell_idx = i + j * (dimX - 1) + k * (dimY - 1) * (dimX - 1);
          float3 center = make_float3((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz);

          mixingLenOutputFile << icell_idx << ", "
                              << center.x << ","
                              << center.y << ","
                              << center.z << ","
                              << mixingLengths[icell_idx] << std::endl;
        }
      }
    }

    /* for(int i = 0; i < icellflag.size(); i++){
       if(icellflag[i] == 1){
       mixingLenOutputFile<<i<<","<<mixingLengths[i]<<std::endl;
       }
       }
    */

    mixingLenOutputFile.close();
  }

  // free memory
  cleanState();
}


void OptixRayTrace::initParams(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag)
{

  // allocate memory for hits (container for the t info from device to host)
  size_t output_buffer_size_in_bytes = sizeof(Hit) * icellflag.size();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.outputBuffer), output_buffer_size_in_bytes));


  // Assign the acceleration structure to be passed to device
  state.params.handle = state.gas_handle;


  // Initialize icellflag array to be passed to device
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.icellflagArray_d), icellflag.size() * sizeof(int)));

  int *tempArray = (int *)malloc(icellflag.size() * sizeof(int));

  for (int i = 0; i < icellflag.size(); i++) {
    tempArray[i] = icellflag[i];
  }

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.icellflagArray_d),
    reinterpret_cast<void *>(tempArray),
    icellflag.size() * sizeof(int),
    cudaMemcpyHostToDevice));

  state.params.icellflagArray = (int *)state.icellflagArray_d;

  // init params dx, dy, dz (used to calculate cell centers in device)
  state.params.dx = dx;
  state.params.dy = dy;
  state.params.dz = dz;
}


extern "C" char embedded_ptx_code[];// The generated ptx file

void OptixRayTrace::createModule()
{

  // module compile options
  OptixModuleCompileOptions module_compile_options = {};
  module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

  // pipeline compile options
  //  state.pipeline_compile_options.usesMotionBlur = false;
  state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  state.pipeline_compile_options.numPayloadValues = 1;

  state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";


  // OptiX error reporting
  char log[2048];
  size_t sizeof_log = sizeof(log);


  // Grab the ptx string from the generated ptx file
  // This should be located at compile time in the "ptx" folder
  ptx = embedded_ptx_code;


  OPTIX_CHECK(optixModuleCreateFromPTX(
    state.context,
    &module_compile_options,
    &state.pipeline_compile_options,
    ptx.c_str(),
    ptx.size(),
    log,
    &sizeof_log,
    &state.ptx_module));
}


void OptixRayTrace::createProgramGroups()
{

  // OptiX error reporting var
  char log[2048];

  // program group descriptions
  OptixProgramGroupOptions program_group_options = {};

  OptixProgramGroupDesc raygen_prog_group_desc = {};
  raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_prog_group_desc.raygen.module = state.ptx_module;
  raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__from_cell";


  size_t sizeof_log = sizeof(log);

  OPTIX_CHECK(optixProgramGroupCreate(state.context,
    &raygen_prog_group_desc,
    1,
    &program_group_options,
    log,
    &sizeof_log,
    &state.raygen_prog_group));

  OptixProgramGroupDesc miss_prog_group_desc = {};
  miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_prog_group_desc.miss.module = state.ptx_module;
  miss_prog_group_desc.miss.entryFunctionName = "__miss__miss";

  sizeof_log = sizeof(log);

  OPTIX_CHECK(optixProgramGroupCreate(state.context,
    &miss_prog_group_desc,
    1,
    &program_group_options,
    log,
    &sizeof_log,
    &state.miss_prog_group));

  OptixProgramGroupDesc hit_prog_group_desc = {};
  hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
  hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__mixlength";

  sizeof_log = sizeof(log);

  OPTIX_CHECK(optixProgramGroupCreate(state.context,
    &hit_prog_group_desc,
    1,
    &program_group_options,
    log,
    &sizeof_log,
    &state.hit_prog_group));
}


void OptixRayTrace::createPipeline()
{
  // OptiX error reporting var
  char log[2048];

  OptixProgramGroup program_groups[3] = {
    state.raygen_prog_group,
    state.miss_prog_group,
    state.hit_prog_group
  };

  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth = 1;
  pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  // pipeline_link_options.overrideUsesMotionBlur = false;

  size_t sizeof_log = sizeof(log);

  OPTIX_CHECK(optixPipelineCreate(state.context,
    &state.pipeline_compile_options,
    &pipeline_link_options,
    program_groups,
    sizeof(program_groups) / sizeof(program_groups[0]),
    log,
    &sizeof_log,
    &state.pipeline));
}


void OptixRayTrace::createSBT()
{
  // raygen
  CUdeviceptr d_raygen_record = 0;

  const size_t raygen_record_size = sizeof(RayGenRecord);

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygen_record), raygen_record_size));

  RayGenRecord sbt_raygen;

  OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &sbt_raygen));

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_raygen_record),
    &sbt_raygen,
    raygen_record_size,
    cudaMemcpyHostToDevice));


  // miss
  CUdeviceptr d_miss_record = 0;

  const size_t miss_record_size = sizeof(MissRecord);

  MissRecord sbt_miss;

  OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &sbt_miss));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss_record), miss_record_size));

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_miss_record),
    &sbt_miss,
    miss_record_size,
    cudaMemcpyHostToDevice));


  // hit

  CUdeviceptr d_hit_record = 0;
  const size_t hit_record_size = sizeof(HitGroupRecord);

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hit_record), hit_record_size));

  HitGroupRecord sbt_hit;
  OPTIX_CHECK(optixSbtRecordPackHeader(state.hit_prog_group, &sbt_hit));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hit_record),
    &sbt_hit,
    hit_record_size,
    cudaMemcpyHostToDevice));


  // update state
  state.sbt.raygenRecord = d_raygen_record;
  state.sbt.missRecordBase = d_miss_record;
  state.sbt.missRecordStrideInBytes = sizeof(MissRecord);
  state.sbt.missRecordCount = 1;
  state.sbt.hitgroupRecordBase = d_hit_record;
  state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
  state.sbt.hitgroupRecordCount = 1;
}


void OptixRayTrace::launch()
{
  // create the CUDA stream
  CUDA_CHECK(cudaStreamCreate(&state.stream));


  // Assign output buffer to write from device to host
  state.params.hits = reinterpret_cast<Hit *>(state.outputBuffer);

  // Allocate memory to pass initialized params variable to device
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.paramsBuffer),
    reinterpret_cast<void *>(&state.params),
    1 * sizeof(Params),
    cudaMemcpyHostToDevice));

  // launch OptiX
  OPTIX_CHECK(optixLaunch(state.pipeline,
    state.stream,
    state.paramsBuffer,
    sizeof(Params),
    &state.sbt,
    state.nx - 1,
    state.ny - 1,
    state.nz - 1));


  CUDA_SYNC_CHECK();
}


void OptixRayTrace::cleanState()
{
  // destroy pipeline
  OPTIX_CHECK(optixPipelineDestroy(state.pipeline));

  // destroy program groups
  OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
  OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
  OPTIX_CHECK(optixProgramGroupDestroy(state.hit_prog_group));

  // destroy module
  OPTIX_CHECK(optixModuleDestroy(state.ptx_module));

  // destroy context
  // OPTIX_CHECK(optixDeviceContextDestroy(state.context));

  // free cuda stuff

  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.params.hits)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.missRecordBase)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.hitgroupRecordBase)));

  // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.outputBuffer)));
  // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.paramsBuffer)));
}
