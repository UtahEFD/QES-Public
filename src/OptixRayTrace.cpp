#include "OptixRayTrace.h"



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
         strStream << "CUDA error on synchronize with error '"  \
                   << cudaGetErrorString(error)                 \
                   <<"' ("__FILE__<<":"<<__LINE__<<")\n";       \
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

OptixRayTrace::OptixRayTrace(std::vector<Triangle*> tris){
   OPTIX_CHECK(optixInit());

   createContext();

   buildAS(tris);

}

static void context_log_cb(unsigned int level, const char* tag,
                           const char* message, void* /*cbdata*/){

   std::cout<<"Enters context log call back"<<std::endl;

   std::cerr<<"[level, tag, message] = ["
            <<level<<", "<<tag<<", "<<message<<"]"<<"\n";
}

void OptixRayTrace::createContext(){

   std::cout<<"Enters createContext()"<<std::endl;

   CUDA_CHECK(cudaFree(0));

   OptixDeviceContext context;
   CUcontext cuCtx = 0; //0 = current context
   OptixDeviceContextOptions  options = {};
   options.logCallbackFunction = &context_log_cb;
   options.logCallbackLevel = 4;

   OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

   state.context = context;


}


size_t OptixRayTrace::roundUp(size_t x, size_t y){
   return ((x + y -1) / y) * y;
}


void OptixRayTrace::buildAS(std::vector<Triangle*> tris){

   std::cout<<"Enters buildAS"<<std::endl;


   OptixAccelBuildOptions accel_options = {};
   accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
   accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

//need to translate tris --> to a vector array of measurable size type
   std::vector<Vertex> trisArray(tris.size()*3);  //each triangle 3 Vertex-es
   convertVecMeshType(tris, trisArray);

   const size_t tris_size = sizeof(Vertex)*trisArray.size();
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_tris), tris_size));

   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_tris),
                         trisArray.data(),
                         tris_size,
                         cudaMemcpyHostToDevice)
              );



   std::cout<<"In buildAS, finishes memory allocation for list of triangles"<<std::endl;


   const uint32_t triangle_input_flags[1] ={OPTIX_GEOMETRY_FLAG_NONE};
//could add flag to disable anyhit (above)

   OptixBuildInput triangle_input  = {};
   triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
   triangle_input.triangleArray.vertexBuffers = &state.d_tris;
   triangle_input.triangleArray.numVertices = static_cast<uint32_t>(trisArray.size());
   triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
   triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
   triangle_input.triangleArray.flags = triangle_input_flags;
   triangle_input.triangleArray.numSbtRecords = 1;


   std::cout<<"In buildAS, finishes init options for triangle_input"<<std::endl;




   OptixAccelBufferSizes gas_buffer_sizes;
   OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context,
                                            &accel_options,
                                            &triangle_input,
                                            1,
                                            &gas_buffer_sizes)
               );

   CUdeviceptr d_temp_buffer_gas;
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas),
                         gas_buffer_sizes.tempSizeInBytes)
              );

   CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
   size_t compactedSizeOffset = roundUp(gas_buffer_sizes.outputSizeInBytes, 8ull);
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
                         compactedSizeOffset+8)
              );

   OptixAccelEmitDesc emit_property= {};
   emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
   emit_property.result =
      (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size
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
                               1)
               );

   CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
   CUDA_CHECK(cudaFree((void*)state.d_tris));

   size_t compacted_gas_size;
   CUDA_CHECK(cudaMemcpy(&compacted_gas_size,
                         (void*)emit_property.result, sizeof(size_t), cudaMemcpyDeviceToHost)
              );

   if(compacted_gas_size < gas_buffer_sizes.outputSizeInBytes){
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**> (&state.d_gas_output_buffer),
                            compacted_gas_size)
                 );

      OPTIX_CHECK(optixAccelCompact(state.context,
                                    0,
                                    state.gas_handle,
                                    state.d_gas_output_buffer,
                                    compacted_gas_size,
                                    &state.gas_handle)
                  );
      CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
   }else{
      state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
   }


   std::cout<<"In buildAS, finsihes building"<<std::endl;

}

void OptixRayTrace::convertVecMeshType(std::vector<Triangle*> &tris, std::vector<Vertex> &trisArray){

   std::cout<<"\tconverting Triangle mesh to Optix Triangles"<<std::endl;

   Vertex tempVertexA, tempVertexB, tempVertexC;
   for(int i = 0; i < tris.size(); i++){ //get access to the Triangle at index
      tempVertexA.x = (*(tris[i]->a))[0];
      tempVertexA.y = (*(tris[i]->a))[1];
      tempVertexA.z = (*(tris[i]->a))[2];

      tempVertexB.x = (*(tris[i]->b))[0];
      tempVertexB.y = (*(tris[i]->b))[1];
      tempVertexB.z = (*(tris[i]->b))[2];

      tempVertexC.x = (*(tris[i]->c))[0];
      tempVertexC.y = (*(tris[i]->c))[1];
      tempVertexC.z = (*(tris[i]->c))[2];

      trisArray.push_back(tempVertexA);
      trisArray.push_back(tempVertexB);
      trisArray.push_back(tempVertexC);

   }
}


void OptixRayTrace::calculateMixingLength(int numSamples, int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag, std::vector<double> &mixingLengths){
   state.samples_per_cell = numSamples;

   createModule();

   std::cout<<"In calculateMixingLength, createModule() done"<<std::endl;

   createProgramGroups();

   std::cout<<"In calculateMixingLength, createProgramGroups() done"<<std::endl;

   createPipeline();

   std::cout<<"In calculateMixingLength, createPipeline() done"<<std::endl;


   createSBT();


   std::cout<<"In calculateMixingLength, createSBT() done"<<std::endl;


   initParams(dimX, dimY, dimZ, dx, dy, dz, icellflag);

   std::cout<<"In calculateMixingLength, initParams() done"<<std::endl;


   launch();

   std::cout<<"In calculateMixingLength, launch() done"<<std::endl;


   //Hits should be init
   for(int i = 0; i < state.num_cells; i++){
      mixingLengths.push_back(state.params.hits[i].t);
      std::cout<<"Hit at index "<<i<<" = "<<state.params.hits[i].t<<std::endl;
   }

}

void OptixRayTrace::initParams(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag){
   //memory allocation
   OptixRay* rays_d;



   //TODO: optimize
   //Really, really inefficient way of getting the number of cells to
   //set the ray size
   //init ray data

   int numAirCells = 0;

   /*for(int k = 0; k < dimZ -1; k++){
     for(int j = 0; j < dimY -1; j++){
     for(int i = 0; i < dimX -1; i++){
     int icell_idx = i + j*(dimX-1) + k*(dimY -1)*(dimX-1);

     if(icellflag[icell_idx] == 1){ //only want air cells
     numCells++;
     }
     }
     }
     }*/

   //std::cout<<"In initParams(), finished counting air cells:"<<numCells<<std::endl;

   std::cout<<"In initParams(), icellflag size = "<<icellflag.size()<<std::endl;
   //OptixRay rays_d[icellflag.size()];
   //size_t rays_size_in_bytes = sizeof(OptixRay)*numCells;
   size_t rays_size_in_bytes = sizeof(OptixRay)*icellflag.size();

   //CUDA_CHECK(cudaMalloc(&rays_d, rays_size_in_bytes));
   rays_d = (OptixRay*) malloc(rays_size_in_bytes);

   std::cout<<"In initParams(), init ray_size_in_bytes"<<std::endl;

   Hit* hits_d = 0;
//   size_t hits_size_in_bytes = sizeof(Hit)*numCells;
   size_t hits_size_in_bytes = sizeof(Hit)*icellflag.size();

   //CUDA_CHECK(cudaMalloc(&hits_d, hits_size_in_bytes));
   hits_d = (Hit*) malloc(hits_size_in_bytes);
   std::cout<<"In initParams(), init hits_size_in_bytes"<<std::endl;

   std::cout<<"size of rays_d = "<<sizeof(rays_d)/sizeof(rays_d[0])<<std::endl;

   //init ray data
   for(int k = 0; k < dimZ -1; k++){
      for(int j = 0; j < dimY -1; j++){
         for(int i = 0; i < dimX -1; i++){
            int icell_idx = i + j*(dimX-1) + k*(dimY -1)*(dimX-1);


            if(icellflag[icell_idx] == 1){ //only want air cells

               rays_d[icell_idx].origin = {(i+0.5)*dx,(j+0.5)*dy,(k+05)*dz};
               //rays_d[icell_idx].origin = make_float3((i+0.5)*dx,(j+0.5)*dy,(k+05)*dz);

               //(rays_d+icell_idx)->origin = {(i+0.5)*dx,(j+0.5)*dy,(k+05)*dz};


               rays_d[icell_idx].tmin = 0.0f;

               rays_d[icell_idx].dir = {0,0,-1}; //set to bottom direction for now
               rays_d[icell_idx].tmax = std::numeric_limits<float>::max();

               numAirCells++;
            }
         }
      }
   }

   std::cout<<"In initParams(), finished init ray data"<<std::endl;


   //init params
   state.params.handle = state.gas_handle;
   state.params.rays = rays_d;
   state.params.hits = hits_d;
   state.num_cells = numAirCells;
   std::cout<<"In initParams(), updated state params"<<std::endl;

}

extern "C" char embedded_ptx_code[];

void OptixRayTrace::createModule(){

   //module compile options
   OptixModuleCompileOptions module_compile_options = {};
   module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
   module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
   module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

   //pipeline compile options
   state.pipeline_compile_options.usesMotionBlur = false;
   state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
   state.pipeline_compile_options.numPayloadValues = 1;
   state.pipeline_compile_options.numAttributeValues = 2;
   state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

   state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";


   //OptiX error reporting
   char log[2048]; size_t sizeof_log = sizeof(log);


   ptx = embedded_ptx_code;
   // std::cout<<"PTX String: "<<ptx<<std::endl;

   OPTIX_CHECK(optixModuleCreateFromPTX(
                  state.context,
                  &module_compile_options,
                  &state.pipeline_compile_options,
                  ptx.c_str(),
                  ptx.size(),
                  log,
                  &sizeof_log,
                  &state.ptx_module)
               );
}

void OptixRayTrace::createProgramGroups(){

   //OptiX error reporting var
   char log[2048];
   size_t sizeof_log = sizeof(log);

   //program group descriptions
   OptixProgramGroupOptions program_group_options = {};

   OptixProgramGroupDesc raygen_prog_group_desc = {};
   raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
   raygen_prog_group_desc.raygen.module = state.ptx_module;
   raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__from_cell";

   OPTIX_CHECK(optixProgramGroupCreate(state.context,
                                       &raygen_prog_group_desc,
                                       1,
                                       &program_group_options,
                                       log,
                                       &sizeof_log,
                                       &state.raygen_prog_group)
               );

   OptixProgramGroupDesc miss_prog_group_desc = {};
   miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
   miss_prog_group_desc.miss.module = state.ptx_module;
   miss_prog_group_desc.miss.entryFunctionName = "__miss__miss";

   OPTIX_CHECK(optixProgramGroupCreate(state.context,
                                       &miss_prog_group_desc,
                                       1,
                                       &program_group_options,
                                       log,
                                       &sizeof_log,
                                       &state.miss_prog_group)
               );

   OptixProgramGroupDesc hit_prog_group_desc = {};
   hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
   hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
   hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__mixlength";
   OPTIX_CHECK(optixProgramGroupCreate(state.context,
                                       &hit_prog_group_desc,
                                       1,
                                       &program_group_options,
                                       log,
                                       &sizeof_log,
                                       &state.hit_prog_group)
               );
}

void OptixRayTrace::createPipeline(){
   //OptiX error reporting var
   char log[2048];
   size_t sizeof_log = sizeof(log);

   OptixProgramGroup program_groups[3] = {
      state.raygen_prog_group,
      state.miss_prog_group,
      state.hit_prog_group
   };

   OptixPipelineLinkOptions pipeline_link_options = {};
   pipeline_link_options.maxTraceDepth = 1;
   pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
   pipeline_link_options.overrideUsesMotionBlur = false;

   OPTIX_CHECK(optixPipelineCreate(state.context,
                                   &state.pipeline_compile_options,
                                   &pipeline_link_options,
                                   program_groups,
                                   sizeof(program_groups)/sizeof(program_groups[0]),
                                   log,
                                   &sizeof_log,
                                   &state.pipeline)
               );
}

void OptixRayTrace::createSBT(){
   //raygen
   CUdeviceptr d_raygen_record = 0;

   const size_t raygen_record_size = sizeof(RayGenRecord);

   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

   RayGenRecord sbt_raygen;

   OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &sbt_raygen));

   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*> (d_raygen_record),
                         &sbt_raygen,
                         raygen_record_size,
                         cudaMemcpyHostToDevice)
              );


   std::cout<<"In createSBT(), raygen record created"<<std::endl;

   //miss
   CUdeviceptr d_miss_record = 0;

   const size_t miss_record_size = sizeof(MissRecord);

   MissRecord sbt_miss;

   OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &sbt_miss));

   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), miss_record_size));

   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*> (d_miss_record),
                         &sbt_miss,
                         miss_record_size,
                         cudaMemcpyHostToDevice)
              );


   std::cout<<"In createSBT(), miss record created"<<std::endl;

//hit
   CUdeviceptr d_hit_record = 0;
   const size_t hit_record_size = sizeof(HitGroupRecord);


   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), hit_record_size*RAY_TYPE_COUNT));

   HitGroupRecord hit_records[RAY_TYPE_COUNT];

   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hit_record),
                         hit_records,
                         hit_record_size*RAY_TYPE_COUNT,
                         cudaMemcpyHostToDevice)
              );

   std::cout<<"In createSBT(), hit record created"<<std::endl;

//update state
   state.sbt.raygenRecord = d_raygen_record;

   state.sbt.missRecordBase = d_miss_record;
   state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
   state.sbt.missRecordCount = 1;

   state.sbt.hitgroupRecordBase = d_hit_record;
   state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hit_record_size);
   state.sbt.hitgroupRecordCount = RAY_TYPE_COUNT;

   std::cout<<"In createSBT(), state records updated"<<std::endl;


}


void OptixRayTrace::launch(){
   //create the CUDA stream
   CUDA_CHECK(cudaStreamCreate(&state.stream));

   Params* d_params = 0;
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
   CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
                              &state.params,
                              sizeof(Params),
                              cudaMemcpyHostToDevice,
                              state.stream)
              );

   OPTIX_CHECK(optixLaunch(state.pipeline,
                           state.stream,
                           reinterpret_cast<CUdeviceptr>(d_params),
                           sizeof(Params),
                           &state.sbt,
                           state.num_cells,
                           1,//state.num_cells,//state.samples_per_cell,
                           1)
               );
   CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, state.num_cells*state.samples_per_cell));
   CUDA_SYNC_CHECK();
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
}

void OptixRayTrace::cleanState(){
   //destroy pipeline
   OPTIX_CHECK(optixPipelineDestroy(state.pipeline));

   //destroy program groups
   OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
   OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
   OPTIX_CHECK(optixProgramGroupDestroy(state.hit_prog_group));

   //destroy module
   OPTIX_CHECK(optixModuleDestroy(state.ptx_module));

   //destroy context
   OPTIX_CHECK(optixDeviceContextDestroy(state.context));

   //free cuda stuff
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.rays)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.hits)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
}
