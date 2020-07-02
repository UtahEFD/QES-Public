#include "OptixRayTrace.h"


#define TEST 0 //Set to true for ground-only AS


/*Initializes OptiX and creates the context
 *If not testing, the acceleration structure will be based off of the
 *provided list of Triangles
 */

OptixRayTrace::OptixRayTrace(std::vector<Triangle*> tris){

   initOptix();

   createContext();


   if(!TEST){
      buildAS(tris);
   }
}

static void context_log_cb(unsigned int level, const char* tag,
                           const char* message, void* /*cbdata*/){


   std::cerr<<"[level, tag, message] = ["
            <<level<<", "<<tag<<", "<<message<<"]"<<"\n";
}




/**
 *Creates and configures a optix device context for primary GPU device
 */

void OptixRayTrace::createContext(){

   OptixDeviceContext context;
   CUcontext cuCtx = 0; //0 = current context
   OptixDeviceContextOptions  options = {};
   options.logCallbackFunction = &context_log_cb;
   options.logCallbackLevel = 4;

   OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

   state.context = context;
}


/**
 *Initializes OptiX and confirms OptiX compatible devices are present
 *
 *@throws RuntimeException On no OptiX 7.0 compatible devices
 */
void OptixRayTrace::initOptix(){

//check for Optix 7 compatible devices
   CUDA_CHECK(cudaFree(0));

   int numDevices;
   cudaGetDeviceCount(&numDevices);
   if(numDevices == 0){
      throw std::runtime_error("No OptiX 7.0 compatible devices!");
   }

   OPTIX_CHECK(optixInit());


}

size_t OptixRayTrace::roundUp(size_t x, size_t y){
   return ((x + y -1) / y) * y;
}



/**
 *Non-test version of AS
 *builds acceleration structure with provided list of Triangles
 *
 *@param tris List of Triangle objects representing given terrain and buildings
 */

void OptixRayTrace::buildAS(std::vector<Triangle*> tris){

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





   const uint32_t triangle_input_flags[1] ={OPTIX_GEOMETRY_FLAG_NONE};
//could add flag to disable anyhit (above)

   OptixBuildInput triangle_input  = {};
   triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
   triangle_input.triangleArray.vertexBuffers = &state.d_tris;
   triangle_input.triangleArray.numVertices = static_cast<uint32_t>(trisArray.size());
   triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
//triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
   triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
   triangle_input.triangleArray.flags = triangle_input_flags;
   triangle_input.triangleArray.numSbtRecords = 1;






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

   CUDA_SYNC_CHECK(); //need?

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
      CUDA_SYNC_CHECK(); //need?

      CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
   }else{
      state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
   }


}





/**
 *Test version of AS
 *builds acceleration structure with 2 Triangles representing the
 *ground of the domain
 *
 */

void OptixRayTrace::buildAS(){

   std::cout<<"\033[1;31m You are using testing acceleration structure in OptiX \033[0m"<<std::endl;



   OptixAccelBuildOptions accel_options = {};
   accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
   accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

   const std::array<float3, 6> trisArray = {
      {
         {0,0,0},
         {state.nx*state.dx, 0, 0},
         {state.nx*state.dx, state.ny*state.dy, 0},

         {0,0,0},
         {state.nx*state.dx, state.ny*state.dy, 0},
         {0, state.ny*state.dy, 0}
      }
   };


   const size_t tris_size = sizeof(float3)*trisArray.size();
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_tris), tris_size));

   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_tris),
                         trisArray.data(),
                         tris_size,
                         cudaMemcpyHostToDevice)
              );





   const uint32_t triangle_input_flags[1] ={OPTIX_GEOMETRY_FLAG_NONE};
//could add flag to disable anyhit (above)

   OptixBuildInput triangle_input  = {};
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

   CUDA_SYNC_CHECK(); //need?

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
      CUDA_SYNC_CHECK(); //need?

      CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
   }else{
      state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
   }


}


/**
 *Converts the list of Traingle objects to a list of Vertex objects
 *This is for the purpose of OptiX and to not conflict with other
 *parts of the code
 *
 *@params tris The list of Triangle objects representing the buildings
 *        and terrain
 *@params trisArray The list of Vertex struct objects representing the
 *        converted list of Triangles
 */
void OptixRayTrace::convertVecMeshType(std::vector<Triangle*> &tris, std::vector<Vertex> &trisArray){
   int tempIdx = 0;
   for(int i = 0; i < tris.size(); i++){ //get access to the Triangle at index


      trisArray[tempIdx] = {(*(tris[i]->a))[0], (*(tris[i]->a))[1],(*(tris[i]->a))[2]};
      tempIdx++;
      trisArray[tempIdx] = {(*(tris[i]->b))[0], (*(tris[i]->b))[1], (*(tris[i]->b))[2]};
      tempIdx++;
      trisArray[tempIdx] = {(*(tris[i]->c))[0], (*(tris[i]->c))[1], (*(tris[i]->c))[2]};
      tempIdx++;

   }

}


/**Calculates the mixing length
 *
 *@param numSamples The probablistic sampling of per-cell launch
 *       directions
 *@param dimX Domain info in the x plane
 *@param dimY Domain info in the y plane
 *@param dimZ Domain info in the z plane
 *@param dx Grid info in the x plane
 *@param dy Grid info in the y plane
 *@param dz Grid info in the z plane
 *@param icellflag Cell type
 *@param mixingLengths Array of mixinglengths for all cells that will be updated
 */

void OptixRayTrace::calculateMixingLength(int numSamples, int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag, std::vector<double> &mixingLengths){
   state.samples_per_cell = numSamples;

   state.params.numSamples = numSamples;


   state.nx = dimX;
   state.ny = dimY;
   state.nz = dimZ;

   state.dx = dx;
   state.dy = dy;
   state.dz = dz;


   if(TEST){
      buildAS(); /*for test AS version*/
   }


   createModule();


   createProgramGroups();


   createPipeline();



   createSBT();




   state.paramsBuffer.alloc(sizeof(state.params));


   initParams(dimX, dimY, dimZ, dx, dy, dz, icellflag);


   launch();


   std::vector<Hit> hitList(icellflag.size());
   state.d_hits.download( hitList.data(), icellflag.size() );

   for(int i = 0; i < icellflag.size(); i++){
      //std::cout << "ml2: " << hitList[i].t << std::endl;
      mixingLengths[i] = hitList[i].t;
   }


   cleanState();


}



/**
 *Initialize members of the Params stuct for state.params
 *
 *@param dimX Domain info in the x plane
 *@param dimY Domain info in the y plane
 *@param dimZ Domain info in the z plane
 *@param dx Grid info in the x plane
 *@param dy Grid info in the y plane
 *@param dz Grid info in the z plane
 *@param icellflag Cell type
 */
void OptixRayTrace::initParams(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag){



   //allocate memory for hits (container for the t info from device to host)
   size_t hits_size_in_bytes = sizeof(Hit)*icellflag.size();
   state.d_hits.alloc(hits_size_in_bytes);

   //acceleration structure handle
   state.params.handle = state.gas_handle;


   //allocate memory to device-side icellflag memory
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.icellflagArray_d), icellflag.size()*sizeof(int)));


   //copy std::vector data to int array

   int *tempArray = (int*) malloc(icellflag.size()*sizeof(int));

   for(int i = 0; i < icellflag.size(); i++){
      tempArray[i] = icellflag[i];
   }

   //copy data from std::vector icellflag to device-side memory
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.icellflagArray_d),
                         reinterpret_cast<void*>(tempArray),
                         icellflag.size()*sizeof(int),
                         cudaMemcpyHostToDevice));


   //assign params icellflag pointer to point to icellflagArray_d
   state.params.icellflagArray = (int *) state.icellflagArray_d;


   //init params dx, dy, dz
   state.params.dx = dx;
   state.params.dy = dy;
   state.params.dz = dz;


}


/**
 *Creates OptiX module from generated ptx file
 *
 */

extern "C" char embedded_ptx_code[]; //The generated ptx file

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

   state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";



   //OptiX error reporting
   char log[2048]; size_t sizeof_log = sizeof(log);


   ptx = embedded_ptx_code;


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


/**
 *Creates OptiX program groups
 *Three groups: raygen, miss, and closest hit
 */
void OptixRayTrace::createProgramGroups(){

   //OptiX error reporting var
   char log[2048];

   //program group descriptions
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
                                       &state.raygen_prog_group)
               );

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
                                       &state.miss_prog_group)
               );

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
                                       &state.hit_prog_group)
               );


}


/**Creates OptiX pipeline
 */
void OptixRayTrace::createPipeline(){
   //OptiX error reporting var
   char log[2048];

   OptixProgramGroup program_groups[3] = {
      state.raygen_prog_group,
      state.miss_prog_group,
      state.hit_prog_group
   };

   OptixPipelineLinkOptions pipeline_link_options = {};
   pipeline_link_options.maxTraceDepth = 1;
   pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
   pipeline_link_options.overrideUsesMotionBlur = false;

   size_t sizeof_log = sizeof(log);

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


/**Creates OptiX SBT record
 *
 */

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



//hit

   CUdeviceptr d_hit_record = 0;
   const size_t hit_record_size = sizeof(HitGroupRecord);

   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), hit_record_size));

   HitGroupRecord sbt_hit;
   OPTIX_CHECK(optixSbtRecordPackHeader(state.hit_prog_group, &sbt_hit));
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hit_record),
                         &sbt_hit,
                         hit_record_size,
                         cudaMemcpyHostToDevice
                         ));




//update state
   state.sbt.raygenRecord = d_raygen_record;

   state.sbt.missRecordBase = d_miss_record;

   state.sbt.missRecordStrideInBytes = sizeof(MissRecord);
   state.sbt.missRecordCount = 1;

   state.sbt.hitgroupRecordBase = d_hit_record;

   state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);

   state.sbt.hitgroupRecordCount = 1;


}


/*Launches OptiX
 */

void OptixRayTrace::launch(){
   //create the CUDA stream
   CUDA_CHECK(cudaStreamCreate(&state.stream));

   state.params.hits = (Hit *) state.d_hits.d_ptr;

   state.paramsBuffer.upload(&state.params, 1);



   OPTIX_CHECK(optixLaunch(state.pipeline,
                           state.stream,
                           state.paramsBuffer.d_pointer(),
                           state.paramsBuffer.sizeInBytes,
                           &state.sbt,
                           state.nx,
                           state.ny,
                           state.nz
                           )
               );


   CUDA_SYNC_CHECK();  //this line is necessary!!


}


/**Frees up memory from state variables
 */
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
   //OPTIX_CHECK(optixDeviceContextDestroy(state.context));

   //free cuda stuff
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.rays)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.hits)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
}
