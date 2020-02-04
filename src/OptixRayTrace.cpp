
OptixRayTrace::OptixRayTrace(vector<Triangle*> tris){
   RayTracingState rts;
   buildAS(rts.state, tris);
}

void OptixRayTrace::buildAS(RayTracingState& state, std::vector<Triangle*> tris){
   OptixAccelBuildOptions accel_options = {};
   accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
   accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

   //need to translate tris --> to a vector array of measurable size type
   std::vector<Vertex> trisArray(tris.size()*3);
   convertVecMeshType(&tris, &trisArray);

   const size_t tris_size = sizeof(Vertex)*trisArray.size();
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tris), tris_size));
   //TODO: make sure memory allocation is actually working
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(&d_tris),
                         &trisArray[0],
                         tris_size,
                         cudaMemcpyHostToDevice));

   const uint32_t triangle_input_flags[1] ={OPTIX_GEOMETRY_FLAG_NONE};
   //could add flag to disable anyhit (above)

   OptixBuildInput triangle_input  = {};
   triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
   triangle_input.vertexBuffers = &d_tris;
   triangle_input.numVertices = static_cast<uint32_t>(trisArray.size());
   triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
   triangle_input.triangleArray.vertextStrideInBytes = sizeof(Vertex);
   triangle_input.triangleArray.flags = triangle_input_flags;
   triangle_input.triangleArray.numSbtRecords = 1;

   OptixAccelBufferSizes gas_buffer_sizes;
   OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context,
                                            &accel_options,
                                            &triangle_input,
                                            1,
                                            &gas_buffer_sizes));
   CUdeviceptr d_temp_buffer_gas;
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas),
                         gas_buffer_sizes.tempSizeInBytes));


   CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
   size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
                         compactedSizeOffset+8));

   OptixAccelEmitDesc emit_property= {};
   emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
   emi_property.result =
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
                               &state.gas_handle,
                               &emitProperty,
                               1));

   CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
   CUDA_CHECK(cudaFree((void*)d_tris));

   size_t compacted_gas_size;
   CUDA_CHECK(cudaMemcpy(&compacted_gas_size,
                         (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

   if(compacted_gas_size < gas_buffer_sizes.outputSizeInBytes){
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>
                            (&state.d_gas_output_buffer), compacted_gas_size));

      OPTIX_CHECK(optixAccelCompact(state.context,
                                    0,
                                    state.gas_handle,
                                    state.d_gas_ouput_buffer,
                                    compacted_gas_size,
                                    &state.gas_handle));
      CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
   }else{
      state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
   }
}

void OptixRayTrace::convertVecMeshType(std::vector<Triangle*> &tris, std::vector<Vertex> &trisArray){
   Vertex tempVertexA, tempVertexB, tempVertexC;
   for(int i = 0; i < tris->size(); i++){ //get access to the Triangle at index

      tempVertexA->x = (*tris)[i]->a[0];
      tempVertexA->y = (*tris)[i]->a[1];
      tempVertexA->z = (*tris)[i]->a[2];

      tempVertexB->x = (*tris)[i]->b[0];
      tempVertexB->y = (*tris)[i]->b[1];
      tempVertexB->z = (*tris)[i]->b[2];

      tempVertexC->x = (*tris)[i]->c[0];
      tempVertexC->y = (*tris)[i]->c[1];
      tempVertexC->z = (*tris)[i]->c[2];

      trisArray->pushback(tempVertexA);
      trisArray->pushback(tempVertexB);
      trisArray->pushback(tempVertexC);

   }
}


void OptixRayTrace::calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const vector<int> &icellflag, vector<double> &mixingLengths){

}


void OptixRayTrace::createModule(RayTracingState& state){

   //module compile options
   OptixModuleCompileOptions module_compile_options = {};
   module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
   module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
   module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

   //pipeline compile options
   state.pipeline_compile_options.usesMotionBlur = false;
   //TODO: set state.pipeline_compile_options.traversableGraphFlags
   //TODO: set state.pipeline_compile_options.numPayloadValues
   //TODO: set state.pipeline_compile_options.numAttributeValues
   state.pipeline_compile_options.exceptionFlags = OPTIX_EXEPTION_FLAG_NONE;
   //can also set to OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW or
   //OPTIX_EXCEPTION_FLAG_DEBUG
   state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

   //TODO: create optixRayTracing.cu file to set string ptx to


   //OptiX error reporting
   char log[2048];
   size_t sizeof_log = sizeof(log);
   OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
                      state.context,
                      &module_compile_options,
                      &state.pipeline_compile_options,
                      ptx.c_str(),
                      ptx.size(),
                      log,
                      &sizeof_log,
                      &state.ptx_module));
}

void OptixRayTrace::createProgramGroups(RayTracingState& state){

   //OptiX error reporting var
   char log[2048];
   size_t sizeof_log = sizeof(log);

   //program group descriptions
   OptixProgramGroupOptions program_group_options = {};

   OptixProgramGroupDesc raygen_prog_group_desc = {};
   raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
   raygen_prog_group_desc.raygen.module = state.ptx_module;
   raygen_prog_group_desc.raygen.entryFunctionName =
      "__raygen__from_buffer";//could name something else

   OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context,
                                           &raygen_prog_group_desc,
                                           1,
                                           &program_group_options,
                                           log,
                                           &sizeof_log,
                                           &state.raygen_prog_group));

   OptixProgramGroupDesc miss_prog_group_desc = {};
   miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
   miss_prog_group_desc.miss.module = state.ptx_module;
   miss_prog_group_desc.miss.entryFunctionName = "__miss__";

   OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context,
                                           &miss_prog_group_desc,
                                           1,
                                           &program_group_options,
                                           log,
                                           &sizeof_log,
                                           &state.miss_prog_group));
//memset miss needed?
   OptixProgramGroupDesc hit_prog_group_desc = ();
   hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
   hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
   hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__";
   OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context,
                                           &hit_prog_group_desc,
                                           1,
                                           &program_group_options,
                                           log,
                                           &sizeof_log,
                                           &state.hit_prog_group));
   //last line may change
   //memset needed?

}

void OptixRayTrace::createPipeline(RayTracingState& state){
   //OptiX error reporting var
   char log[2048];
   size_t sizeof_log = sizeof(log);

   OptixProgramGroup program_groups[] = {
      //list needed program groups
   };

   OptixPipelineLinkOptions pipeline_link_options = {};
   //pipelink_link_options.maxTraceDepth =
   pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
   pipeline_link_options.ovrideUsesMotionBlur = false;

   OPTIX_CHECK_LOG(optixPipelineCreate(state.context,
                                       &state.pipeline_compile_options,
                                       &pipeline_link_options,
                                       program_groups,
                                       sizeof(program_groups)/sizeof(program_groups[0]),
                                       log,
                                       &sizeof_log,
                                       &state.pipeline));
   //if more than 1 pipeline then more code is needed (look at
   //ray casting example)

   //if like Pathtracer example, depts of more than 1 (presumably)
   //need to specify max traversial depth
}

void OptixRayTrace::createSBT(RayTracingState& state){
   //CUdeviceptr d_raygen_record =
   //const size_t raygen_record_size = sizeof();
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

   //will differ depending on needs

}

void OptixRayTrace::initLaunchParams(RayTracingState& state){
//will differ based on purposes
}

void OptixRayTrace::cleanState(RayTracing& state){
   //destroy pipelines
   //destory program groups
   //free CUDA stuff
   //destroy module
   OPTIX_CHECK(optixModuleDestory(state.ptx_module));
   OPTIX_CHECK(optixDeviceContextDestroy(state.context));
   //destory context
}
