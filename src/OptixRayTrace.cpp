
OptixRayTrace::OptixRayTrace(vector<Triangle*> tris){
   buildAS(tris);
}

void OptixRayTrace::buildAS(vector<Triangle*> tris){

}

void OptixRayTrace::calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const vector<int> &icellflag, vector<double> &mixingLengths){

}


void OptixRayTrace::createModule(RayTracingState& state){

   //module compile options
   OptixModuleCompileOptions module_compile_options = {};
   module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
   module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
   module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

   //pipeline states
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


}

void OptixRayTrace::createPipeline(RayTracingState& state){
   //OptiX error reporting var
   char log[2048];
   size_t sizeof_log = sizeof(log);


}

void OptixRayTrace::createSBT(RayTracingState& state){

}

void OptixRayTrace::initLaunchParams(RayTracingState& state){

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
