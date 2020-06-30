#include "OptixRayTrace.h"



OptixRayTrace::OptixRayTrace(std::vector<Triangle*> tris){
//OPTIX_CHECK(optixInit());

   initOptix();

   createContext();

   buildAS(tris);

}

static void context_log_cb(unsigned int level, const char* tag,
                           const char* message, void* /*cbdata*/){

   std::cout<<"Enters context log call back"<<std::endl;

   std::cerr<<"[level, tag, message] = ["
            <<level<<", "<<tag<<", "<<message<<"]"<<"\n";
}




//creates and configures a optix device context for primary GPU device
//only
void OptixRayTrace::createContext(){

   std::cout<<"Enters createContext()"<<std::endl;

//   CUDA_CHECK(cudaFree(0));

   OptixDeviceContext context;
   CUcontext cuCtx = 0; //0 = current context
   OptixDeviceContextOptions  options = {};
   options.logCallbackFunction = &context_log_cb;
   options.logCallbackLevel = 4;

   OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

   state.context = context;

   std::cout<<"\033[1;31m createContext() done \033[0m"<<std::endl;

/*
  const int deviceID = 0;
  CUDA_CHECK(cudaSetDevice(deviceID));

  //create the CUDA stream
  CUDA_CHECK(cudaStreamCreate(&state.stream));

  cudaGetDeviceProperties(&deviceProps, deviceID);
  std::cout<<"Running on device "<<deviceProps.name<<std::endl;

  CUresult cuRes = cuCtxGetCurrent(&cudaContext);
  if(cuRes != CUDA_SUCCESS){
  throw std::runtime_error("error quering current context");
  }

  OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &state.context));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(state.context, context_log_cb, nullptr, 4));
*/
}



void OptixRayTrace::initOptix(){
   std::cout<<"Initializing Optix"<<std::endl;

//check for Optix 7 compatible devices
   CUDA_CHECK(cudaFree(0));
   int numDevices;
   cudaGetDeviceCount(&numDevices);
   if(numDevices == 0){
      throw std::runtime_error("No OptiX 7.0 compatible devices!");
   }

//init optix
   OPTIX_CHECK(optixInit());

   std::cout<<"\033[1;33m"
            <<"optix successfully initialized"
            <<"\033[0m"
            <<std::endl;
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

   std::cout<<"tris.size() = "<<tris.size()<<std::endl;





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
//triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
   triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
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

   std::cout<<"\033[1;31m buildAS() done \033[0m"<<std::endl;
}


/*



//defult bottom tri version
void OptixRayTrace::buildAS(){

std::cout<<"Enters buildAS"<<std::endl;


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



std::cout<<"In buildAS, finishes memory allocation for list of triangles"<<std::endl;


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

std::cout<<"\033[1;31m buildAS() done \033[0m"<<std::endl;
}
*/
void OptixRayTrace::convertVecMeshType(std::vector<Triangle*> &tris, std::vector<Vertex> &trisArray){
//void OptixRayTrace::convertVecMeshType(std::vector<Triangle*> &tris, std::array<Vertex, num_tri*3> &trisArray){

   std::cout<<"\tconverting Triangle mesh to Optix Triangles"<<std::endl;


   int tempIdx = 0;
   for(int i = 0; i < tris.size(); i++){ //get access to the Triangle at index


      trisArray[tempIdx] = {(*(tris[i]->a))[0], (*(tris[i]->a))[1],(*(tris[i]->a))[2]};
      tempIdx++;
      trisArray[tempIdx] = {(*(tris[i]->b))[0], (*(tris[i]->b))[1], (*(tris[i]->b))[2]};
      tempIdx++;
      trisArray[tempIdx] = {(*(tris[i]->c))[0], (*(tris[i]->c))[1], (*(tris[i]->c))[2]};
      tempIdx++;

   }

   std::cout<<"\033[1;31m converting mesh done \033[0m"<<std::endl;
}


void OptixRayTrace::calculateMixingLength(int numSamples, int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag, std::vector<double> &mixingLengths){
   state.samples_per_cell = numSamples;

   state.nx = dimX;
   state.ny = dimY;
   state.nz = dimZ;

   state.dx = dx;
   state.dy = dy;
   state.dz = dz;


   // buildAS();



   createModule();

   std::cout<<"In calculateMixingLength, createModule() done"<<std::endl;

   createProgramGroups();

   std::cout<<"In calculateMixingLength, createProgramGroups() done"<<std::endl;

   createPipeline();

   std::cout<<"In calculateMixingLength, createPipeline() done"<<std::endl;


   createSBT();


   std::cout<<"In calculateMixingLength, createSBT() done"<<std::endl;


   state.paramsBuffer.alloc(sizeof(state.params));


   initParams(dimX, dimY, dimZ, dx, dy, dz, icellflag);

   std::cout<<"In calculateMixingLength, initParams() done"<<std::endl;

   std::cout<<"In mix length BEFORE launch, params.flag (should be 3) = "<<state.params.flag<<std::endl;

   launch();

   std::cout<<"In mix length after launch, params.flag (should be 40) = "<<state.params.flag<<std::endl;

   std::vector<Hit> hitList(icellflag.size());
   state.d_hits.download( hitList.data(), icellflag.size() );

   std::cout<<"In calculateMixingLength, launch() done"<<std::endl;
   //std::cout<<"Params count = "<<state.params.count<<std::endl;

   //Hits should be init
   std::cout<<"\033[1;35m Hits output print if t != 0 \033[0m"<<std::endl;

   //state.d_hits.download(hitList,icellflag.size());
//   state.d_hits.download(hitList.data(), icellflag.size());


   std::cout<<"HitList at 0 = "<<hitList[0].t<<std::endl;


//   for(int i = 0; i < state.num_cells; i++){
   for(int i = 0; i < icellflag.size(); i++){

      // std::cout << "ml: " << state.params.hits[i].t << std::endl;
      std::cout << "ml2: " << hitList[i].t << std::endl;

      mixingLengths[i] = hitList[i].t;

      if(hitList[0].t != 0){
         std::cout<<"In mixlength, hit at index "<<i<<" = "<<hitList[i].t<<std::endl;
      }
   }
   /*for(int i = 0; i < state.num_cells; i++){
     mixingLengths.push_back(state.params.hits[i].t);


     if(state.params.hits[i].t != 0){
     std::cout<<"Hit at index "<<i<<" = "<<state.params.hits[i].t<<std::endl;
     }
     }*/

   //cleanState();

   std::cout<<"\033[1;31m End of calcMixLength() \033[0m"<<std::endl;
}


void OptixRayTrace::initParams(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag){



   //allocate memory for hits (container for the t info from device to host)
   size_t hits_size_in_bytes = sizeof(Hit)*icellflag.size();
   state.d_hits.alloc(hits_size_in_bytes);

   //acceleration structure handle
   state.params.handle = state.gas_handle;

   std::cout<<"Hits memory allocated and params.handle init"<<std::endl;

   //allocate memory to device-side icellflag memory
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.icellflagArray_d), icellflag.size()*sizeof(int)));

   std::cout<<"Finished allocating memory to icellflag_d"<<std::endl;

   //temporary fix: copy std::vector data to int array
   std::cout<<"icellflag size = "<<icellflag.size()<<std::endl;

   int *tempArray = (int*) malloc(icellflag.size()*sizeof(int));

   for(int i = 0; i < icellflag.size(); i++){
      tempArray[i] = icellflag[i];
   }
   std::cout<<"Finished copying std::vector to array"<<std::endl;

   //copy data from std::vector icellflag to device-side memory
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.icellflagArray_d),
                         reinterpret_cast<void*>(tempArray),
                         icellflag.size()*sizeof(int),
                         cudaMemcpyHostToDevice));

   std::cout<<"finished copying icell host to device"<<std::endl;

   //assign params icellflag pointer to point to icellflagArray_d
   state.params.icellflagArray = (int *) state.icellflagArray_d;

   std::cout<<"Params icellflag alloc and assignment finished"<<std::endl;

   //init params dx, dy, dz
   state.params.dx = dx;
   state.params.dy = dy;
   state.params.dz = dz;

   std::cout<<"In init params, params.dx, dy, and dz are: "<<state.params.dx<<", "<<state.params.dy<<", "<<state.params.dz<<std::endl;

   //start of tests
   state.params.flag = 3; //test value

   //end of tests


   std::cout<<"\033[1;31m initParams() done \033[0m"<<std::endl;
}



/*
  void OptixRayTrace::initParams(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag){
  //memory allocation


  OptixRay* rays_d = 0;

  int numAirCells = 0;

  std::cout<<"In initParams(), icellflag size = "<<icellflag.size()<<std::endl;

  //size_t rays_size_in_bytes = sizeof(OptixRay)*numCells;
  size_t rays_size_in_bytes = sizeof(OptixRay)*icellflag.size();

  std::cout<<"In initParams(), rays_size_in_bytes ="<<rays_size_in_bytes<<std::endl;

  // CUDA_CHECK(cudaMalloc(&rays_d, rays_size_in_bytes));

  rays_d = (OptixRay*) malloc(rays_size_in_bytes);

  std::cout<<"In initParams(), init ray_size_in_bytes"<<std::endl;

  //Hit* hits_d = 0;
//   size_t hits_size_in_bytes = sizeof(Hit)*numCells;


size_t hits_size_in_bytes = sizeof(Hit)*icellflag.size();
state.d_hits.alloc(hits_size_in_bytes);


//CUDA_CHECK(cudaMalloc(&hits_d, hits_size_in_bytes));
//hits_d = (Hit*) malloc(hits_size_in_bytes);
std::cout<<"In initParams(), init hits_size_in_bytes"<<std::endl;

std::cout<<"size of rays_d = "<<sizeof(rays_d)/sizeof(rays_d[0])<<std::endl;

//init ray data
for(int k = 0; k < dimZ -1; k++){
for(int j = 0; j < dimY -1; j++){
for(int i = 0; i < dimX -1; i++){

int icell_idx = i + j*(dimX-1) + k*(dimY -1)*(dimX-1);


if(icellflag[icell_idx] == 1){ //only want air cells

//std::cout<<"Enters if condition in initParams"<<std::endl;
rays_d[icell_idx].origin = {(i+0.5)*dx,(j+0.5)*dy,(k+0.5)*dz};
//rays_d[icell_idx].origin = make_float3((i+0.5)*dx,(j+0.5)*dy,(k+0.5)*dz);
//std::cout<<"icell_idx "<<icell_idx<<std::endl;

rays_d[icell_idx].isRay = true;
//(rays_d+icell_idx)->origin = make_float3((i+0.5)*dx,(j+0.5)*dy,(k+0.5)*dz);


rays_d[icell_idx].tmin = 0.0f;

rays_d[icell_idx].dir = {0,0,-1}; //set to bottom direction for now
rays_d[icell_idx].tmax = std::numeric_limits<float>::max();

numAirCells++;
}else{
rays_d[icell_idx].isRay = false;
rays_d[icell_idx].dir = {0,0,-1};  //just to see if this is working

}
}
}
}



//testing

state.params.flag = 3;
//   state.params.testOptixRay.origin = {1,2,3};

// state.params.testOptixRay.flag =50;

//state.params.testOptixRay.origin.x = 1;
//state.params.testOptixRay.origin.y = 2;
// state.params.testOptixRay.origin.z = 3;

//state.params.testOptixRay.origin = make_float3(1.0,2.0,3.0);
//state.testRays_d.free();

state.testRays_d.alloc(rays_size_in_bytes);
state.testRays_d.upload(rays_d, icellflag.size());
//CUDA_CHECK(cudaMemcpy(&state.testRays_d, rays_d, icellflag.size(), cudaMemcpyHostToDevice));
state.params.rays = (OptixRay *) state.testRays_d.d_pointer();

//state.params.testRays = (OptixRay *) state.testRays_d.d_ptr;
//state.testRays_d.alloc(rays_size_in_bytes);
//state.testRays_d.upload(rays_d, icellflag.size());

state.params.sizeRays = rays_size_in_bytes;
state.params.sizeIcell = icellflag.size();


//end testing




std::cout<<"In initParams(), OptiX Ray print test"<<rays_d[102618]<<std::endl;
std::cout<<"In initParams(), finished init ray data"<<std::endl;
//   std::cout<<"size of rays_d = "<<sizeof(rays_d)/sizeof(rays_d[0])<<std::endl;


std::cout<<"\n---Example ray data at index 102618----"<<std::endl;
std::cout<<"origin = <"<<rays_d[102618].origin.x<<", "<<rays_d[102618].origin.y<< ", "<<rays_d[102618].origin.z<<">"<<std::endl;
std::cout<<"dir = <"<<rays_d[102618].dir.x<<", "<<rays_d[102618].dir.y<< ", "<<rays_d[102618].dir.z<<">"<<std::endl;
std::cout<<"------------------------------------"<<std::endl;

std::cout<<"num air cells = "<<numAirCells<<std::endl;
//init params
state.params.handle = state.gas_handle;
state.params.rays = rays_d;  //not device side

//state.params.hits = hits_d;
//state.params.count = 0; //temp

state.num_cells = numAirCells;


std::cout<<"In initParams(), updated state params"<<std::endl;

std::cout<<"size of params.rays = "<<sizeof(state.params.rays)<<std::endl;



std::cout<<"\n---Example ray data at index 102618 in state.params.rays----"<<std::endl;
std::cout<<"origin = <"<<state.params.rays[102618].origin.x<<", "<<state.params.rays[102618].origin.y<< ", "<<state.params.rays[102618].origin.z<<">"<<std::endl;
std::cout<<"dir = <"<<state.params.rays[102618].dir.x<<", "<<state.params.rays[102618].dir.y<< ", "<<state.params.rays[102618].dir.z<<">"<<std::endl;
std::cout<<"------------------------------------"<<std::endl;



std::cout<<"\033[1;31m initParams() done \033[0m"<<std::endl;
}

*/

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
   //state.pipeline_compile_options.numAttributeValues = 2;
   //state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

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

   std::cout<<"\033[1;31m createModule() done \033[0m"<<std::endl;
}

void OptixRayTrace::createProgramGroups(){

   //OptiX error reporting var
   char log[2048];
//   size_t sizeof_log = sizeof(log);

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

   std::cout<<"\033[1;31m createProgramGroups() done \033[0m"<<std::endl;
}

void OptixRayTrace::createPipeline(){
   //OptiX error reporting var
   char log[2048];
   //size_t sizeof_log = sizeof(log);

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

   std::cout<<"\033[1;31m createPipeline() done \033[0m"<<std::endl;
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
   sbt_miss.data.missNum = std::numeric_limits<float>::max(); //test value

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

   CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), hit_record_size));

   HitGroupRecord sbt_hit;
   OPTIX_CHECK(optixSbtRecordPackHeader(state.hit_prog_group, &sbt_hit));
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hit_record),
                         &sbt_hit,
                         hit_record_size,
                         cudaMemcpyHostToDevice
                         ));


/*CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), hit_record_size*RAY_TYPE_COUNT));

  HitGroupRecord hit_records[RAY_TYPE_COUNT];

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hit_record),
  hit_records,
  hit_record_size*RAY_TYPE_COUNT,
  cudaMemcpyHostToDevice)
  );


*/

   std::cout<<"In createSBT(), hit record created"<<std::endl;

//update state
   state.sbt.raygenRecord = d_raygen_record;

   state.sbt.missRecordBase = d_miss_record;
//state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
   state.sbt.missRecordStrideInBytes = sizeof(MissRecord);
   state.sbt.missRecordCount = 1;

   state.sbt.hitgroupRecordBase = d_hit_record;
//state.sbt.hitgroupRecordStrideInBytes =  static_cast<uint32_t>(hit_record_size);
   state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
//   state.sbt.hitgroupRecordCount = RAY_TYPE_COUNT;
   state.sbt.hitgroupRecordCount = 1;

   std::cout<<"In createSBT(), state records updated"<<std::endl;

   std::cout<<"\033[1;31m createSBT() done \033[0m"<<std::endl;

}


void OptixRayTrace::launch(){
   //create the CUDA stream
   CUDA_CHECK(cudaStreamCreate(&state.stream));


   //state.params.testRays = (OptixRay *) state.testRays_d.d_ptr;
   //state.testRays_d.alloc(state.params.sizeRays);
   //state.testRays_d.upload(state.params.rays, state.params.sizeIcell);
   //state.testRays_d.alloc_and_upload(state.params.rays, state.params.sizeIcell);

   state.params.hits = (Hit *) state.d_hits.d_ptr;

   state.paramsBuffer.upload(&state.params, 1);

   std::cout<<"nx, ny, nz: "<<state.nx<<", "<<state.ny<<", "<<state.nz<<std::endl;


   OPTIX_CHECK(optixLaunch(state.pipeline,
                           state.stream,
                           state.paramsBuffer.d_pointer(),
                           state.paramsBuffer.sizeInBytes, //sizeof(Params),
                           &state.sbt,
                           state.nx, //state.num_cells,
                           state.ny,//1,//state.num_cells,//state.samples_per_cell,
                           state.nz//1
                           )
               );


   CUDA_SYNC_CHECK();  //this line is necessary!!

   std::cout<<"\033[1;31m launch() done \033[0m"<<std::endl;
}




/*
  void OptixRayTrace::launch(){
  //create the CUDA stream
  CUDA_CHECK(cudaStreamCreate(&state.stream));

  state.params.testRays = (OptixRay *) state.testRays_d.d_ptr;
  state.testRays_d.alloc(state.params.sizeRays);
  state.testRays_d.upload(state.params.rays, state.params.sizeIcell);
  //state.testRays_d.alloc_and_upload(state.params.rays, state.params.sizeIcell);

  state.params.hits = (Hit *) state.d_hits.d_ptr;
  //state.paramsBuffer.upload(&state.params, 1);

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
  state.nx, //state.num_cells,
  state.ny,//1,//state.num_cells,//state.samples_per_cell,
  state.nz//1
  )
  );
  CUDA_SYNC_CHECK();
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));

  std::cout<<"\033[1;31m launch() done \033[0m"<<std::endl;
  }

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
   OPTIX_CHECK(optixDeviceContextDestroy(state.context));

   //free cuda stuff
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.rays)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.hits)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));

   std::cout<<"\033[1;31m cleanState() done \033[0m"<<std::endl;
}
