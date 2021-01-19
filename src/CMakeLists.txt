include_directories(${CMAKE_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})


CUDA_ADD_LIBRARY( qeswindscore
  BVH.cpp
  Canopy.cpp
  Cell.cpp
  CPUSolver.cpp
  DTEHeightField.cpp
  DynamicParallelism.cu
  ESRIShapefile.cpp ESRIShapefile.h
  handleWINDSArgs.cpp
  Mesh.cpp
  NetCDFInput.cpp
  NetCDFOutput.cpp
  QESNetCDFOutput.cpp
  PolyBuilding.cpp PolyBuilding.h
  Sensor.cpp
  Sensor.cu
  Solver.cpp
  Triangle.cpp
  WINDSInputData.h
  WINDSGeneralData.cpp
  WINDSOutputVisualization.cpp
  WINDSOutputWorkspace.cpp
  Wall.cpp Wall.h
  UpwindCavity.cpp
  PolygonWake.cpp
  StreetCanyon.cpp
  Sidewall.cpp
  Rooftop.cpp

  GlobalMemory.cu
  GlobalMemory.h
  SharedMemory.cu
  SharedMemory.h
  TimeSeries.h
  )
 
IF (HAS_OPTIX_SUPPORT MATCHES ON)
  cuda_compile_and_embed(embedded_ptx_code OptixRayTrace.cu)
 
  add_library(qesOptix
    ${embedded_ptx_code}
    OptixRayTrace.h
    OptixRayTrace.cpp
    )  
ENDIF()




