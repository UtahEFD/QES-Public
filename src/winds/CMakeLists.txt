include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

CUDA_ADD_LIBRARY( qeswindscore
  Canopy.cpp
  CanopyElement.cpp
  CanopyHomogeneous.cpp
  CanopyIsolatedTree.cpp
  CanopyWindbreak.cpp
  CanopyROC.cpp
  Cell.cpp
  CPUSolver.cpp
  Cut_cell.cpp
  DTEHeightField.cpp
  DynamicParallelism.cu
  handleWINDSArgs.cpp
  LocalMixing.cpp
  LocalMixingDefault.cpp
  LocalMixingNetCDF.cpp
  LocalMixingSerial.cpp
  LocalMixingOptix.cpp
  PolyBuilding.cpp PolyBuilding.h
  Sensor.cpp
  Solver.cpp
  TURBParams.h
  TURBGeneralData.cpp
  TURBGeneralData.cu
  TURBGeneralData.h
  TURBOutput.cpp
  TURBWall.cpp
  TURBWallBuilding.cpp
  TURBWallTerrain.cpp
  WindProfilerSensorType.cpp
  WindProfilerBarnCPU.cpp
  WindProfilerBarnGPU.cpp
  WindProfilerBarnGPU.cu
  WindProfilerWRF.cpp
  WINDSInputData.h
  WINDSGeneralData.cpp WINDSGeneralData.h
  WINDSOutputVisualization.cpp
  WINDSOutputWorkspace.cpp
  WRFInput.cpp WRFInput.h
  Wall.cpp Wall.h
  UpwindCavity.cpp
  NonLocalMixing.cpp
  PolygonWake.cpp
  StreetCanyon.cpp
  StreetCanyonModified.cpp
  Sidewall.cpp
  Rooftop.cpp
  StreetIntersection.cpp

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



