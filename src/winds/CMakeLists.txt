include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

SET( qesWindCoreSources 
  BVH.cpp
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
  handleWINDSArgs.cpp
  LocalMixing.cpp
  LocalMixingDefault.cpp
  LocalMixingNetCDF.cpp
  LocalMixingSerial.cpp
  LocalMixingOptix.cpp
  Mesh.cpp
  PolyBuilding.cpp PolyBuilding.h
  Sensor.cpp
  Solver.cpp
  Triangle.cpp
  TURBParams.h
  TURBGeneralData.cpp
  TURBGeneralData.h
  TURBOutput.cpp
  TURBWall.cpp
  TURBWallBuilding.cpp
  TURBWallTerrain.cpp
  WindProfilerSensorType.cpp
  WindProfilerBarnCPU.cpp
  WindProfilerBarnGPU.cpp
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

  Ray.cpp Ray.h
  SphereDirections.cpp SphereDirections.h
  HitRecord.cpp HitRecord.h

  TimeSeries.h
  )

IF ($CACHE{HAS_CUDA_SUPPORT})
   CUDA_ADD_LIBRARY( qeswindscore
      ${qesWindCoreSources}
   )
ELSE ($CACHE{HAS_CUDA_SUPPORT})
   ADD_LIBRARY( qeswindscore
      ${qesWindCoreSources}
   )
ENDIF ($CACHE{HAS_CUDA_SUPPORT})
 

IF ($CACHE{HAS_CUDA_SUPPORT})
  CUDA_ADD_LIBRARY(qeswindsgpu
    DynamicParallelism.cu
    GlobalMemory.cu GlobalMemory.h
    SharedMemory.cu SharedMemory.h    
    TURBGeneralData.cu
    WindProfilerBarnGPU.cu)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})
  

IF ($CACHE{HAS_OPTIX_SUPPORT})
  cuda_compile_and_embed(embedded_ptx_code OptixRayTrace.cu)
 
  add_library(qesOptix
    ${embedded_ptx_code}
    OptixRayTrace.h
    OptixRayTrace.cpp
    )  
ENDIF ($CACHE{HAS_OPTIX_SUPPORT})



