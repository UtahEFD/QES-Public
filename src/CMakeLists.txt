include_directories(${CMAKE_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/unitTests)

add_subdirectory( unitTests )

CUDA_ADD_LIBRARY( qeswindscore
  BVH.cpp
  Canopy.cpp
  Cell.cpp
  CPUSolver.cpp
  Cut_cell.cpp
  DTEHeightField.cpp
  DynamicParallelism.cu
  ESRIShapefile.cpp ESRIShapefile.h
  FIREOutput.cpp
  handleWINDSArgs.cpp
  LocalMixing.cpp
  LocalMixingDefault.cpp
  LocalMixingNetCDF.cpp
  LocalMixingSerial.cpp
  LocalMixingOptix.cpp
  Mesh.cpp
  NetCDFInput.cpp
  NetCDFOutput.cpp
  QESNetCDFOutput.cpp
  PolyBuilding.cpp PolyBuilding.h
  Sensor.cpp
  Solver.cpp
  Triangle.cpp
  TURBParams.h
  TURBGeneralData.cpp
  TURBOutput.cpp
  TURBWall.cpp
  TURBWallBuilding.cpp
  TURBWallTerrain.cpp
  WINDSInputData.h
  WINDSGeneralData.cpp
  WINDSOutputVisualization.cpp
  WINDSOutputWorkspace.cpp
  WRFInput.cpp WRFInput.h
  Wall.cpp Wall.h
  UpwindCavity.cpp
  NonLocalMixing.cpp
  PolygonWake.cpp
  StreetCanyon.cpp
  Sidewall.cpp
  Rooftop.cpp
  StreetIntersection.cpp

  Ray.cpp Ray.h
  SphereDirections.cpp SphereDirections.h
  HitRecord.cpp HitRecord.h
  GlobalMemory.cu
  GlobalMemory.h
  SharedMemory.cu
  SharedMemory.h
  TimeSeries.h
  )

CUDA_ADD_LIBRARY( qesfirecore
  Fire.cpp
  )

 
IF (HAS_OPTIX_SUPPORT MATCHES ON)
  cuda_compile_and_embed(embedded_ptx_code OptixRayTrace.cu)
 
  add_library(qesOptix
    ${embedded_ptx_code}
    OptixRayTrace.h
    OptixRayTrace.cpp
    )  
ENDIF()
  
cuda_add_executable( unitTest
  unitTests/testsMain.cpp
  )


IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(unitTest qesOptix)
ENDIF()

target_link_libraries(unitTest qeswindscore)
target_link_libraries(unitTest qeswindsutil)
target_link_libraries(unitTest unitTestLib)
target_link_libraries(unitTest ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(unitTest ${GDAL_LIBRARY})
target_link_libraries(unitTest ${NETCDF_LIBRARIES_CXX})
target_link_libraries(unitTest ${NETCDF_LIBRARIES_C})
target_link_libraries(unitTest cudadevrt)
target_link_libraries(unitTest ${CUDA_LIBRARIES})

add_test(unitTest unitTest)



