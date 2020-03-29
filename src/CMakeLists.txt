include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/unitTests)

add_subdirectory( unitTests )

CUDA_ADD_LIBRARY( cudaurbcore
  BVH.cpp
  Canopy.cpp
  Cell.cpp
  CPUSolver.cpp
  Cut_cell.cpp
  DTEHeightField.cpp
  DynamicParallelism.cu
  ESRIShapefile.cpp ESRIShapefile.h
  handleURBArgs.cpp
  Mesh.cpp
  NetCDFOutput.cpp NetCDFOutput.h
  PolyBuilding.cpp PolyBuilding.h
  Sensor.cpp
  Solver.cpp
  Triangle.cpp
  URBInputData.h
  URBGeneralData.cpp
  URBOutputData.cpp
  URBOutput_Generic.cpp 
  URBOutput_VizFields.cpp
  URBOutput_TURBInputFile.cpp
  WRFInput.cpp WRFInput.h
  Wall.cpp Wall.h
  UpwindCavity.cpp
  PolygonWake.cpp
  StreetCanyon.cpp
  Sidewall.cpp
  Rooftop.cpp
  StreetIntersection.cpp
  Ray.cpp Ray.h
  SphereDirections.cpp SphereDirections.h
  HitRecord.cpp HitRecord.h
  )

#OPTIX_add_library(target_name 
 # OptixRayTrace.cu
  #OptixRayTrace.cpp
  #OptixRayTrace.h
 # )

# include_directories(${OptiX_INCLUDE})
 
 cuda_compile_and_embed(embedded_ptx_code OptixRayTrace.cu)
 add_library(optix_lib
   ${embedded_ptx_code}
   OptixRayTrace.h
   OptixRayTrace.cpp
   )

cuda_add_executable( unitTest
  unitTests/testsMain.cpp
  )


target_link_libraries(unitTest optix_lib)
target_link_libraries(unitTest cudaurbcore)
target_link_libraries(unitTest cudaurbutil)
target_link_libraries(unitTest unitTestLib)
target_link_libraries(unitTest ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(unitTest ${GDAL_LIBRARY})
target_link_libraries(unitTest ${NETCDF_LIBRARIES_CXX})
target_link_libraries(unitTest ${NETCDF_LIBRARIES_C})
target_link_libraries(unitTest cudadevrt)
target_link_libraries(unitTest ${CUDA_LIBRARIES})

add_test(unitTest unitTest)



