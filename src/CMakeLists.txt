include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/unitTests)

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
  Output.cpp
  PolyBuilding.cpp PolyBuilding.h
  Sensor.cpp
  Solver.cpp
  Triangle.cpp
  URBInputData.h
  URBGeneralData.cpp URBGeneralData.h
  WRFInput.cpp WRFInput.h
  Wall.cpp Wall.h
  )

CUDA_ADD_LIBRARY( cudafirecore
  Fire.cpp
  )

cuda_add_executable( unitTest
  unitTests/testsMain.cpp
  )

target_link_libraries(unitTest cudaurbcore )
target_link_libraries(unitTest cudaurbutil)
target_link_libraries(unitTest unitTestLib)
target_link_libraries(unitTest ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(unitTest ${GDAL_LIBRARY})
target_link_libraries(unitTest ${NETCDF_LIBRARIES_CXX})
target_link_libraries(unitTest ${NETCDF_LIBRARIES_C})
target_link_libraries(unitTest cudadevrt)
target_link_libraries(unitTest ${CUDA_LIBRARIES})

add_test(unitTest unitTest)
