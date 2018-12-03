
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/unitTests)

add_subdirectory( unitTests )

CUDA_ADD_LIBRARY( cudaurbcore
  DynamicParallelism.cu
  handleURBArgs.cpp
  Solver.cpp
  CPUSolver.cpp
  NetCDFData.cpp
  Mesh.cpp
  Triangle.cpp
  BVH.cpp
  DTEHeightField.cpp
  Cell.cpp
  Canopy.cpp
  Cut_cell.cpp
  Sensor.cpp
  )

cuda_add_executable( unitTest
  unitTests/testsMain.cpp
  )

target_link_libraries(unitTest cudaurbcore )
target_link_libraries(unitTest qesutil)
target_link_libraries(unitTest unitTestLib)
target_link_libraries(unitTest ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(unitTest netcdf)
target_link_libraries(unitTest netcdf_c++4)
target_link_libraries(unitTest ${GDAL_LIBRARY})
target_link_libraries(unitTest cudadevrt)
target_link_libraries(unitTest ${CUDA_LIBRARIES})

add_test(unitTest unitTest)
