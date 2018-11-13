
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/util)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/unitTests)

add_subdirectory( util )
add_subdirectory( unitTests )

CUDA_ADD_EXECUTABLE( cudaUrb
  DynamicParallelism.cu
  handleURBArgs.cpp
  Solver.cpp
  CPUSolver.cpp
  NetCDFData.cpp
  main/cudaUrbMain.cpp
  Mesh.cpp
  Triangle.cpp
  BVH.cpp
  DTEHeightField.cpp
  Cell.cpp
  )

cuda_add_executable( unitTest
  handleURBArgs.cpp
  Solver.cpp
  CPUSolver.cpp
  NetCDFData.cpp
  unitTests/testsMain.cpp
  DynamicParallelism.cu
  Mesh.cpp
  Triangle.cpp
  BVH.cpp
  DTEHeightField.cpp
  Cell.cpp
  )


target_link_libraries(cudaUrb qesutil)
target_link_libraries(cudaUrb ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(cudaUrb netcdf)
target_link_libraries(cudaUrb netcdf_c++4)
target_link_libraries(cudaUrb ${GDAL_LIBRARY})
# target_link_libraries(cudaUrb urbgpucore )
target_link_libraries(cudaUrb cudadevrt)
target_link_libraries(cudaUrb ${CUDA_LIBRARIES})

target_link_libraries(unitTest qesutil)
target_link_libraries(unitTest unitTestLib)
target_link_libraries(unitTest ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(unitTest netcdf)
target_link_libraries(unitTest netcdf_c++4)
target_link_libraries(unitTest ${GDAL_LIBRARY})
# target_link_libraries(unitTest urbgpucore )
target_link_libraries(unitTest cudadevrt)
target_link_libraries(unitTest ${CUDA_LIBRARIES})