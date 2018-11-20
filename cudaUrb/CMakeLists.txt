include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( cudaUrb
  cudaUrbMain.cpp
  )

target_link_libraries(cudaUrb cudaurbcore)
target_link_libraries(cudaUrb qesutil)
target_link_libraries(cudaUrb ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(cudaUrb netcdf)
target_link_libraries(cudaUrb netcdf_c++4)
target_link_libraries(cudaUrb ${GDAL_LIBRARY})
target_link_libraries(cudaUrb cudadevrt)
target_link_libraries(cudaUrb ${CUDA_LIBRARIES})
