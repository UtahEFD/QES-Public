include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( cudaFire
  cudaFireMain.cpp
  )

target_link_libraries(cudaFire cudaurbcore)
target_link_libraries(cudaFire cudafirecore)
target_link_libraries(cudaFire qesutil)
target_link_libraries(cudaFire ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(cudaFire ${GDAL_LIBRARY})
target_link_libraries(cudaFire ${NETCDF_LIBRARIES_CXX})
target_link_libraries(cudaFire ${NETCDF_LIBRARIES_C})
target_link_libraries(cudaFire cudadevrt)
target_link_libraries(cudaFire ${CUDA_LIBRARIES})
