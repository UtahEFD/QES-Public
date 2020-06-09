include_directories(${CMAKE_SOURCE_DIR})

# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( cudaCanopy
  cudaCanopyMain.cpp
  )

target_link_libraries(cudaCanopy cudaurbcore)
target_link_libraries(cudaCanopy cudaurbutil)
target_link_libraries(cudaCanopy ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(cudaCanopy ${GDAL_LIBRARY})
target_link_libraries(cudaCanopy ${NETCDF_LIBRARIES_CXX})
target_link_libraries(cudaCanopy ${NETCDF_LIBRARIES_C})
target_link_libraries(cudaCanopy cudadevrt)
target_link_libraries(cudaCanopy ${CUDA_LIBRARIES})
