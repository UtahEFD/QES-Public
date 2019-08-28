include_directories(${CMAKE_SOURCE_DIR})

# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( cudaPlume
  cudaPlumeMain.cpp
  )

target_link_libraries(cudaPlume cudaplumecore)
target_link_libraries(cudaPlume qesutil)
target_link_libraries(cudaPlume ${Boost_PROGRAM_OPTIONS_LIBRARIES})
#target_link_libraries(cudaPlume ${GDAL_LIBRARY})
target_link_libraries(cudaPlume ${NETCDF_LIBRARIES_CXX})
target_link_libraries(cudaPlume ${NETCDF_LIBRARIES_C})
#target_link_libraries(cudaPlume cudadevrt)
#target_link_libraries(cudaPlume ${CUDA_LIBRARIES})