include_directories(${CMAKE_SOURCE_DIR})

# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( cudaUrb
  cudaUrbMain.cpp
  )

target_link_libraries(cudaUrb curand)
target_link_libraries(cudaUrb cudaurbcore)
target_link_libraries(cudaUrb cudaurbutil)

if (${OptiX-NOTFOUND})
  MESSAGE(STATUS "Not linking with OptiX")
ELSE()
  target_link_libraries(cudaUrb qesOptix)
ENDIF()

target_link_libraries(cudaUrb ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(cudaUrb ${GDAL_LIBRARY})
target_link_libraries(cudaUrb ${NETCDF_LIBRARIES_CXX})
target_link_libraries(cudaUrb ${NETCDF_LIBRARIES_C})
target_link_libraries(cudaUrb cudadevrt)
target_link_libraries(cudaUrb ${CUDA_LIBRARIES})
