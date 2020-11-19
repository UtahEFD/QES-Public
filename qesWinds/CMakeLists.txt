include_directories(${CMAKE_SOURCE_DIR})

# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( qesWinds
  qesWindsMain.cpp
  )

target_link_libraries(qesWinds curand)
target_link_libraries(qesWinds qeswindscore)
target_link_libraries(qesWinds qeswindsutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(qesWinds qesOptix)
ENDIF()

target_link_libraries(qesWinds ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(qesWinds ${GDAL_LIBRARY})
target_link_libraries(qesWinds ${NETCDF_LIBRARIES_CXX})
target_link_libraries(qesWinds ${NETCDF_LIBRARIES_C})
target_link_libraries(qesWinds cudadevrt)
target_link_libraries(qesWinds ${CUDA_LIBRARIES})
