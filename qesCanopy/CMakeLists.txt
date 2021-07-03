include_directories(${CMAKE_SOURCE_DIR})

# needs to be relative to the current project when we
# hierarchically integrate projects together.
#include_directories(${PROJECT_SOURCE_DIR}/src/winds)
#include_directories(${PROJECT_SOURCE_DIR}/util)

CUDA_ADD_EXECUTABLE( qesCanopy
  qesCanopyMain.cpp
  )

target_link_libraries(qesCanopy curand)
target_link_libraries(qesCanopy qeswindscore)
target_link_libraries(qesCanopy qesutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(qesCanopy qesOptix)
ENDIF()

target_link_libraries(qesCanopy ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(qesCanopy ${Boost_DATE_TIME_LIBRARIES})
target_link_libraries(qesCanopy ${GDAL_LIBRARY})
target_link_libraries(qesCanopy ${NETCDF_LIBRARIES_CXX})
target_link_libraries(qesCanopy ${NETCDF_LIBRARIES_C})
target_link_libraries(qesCanopy cudadevrt)
target_link_libraries(qesCanopy ${CUDA_LIBRARIES})
