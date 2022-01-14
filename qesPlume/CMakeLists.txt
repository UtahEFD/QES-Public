# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( qesPlume
  qesPlumeMain.cpp
  )

target_link_libraries(qesPlume qesplumecore)
target_link_libraries(qesPlume qeswindscore)
target_link_libraries(qesPlume qesutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(qesPlume qesOptix)
ENDIF()

target_link_libraries(qesPlume ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(qesPlume ${Boost_DATE_TIME_LIBRARIES})
target_link_libraries(qesPlume ${GDAL_LIBRARY})
target_link_libraries(qesPlume ${NETCDF_LIBRARIES_CXX})
target_link_libraries(qesPlume ${NETCDF_LIBRARIES_C})
target_link_libraries(qesPlume cudadevrt)
target_link_libraries(qesPlume ${CUDA_LIBRARIES})
