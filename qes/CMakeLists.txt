# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( qes
		     handleQESArgs.cpp
                     qesMain.cpp	
		     )

target_link_libraries(qes qesplumecore)
target_link_libraries(qes qeswindscore)
target_link_libraries(qes qesutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(qes qesOptix)
ENDIF()

target_link_libraries(qes ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(qes ${Boost_DATE_TIME_LIBRARIES})
target_link_libraries(qes ${Boost_FILESYSTEM_LIBRARIES})
target_link_libraries(qes ${GDAL_LIBRARY})
target_link_libraries(qes ${NETCDF_LIBRARIES_CXX})
target_link_libraries(qes ${NETCDF_LIBRARIES_C})
target_link_libraries(qes cudadevrt)
target_link_libraries(qes ${CUDA_LIBRARIES})
