include_directories(${CMAKE_SOURCE_DIR})

# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( qesWindInterp
  handleQESArgs.cpp                  
  qesWindInterpMain.cpp
  )                     

target_link_libraries(qesWindInterp curand)
target_link_libraries(qesWindInterp qeswindscore)
target_link_libraries(qesWindInterp qeswindsutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(qesWindInterp qesOptix)
ENDIF()

target_link_libraries(qesWindInterp ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(qesWindInterp ${Boost_DATE_TIME_LIBRARIES})
target_link_libraries(qesWindInterp ${GDAL_LIBRARY})
target_link_libraries(qesWindInterp ${NETCDF_LIBRARIES_CXX})
target_link_libraries(qesWindInterp ${NETCDF_LIBRARIES_C})
target_link_libraries(qesWindInterp cudadevrt)
target_link_libraries(qesWindInterp ${CUDA_LIBRARIES})
