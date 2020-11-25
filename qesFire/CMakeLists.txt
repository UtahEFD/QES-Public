include_directories(${CMAKE_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( qesFire
  qesFireMain.cpp
  )

target_link_libraries(qesFire curand)
target_link_libraries(qesFire qeswindscore)
target_link_libraries(qesFire qesfirecore)
target_link_libraries(qesFire qeswindsutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(qesFire qesOptix)
ENDIF()

target_link_libraries(qesFire ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(qesFire ${GDAL_LIBRARY})
target_link_libraries(qesFire ${NETCDF_LIBRARIES_CXX})
target_link_libraries(qesFire ${NETCDF_LIBRARIES_C})
target_link_libraries(qesFire cudadevrt)
target_link_libraries(qesFire ${CUDA_LIBRARIES})
