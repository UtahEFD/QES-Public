include_directories(${CMAKE_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( qesFire
  qesFireMain.cpp
  )

target_link_libraries(qesFire qeswindsbcore)
target_link_libraries(qesFire qesfirecore)
target_link_libraries(qesFire qeswindsutil)
target_link_libraries(qesFire ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(qesFire ${GDAL_LIBRARY})
target_link_libraries(qesFire ${NETCDF_LIBRARIES_CXX})
target_link_libraries(qesFire ${NETCDF_LIBRARIES_C})
target_link_libraries(qesFire cudadevrt)
target_link_libraries(qesFire ${CUDA_LIBRARIES})
