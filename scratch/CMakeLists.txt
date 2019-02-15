include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)

set(BASETESTS
  argparser
  shpTest
  )

foreach(basetest ${BASETESTS})
    add_executable(${basetest} ${basetest}.cpp)

    target_link_libraries(${basetest} cudaurbcore)
    target_link_libraries(${basetest} qesutil)
    target_link_libraries(${basetest} ${Boost_PROGRAM_OPTIONS_LIBRARIES})
    target_link_libraries(${basetest} ${GDAL_LIBRARY})
    target_link_libraries(${basetest} ${NETCDF_LIBRARIES_CXX})
    target_link_libraries(${basetest} ${NETCDF_LIBRARIES_C})
    target_link_libraries(${basetest} cudadevrt)
    target_link_libraries(${basetest} ${CUDA_LIBRARIES})
endforeach(basetest)

