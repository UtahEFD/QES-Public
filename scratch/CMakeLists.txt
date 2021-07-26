include_directories(${CMAKE_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src/winds)


set(BASETESTS
  argparser
  shpTest
  wrfInputTest
  wrf2dem
  )

foreach(basetest ${BASETESTS})
    add_executable(${basetest} ${basetest}.cpp)

    IF (HAS_OPTIX_SUPPORT MATCHES ON)
      target_link_libraries(${basetest} qesOptix)
    ENDIF()

    target_link_libraries(${basetest} qeswindscore)
    target_link_libraries(${basetest} qesutil)
    target_link_libraries(${basetest} ${Boost_LIBRARIES})

    target_link_libraries(${basetest} ${GDAL_LIBRARY})
    target_link_libraries(${basetest} ${NETCDF_LIBRARIES_CXX})
    target_link_libraries(${basetest} ${NETCDF_LIBRARIES_C})
    #target_link_libraries(${basetest} cudadevrt)
    target_link_libraries(${basetest} ${CUDA_LIBRARIES})

endforeach(basetest)

