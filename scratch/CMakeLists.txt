include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/winds)

#  wrfInputTest
#  wrf2dem

set(BASETESTS
  argparser
  shpTest
  wrfCouplingTest
  )

foreach(basetest ${BASETESTS})
    add_executable(${basetest} ${basetest}.cpp)

    IF($CACHE{HAS_OPTIX_SUPPORT})
      target_link_libraries(${basetest} qesOptix)
    ENDIF()

    target_link_libraries(${basetest} qeswindscore)
    target_link_libraries(${basetest} qesutil)
    target_link_libraries(${basetest} ${Boost_LIBRARIES})

    target_link_libraries(${basetest} ${GDAL_LIBRARY})
    target_link_libraries(${basetest} ${NETCDF_LIBRARIES_CXX})
    target_link_libraries(${basetest} ${NETCDF_LIBRARIES_C})
    IF ($CACHE{HAS_CUDA_SUPPORT})
        #target_link_libraries(${basetest} cudadevrt)
    	target_link_libraries(${basetest} ${CUDA_LIBRARIES})
    ENDIF ($CACHE{HAS_CUDA_SUPPORT})

endforeach(basetest)

