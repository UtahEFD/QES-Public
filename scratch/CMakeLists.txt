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
      target_link_libraries(${basetest} PRIVATE qesOptix)
    ENDIF()

    target_link_libraries(${basetest} PRIVATE qeswindscore)
    target_link_libraries(${basetest} PRIVATE qesutil)
    target_link_libraries(${basetest} PRIVATE ${Boost_LIBRARIES})

    target_link_libraries(${basetest} PRIVATE ${GDAL_LIBRARY})
    target_link_libraries(${basetest} PRIVATE ${NETCDF_LIBRARIES_CXX})
    target_link_libraries(${basetest} PRIVATE ${NETCDF_LIBRARIES_C})
    IF ($CACHE{HAS_CUDA_SUPPORT})
        #target_link_libraries(${basetest} cudadevrt)
    	target_link_libraries(${basetest} PRIVATE ${CUDA_LIBRARIES})
    ENDIF ($CACHE{HAS_CUDA_SUPPORT})

endforeach(basetest)

