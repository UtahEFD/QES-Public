include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library (unitTestLib ${LIB_TYPE}
  util.h util.cpp
)


#FILE(GLOB files "*.h")
#INSTALL(FILES ${files} DESTINATION include/unitTests)



cuda_add_executable( unitTestWind
  test_DTEHeightField.h test_DTEHeightField.cpp
  testsMain.cpp
)

set(UNITTESTS
  unitTestWind		
)

foreach(unittest ${UNITTESTS})
  IF (HAS_OPTIX_SUPPORT MATCHES ON)
    target_link_libraries(${unittest} qesOptix)
  ENDIF()
  
  target_link_libraries(${unittest} qesplumecore)
  target_link_libraries(${unittest} qeswindscore)
  target_link_libraries(${unittest} qesutil)
  target_link_libraries(${unittest} unitTestLib)
  target_link_libraries(${unittest} ${Boost_PROGRAM_OPTIONS_LIBRARIES})  
  target_link_libraries(${unittest} ${GDAL_LIBRARY})
  target_link_libraries(${unittest} ${NETCDF_LIBRARIES_CXX})
  target_link_libraries(${unittest} ${NETCDF_LIBRARIES_C})
  #target_link_libraries(${unittest} cudadevrt)
  target_link_libraries(${unittest} ${CUDA_LIBRARIES})
endforeach(unittest)

# add_test(unitTest unitTest)
