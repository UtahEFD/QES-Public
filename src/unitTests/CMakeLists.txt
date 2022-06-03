include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(unitTestLib ${LIB_TYPE} 
  util.h util.cpp)


#FILE(GLOB files "*.h")
#INSTALL(FILES ${files} DESTINATION include/unitTests)


cuda_add_executable(unitTestWind
  test_DTEHeightField.h test_DTEHeightField.cpp
  mainWindTest.cpp)

cuda_add_executable(unitTestTurbulence
  test_WINDSGeneralData.h test_WINDSGeneralData.cpp
  test_TURBGeneralData.h test_TURBGeneralData.cpp
  mainTurbulenceTest.cpp)

cuda_add_executable(unitTestPlume
  test_WINDSGeneralData.h test_WINDSGeneralData.cpp
  test_TURBGeneralData.h test_TURBGeneralData.cpp
  test_PlumeGeneralData.h test_PlumeGeneralData.cpp test_PlumeGeneralData.cu
  mainPlumeTest.cpp)

set(UNITTESTS 
  unitTestWind
  unitTestTurbulence
  unitTestPlume)

foreach (unittest ${UNITTESTS})
  target_link_libraries(${unittest} qesplumecore)
  target_link_libraries(${unittest} qeswindscore)
  target_link_libraries(${unittest} qesutil)
  IF(HAS_OPTIX_SUPPORT MATCHES ON)
    target_link_libraries(${unittest} qesOptix)
  ENDIF()
  target_link_libraries(${unittest} unitTestLib)
  target_link_libraries(${unittest} ${Boost_PROGRAM_OPTIONS_LIBRARIES})
  target_link_libraries(${unittest} ${Boost_FILESYSTEM_LIBRARIES})
  target_link_libraries(${unittest} ${GDAL_LIBRARY})
  target_link_libraries(${unittest} ${NETCDF_LIBRARIES_CXX})
  target_link_libraries(${unittest} ${NETCDF_LIBRARIES_C})
  #target_link_libraries(${unittest} cudadevrt)
  target_link_libraries(${unittest} ${CUDA_LIBRARIES})

  add_test(${unittest} ${unittest})

endforeach(unittest)

#add_test(unitTest unitTest)
