include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(.)

add_library(unitTestLib ${LIB_TYPE}
  util.h util.cpp)

#FILE(GLOB files "*.h")
#INSTALL(FILES ${files} DESTINATION include/unitTests)

add_executable(unitTest_t00 t00.cpp)

target_link_libraries(unitTest_t00 PRIVATE ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(unitTest_t00 PRIVATE ${Boost_DATE_TIME_LIBRARIES})
target_link_libraries(unitTest_t00 PRIVATE ${Boost_FILESYSTEM_LIBRARIES})
target_link_libraries(unitTest_t00 PRIVATE ${GDAL_LIBRARY})
target_link_libraries(unitTest_t00 PRIVATE ${NETCDF_LIBRARIES_CXX})
target_link_libraries(unitTest_t00 PRIVATE ${NETCDF_LIBRARIES_C})
IF ($CACHE{HAS_CUDA_SUPPORT})
    # target_link_libraries(compareSolvers cudadevrt)
    target_link_libraries(unitTest_t00 PRIVATE ${CUDA_LIBRARIES})
ENDIF()

target_link_libraries(unitTest_t00 PRIVATE qesutil)
target_link_libraries(unitTest_t00 PRIVATE Catch2::Catch2WithMain)

add_executable(util_time
        util_time.cpp)

IF ($CACHE{HAS_CUDA_SUPPORT})

   cuda_add_executable(unitTestWind
     test_DTEHeightField.h test_DTEHeightField.cpp
       mainWindTest.cpp)

   cuda_add_executable(turbulence_derivative_CPU
       test_WINDSGeneralData.h test_WINDSGeneralData.cpp
       test_TURBGeneralData.h test_TURBGeneralData.cpp
       turbulence_derivative_CPU.cpp)

   cuda_add_executable(unitTestPlume
      test_WINDSGeneralData.h test_WINDSGeneralData.cpp
      test_TURBGeneralData.h test_TURBGeneralData.cpp
      test_PlumeGeneralData.h test_PlumeGeneralData.cpp test_PlumeGeneralData.cu
      mainPlumeTest.cpp)

ELSE ($CACHE{HAS_CUDA_SUPPORT})

   add_executable(unitTestWind
     test_DTEHeightField.h test_DTEHeightField.cpp
       mainWindTest.cpp)


ENDIF ($CACHE{HAS_CUDA_SUPPORT})

add_executable(turbulence_derivative_CPU
        test_WINDSGeneralData.h test_WINDSGeneralData.cpp
        test_TURBGeneralData.h test_TURBGeneralData.cpp
        turbulence_derivative_CPU.cpp)

add_executable(plume_interpolation_CPU
        testFunctions.h testFunctions.cpp
        test_WINDSGeneralData.h test_WINDSGeneralData.cpp
        test_TURBGeneralData.h test_TURBGeneralData.cpp
        test_PlumeGeneralData.h test_PlumeGeneralData.cpp
        plume_interpolation_CPU.cpp)

add_executable(plume_vector_classes_CPU
        testFunctions.h testFunctions.cpp
        test_WINDSGeneralData.h test_WINDSGeneralData.cpp
        test_TURBGeneralData.h test_TURBGeneralData.cpp
        test_PlumeGeneralData.h test_PlumeGeneralData.cpp
        plume_vector_classes_CPU.cpp)

set(UNITTESTS
  util_time
  unitTestWind
  turbulence_derivative_CPU
  plume_interpolation_CPU
  plume_vector_classes_CPU)


foreach (unittest ${UNITTESTS})
  target_link_libraries(${unittest} qesplumecore)
  target_link_libraries(${unittest} qeswindscore)
  IF ($CACHE{HAS_CUDA_SUPPORT})
    target_link_libraries(${unittest} qeswindsgpu)
  ENDIF()
  target_link_libraries(${unittest} qeswindscore)
  target_link_libraries(${unittest} qesutil)
  IF($CACHE{HAS_OPTIX_SUPPORT})
    target_link_libraries(${unittest} qesOptix)
  ENDIF()
  target_link_libraries(${unittest} unitTestLib)
  link_external_libraries(${unittest})
  target_link_libraries(${unittest} Catch2::Catch2WithMain)

  if (ENABLE_UNIT_TESTS)
    add_test(NAME unit_test::${unittest} COMMAND ${unittest})
  endif (ENABLE_UNIT_TESTS)
endforeach(unittest)

#add_test(unitTest unitTest)
