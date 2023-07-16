include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(.)

add_library(unittestcore ${LIB_TYPE}
        test_functions.h test_functions.cpp
        test_WINDSGeneralData.h test_WINDSGeneralData.cpp
        test_TURBGeneralData.h test_TURBGeneralData.cpp
        test_PlumeGeneralData.h test_PlumeGeneralData.cpp)

#FILE(GLOB files "*.h")
#INSTALL(FILES ${files} DESTINATION include/unitTests)

add_executable(unit_test_example_t00 t00.cpp)

target_link_libraries(unit_test_example_t00 qesutil)
link_external_libraries(unit_test_example_t00)
target_link_libraries(unit_test_example_t00 Catch2::Catch2WithMain)

add_executable(util_time util_time.cpp)

IF ($CACHE{HAS_CUDA_SUPPORT})

   add_executable(winds_terrain
     test_DTEHeightField.h test_DTEHeightField.cpp
     winds_terrain.cpp)
   
   cuda_add_executable(turbulence_derivative_CPU
       turbulence_derivative_CPU.cpp)

   cuda_add_executable(plume_interpolation_CPU
           plume_interpolation_CPU.cpp)

   cuda_add_executable(plume_vector_classes_CPU
           plume_vector_classes_CPU.cpp)

ELSE ($CACHE{HAS_CUDA_SUPPORT})

   add_executable(winds_terrain
     test_DTEHeightField.h test_DTEHeightField.cpp
       winds_terrain.cpp)

   add_executable(turbulence_derivative_CPU
           turbulence_derivative_CPU.cpp)

   add_executable(plume_interpolation_CPU
           plume_interpolation_CPU.cpp)

   add_executable(plume_vector_classes_CPU
           plume_vector_classes_CPU.cpp)

ENDIF ($CACHE{HAS_CUDA_SUPPORT})

set(UNITTESTS
  util_time
  winds_terrain
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
  target_link_libraries(${unittest} unittestcore)
  link_external_libraries(${unittest})
  target_link_libraries(${unittest} Catch2::Catch2WithMain)

  if (ENABLE_UNIT_TESTS)
    add_test(NAME unit_test::${unittest} COMMAND ${unittest})
  endif (ENABLE_UNIT_TESTS)
endforeach(unittest)

#add_test(unitTest unitTest)
