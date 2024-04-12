include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(.)

IF ($CACHE{HAS_CUDA_SUPPORT})
  cuda_add_library(unittestcore
        test_functions.h test_functions.cpp
        test_WINDSGeneralData.h test_WINDSGeneralData.cpp
        test_TURBGeneralData.h test_TURBGeneralData.cpp
        test_PlumeGeneralData.h test_PlumeGeneralData.cpp)
ELSE ($CACHE{HAS_CUDA_SUPPORT})
  add_library(unittestcore
        test_functions.h test_functions.cpp
        test_WINDSGeneralData.h test_WINDSGeneralData.cpp
        test_TURBGeneralData.h test_TURBGeneralData.cpp
        test_PlumeGeneralData.h test_PlumeGeneralData.cpp)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})
#FILE(GLOB files "*.h")
#INSTALL(FILES ${files} DESTINATION include/unitTests)

add_executable(unit_test_example_t00 t00.cpp)

target_link_libraries(unit_test_example_t00 qesutil)
link_external_libraries(unit_test_example_t00)
target_link_libraries(unit_test_example_t00 Catch2::Catch2WithMain)

add_executable(util_time util_time.cpp)

add_executable(util_output
        QESFileOutput.cpp
        DataSource.cpp
        QESNetCDFOutput_v2.cpp
        util_output.cpp)

IF ($CACHE{HAS_CUDA_SUPPORT})

  cuda_add_executable(test_CUDARandomGen
    CUDARandomKernel.cu CUDARandomKernel.h
    test_CUDARandomGen.cpp
    )
   target_link_libraries(test_CUDARandomGen ${CUDA_LIBRARIES})
   target_link_libraries(test_CUDARandomGen ${CUDA_curand_LIBRARY})

   cuda_add_executable(winds_terrain
     test_DTEHeightField.h test_DTEHeightField.cpp
     winds_terrain.cpp)
   
   cuda_add_executable(turbulence_derivative_CPU
       turbulence_derivative_CPU.cpp)

   cuda_add_executable(plume_interpolation_CPU
           plume_interpolation_CPU.cpp)

   cuda_add_executable(plume_vector_classes_CPU
     plume_vector_classes_CPU.cpp)

  cuda_add_executable(plume_particle_factory
          plume_particle_factory.cpp)

  cuda_add_executable(plume_sources
          plume_sources.cpp)

  cuda_add_executable(plume_input
          plume_input.cpp)

  cuda_add_executable(plume_particle_buffer
          plume_particle_buffer.cpp)

  set(UNITTESTS
    util_time
    util_output
    winds_terrain
    turbulence_derivative_CPU
    plume_interpolation_CPU
    plume_vector_classes_CPU
    plume_particle_factory
    plume_sources
    plume_input
    plume_particle_buffer
    test_CUDARandomGen)

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

   add_executable(plume_particle_factory
           plume_particle_factory.cpp)

   add_executable(plume_sources
           plume_sources.cpp)

   add_executable(plume_input
           plume_input.cpp)

   add_executable(plume_particle_buffer
           plume_particle_buffer.cpp)

  set(UNITTESTS
      util_time
      util_output
      winds_terrain
      turbulence_derivative_CPU
      plume_interpolation_CPU
      plume_vector_classes_CPU
      plume_particle_factory
      plume_sources
      plume_input
      plume_particle_buffer)
      
ENDIF ($CACHE{HAS_CUDA_SUPPORT})

foreach (unittest ${UNITTESTS})
  
  target_link_libraries(${unittest} unittestcore)
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

  link_external_libraries(${unittest})
  target_link_libraries(${unittest} Catch2::Catch2WithMain)

  add_test(NAME unit_test::${unittest} COMMAND ${unittest})

endforeach(unittest)

#add_test(unitTest unitTest)
