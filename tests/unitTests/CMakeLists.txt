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
add_executable(util_vector_classes_CPU util_vector_classes_CPU.cpp)

add_executable(plume_IDGenerator plume_IDGenerator.cpp)

IF ($CACHE{HAS_CUDA_SUPPORT})

  cuda_add_executable(test_CUDARandomGen
    CUDARandomKernel.cu CUDARandomKernel.h
    test_CUDARandomGen.cpp
    )
   target_link_libraries(test_CUDARandomGen ${CUDA_LIBRARIES})
   target_link_libraries(test_CUDARandomGen ${CUDA_curand_LIBRARY})

   cuda_add_executable(util_vector_classes_GPU
     CUDA_vector_testkernel.cu CUDA_vector_testkernel.h
     util_vector_classes_GPU.cpp)

   cuda_add_executable(util_advect_GPU
     CUDA_advect_testkernel.cu
     util_advect_GPU.cpp)

   cuda_add_executable(test_particle_partition_GPU
     CUDA_particle_partition_testkernel.cu
     test_particle_partition_GPU.cpp)
   target_link_libraries(test_particle_partition_GPU ${CUDA_LIBRARIES})
   target_link_libraries(test_particle_partition_GPU ${CUDA_curand_LIBRARY})
   
   cuda_add_executable(test_advect_partition_GPU
     CUDA_advect_partition_testkernel.cu
     test_advect_partition_GPU.cpp)
   target_link_libraries(test_advect_partition_GPU ${CUDA_LIBRARIES})
   target_link_libraries(test_advect_partition_GPU ${CUDA_curand_LIBRARY})

   cuda_add_executable(test_interpolation_GPU
     CUDA_interpolation_testkernel.cu
     test_interpolation_GPU.cpp)
   target_link_libraries(test_interpolation_GPU ${CUDA_LIBRARIES})
   target_link_libraries(test_interpolation_GPU ${CUDA_curand_LIBRARY})
   
   cuda_add_executable(util_buffer_GPU
     CUDA_buffer_testkernel.cu
     util_buffer_GPU.cpp)
   
  cuda_add_executable(test_CUDAPartition
    CUDAPartitionKernel.cu CUDAPartitionKernel.h
    test_CUDAPartition.cpp
    )
   target_link_libraries(test_CUDAPartition ${CUDA_LIBRARIES})
   target_link_libraries(test_CUDAPartition ${CUDA_curand_LIBRARY})

   cuda_add_executable(winds_terrain
     test_DTEHeightField.h test_DTEHeightField.cpp
     winds_terrain.cpp)
   
   cuda_add_executable(turbulence_derivative_CPU
       turbulence_derivative_CPU.cpp)

   cuda_add_executable(plume_interpolation_CPU
           plume_interpolation_CPU.cpp)

  cuda_add_executable(plume_input
          plume_input.cpp)

  cuda_add_executable(plume_particle_buffer
          plume_particle_buffer.cpp)

  set(UNITTESTS
    util_time
    util_output
    util_vector_classes_CPU
    util_vector_classes_GPU
    util_advect_GPU
    util_buffer_GPU
    winds_terrain
    turbulence_derivative_CPU
    plume_IDGenerator
    plume_interpolation_CPU
    plume_input
    plume_particle_buffer
    test_CUDARandomGen
    test_CUDAPartition
    test_particle_partition_GPU
    test_advect_partition_GPU
    test_interpolation_GPU)

ELSE ($CACHE{HAS_CUDA_SUPPORT})

   add_executable(winds_terrain
     test_DTEHeightField.h test_DTEHeightField.cpp
       winds_terrain.cpp)

   add_executable(turbulence_derivative_CPU
           turbulence_derivative_CPU.cpp)

   add_executable(plume_interpolation_CPU
           plume_interpolation_CPU.cpp)

   add_executable(plume_input
           plume_input.cpp)

   add_executable(plume_particle_buffer
           plume_particle_buffer.cpp)

  set(UNITTESTS
      util_time
      util_output
      util_vector_classes_CPU
      winds_terrain
      turbulence_derivative_CPU
      plume_IDGenerator
      plume_interpolation_CPU
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
  IF ($CACHE{HAS_CUDA_SUPPORT})
    target_link_libraries(${unittest} qesutilgpu)
  ENDIF()
  target_link_libraries(${unittest} qesutil)
  IF($CACHE{HAS_OPTIX_SUPPORT})
    target_link_libraries(${unittest} qesOptix)
  ENDIF()

  link_external_libraries(${unittest})
  target_link_libraries(${unittest} Catch2::Catch2WithMain)

  add_test(NAME unit_test::${unittest} COMMAND ${unittest})

endforeach(unittest)

#add_test(unitTest unitTest)
