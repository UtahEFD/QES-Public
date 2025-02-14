include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(.)

#FILE(GLOB files "*.h")
#INSTALL(FILES ${files} DESTINATION include/unitTests)

add_executable(unit_test_example_t00 t00.cpp)
target_link_libraries(unit_test_example_t00 PRIVATE qesutil)
link_external_libraries(unit_test_example_t00)
target_link_libraries(unit_test_example_t00 PRIVATE Catch2::Catch2WithMain)

unset(UNITTESTS)
  
add_executable(util_time util_time.cpp)
list(APPEND UNITTESTS util_time)

add_executable(util_vector_classes_CPU util_vector_classes_CPU.cpp)
list(APPEND UNITTESTS util_vector_classes_CPU)

add_executable(winds_terrain
  test_DTEHeightField.h test_DTEHeightField.cpp
  winds_terrain.cpp)
list(APPEND UNITTESTS winds_terrain)

add_executable(turbulence_derivative_CPU turbulence_derivative_CPU.cpp)
list(APPEND UNITTESTS turbulence_derivative_CPU)

add_executable(plume_IDGenerator plume_IDGenerator.cpp)
list(APPEND UNITTESTS plume_IDGenerator)

add_executable(plume_interpolation_CPU
  test_functions.cpp
  plume_interpolation_CPU.cpp)
list(APPEND UNITTESTS plume_interpolation_CPU)

add_executable(plume_input plume_input.cpp)
list(APPEND UNITTESTS plume_input)

add_executable(plume_managed_container plume_managed_container.cpp)
list(APPEND UNITTESTS plume_managed_container)
  
add_executable(test_qesDataTransporter test_qesDataTransporter.cpp)
list(APPEND UNITTESTS test_qesDataTransporter)

IF ($CACHE{HAS_CUDA_SUPPORT})
  
  add_executable(test_GPU test_GPU.cpp)
  TARGET_INCLUDE_DIRECTORIES(test_GPU PRIVATE {CUDAToolkit_INCLUDE_DIRS})
  SET_TARGET_PROPERTIES(test_GPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(test_GPU PRIVATE qesutilgpu)
  target_link_libraries(test_GPU PRIVATE ${CUDA_LIBRARIES})
  list(APPEND UNITTESTS test_GPU)

  add_executable(test_CUDARandomGen
    CUDARandomKernel.cu CUDARandomKernel.h
    test_CUDARandomGen.cpp)
  target_link_libraries(test_CUDARandomGen PRIVATE ${CUDA_LIBRARIES})
  target_link_libraries(test_CUDARandomGen PRIVATE ${CUDA_curand_LIBRARY})
  list(APPEND UNITTESTS test_CUDARandomGen)

  add_executable(util_vector_classes_GPU
    CUDA_vector_testkernel.cu CUDA_vector_testkernel.h
    util_vector_classes_GPU.cpp)
  list(APPEND UNITTESTS util_vector_classes_GPU)

  add_executable(util_advect_GPU
    CUDA_advect_testkernel.cu
    util_advect_GPU.cpp)
  list(APPEND UNITTESTS util_advect_GPU)

  add_executable(test_particle_partition_GPU
    CUDA_particle_partition_testkernel.cu
    test_particle_partition_GPU.cpp)
  target_link_libraries(test_particle_partition_GPU PRIVATE ${CUDA_LIBRARIES})
  target_link_libraries(test_particle_partition_GPU PRIVATE ${CUDA_curand_LIBRARY})
  list(APPEND UNITTESTS test_particle_partition_GPU)

  add_executable(test_advect_partition_GPU
    CUDA_particle_partition.cu
    CUDA_boundary_conditions.cu
    CUDA_advection.cu
    CUDA_advect_partition_testkernel.cu
    test_advect_partition_GPU.cpp)
  SET_TARGET_PROPERTIES(test_advect_partition_GPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(test_advect_partition_GPU PRIVATE qesutilgpu)
  target_link_libraries(test_advect_partition_GPU PRIVATE ${CUDA_LIBRARIES})
  target_link_libraries(test_advect_partition_GPU PRIVATE ${CUDA_curand_LIBRARY})
  list(APPEND UNITTESTS test_advect_partition_GPU)

  add_executable(test_interpolation_GPU
    CUDA_QES_Data.cu
    CUDA_interpolation.cu
    CUDA_interpolation_testkernel.cu
    test_interpolation_GPU.cpp)
  target_link_libraries(test_interpolation_GPU PRIVATE ${CUDA_LIBRARIES})
  target_link_libraries(test_interpolation_GPU PRIVATE ${CUDA_curand_LIBRARY})
  list(APPEND UNITTESTS test_interpolation_GPU)

  add_executable(test_concentration_GPU
    CUDA_concentration.cu
    CUDA_particle_partition.cu
    CUDA_concentration_testkernel.cu
    test_concentration_GPU.cpp)
  target_link_libraries(test_concentration_GPU PRIVATE ${CUDA_LIBRARIES})
  target_link_libraries(test_concentration_GPU PRIVATE ${CUDA_curand_LIBRARY})
  list(APPEND UNITTESTS test_concentration_GPU)

  #add_executable(test_plume_GPU
  #  CUDA_QES_Data.cu
  #  CUDA_interpolation.cu
  #  CUDA_particle_partition.cu
  #  CUDA_boundary_conditions.cu
  #  CUDA_advection.cu
  #  CUDA_concentration.cu
  #  CUDA_plume_testkernel.cu
  #  test_plume_GPU.cpp)
  #SET_TARGET_PROPERTIES(test_plume_GPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  #target_link_libraries(test_plume_GPU PRIVATE qesutilgpu)
  #target_link_libraries(test_plume_GPU PRIVATE ${CUDA_LIBRARIES})
  #target_link_libraries(test_plume_GPU PRIVATE ${CUDA_curand_LIBRARY})
  #list(APPEND UNITTESTS util_vector_classes_GPU)

  add_executable(util_buffer_GPU
    CUDA_buffer_testkernel.cu
    util_buffer_GPU.cpp)
  list(APPEND UNITTESTS util_buffer_GPU)

  add_executable(test_CUDAPartition
    CUDAPartitionKernel.cu CUDAPartitionKernel.h
    test_CUDAPartition.cpp)
  target_link_libraries(test_CUDAPartition PRIVATE ${CUDA_LIBRARIES})
  target_link_libraries(test_CUDAPartition PRIVATE ${CUDA_curand_LIBRARY})
  list(APPEND UNITTESTS test_CUDAPartition)

ENDIF ($CACHE{HAS_CUDA_SUPPORT})

foreach (unittest ${UNITTESTS})
  
  target_link_libraries(${unittest} PRIVATE qesplumecore)
  IF ($CACHE{HAS_CUDA_SUPPORT})
    target_link_libraries(${unittest} PRIVATE qeswindsgpu)
  ENDIF()
  target_link_libraries(${unittest} PRIVATE qeswindscore)
  IF ($CACHE{HAS_CUDA_SUPPORT})
    SET_TARGET_PROPERTIES(${unittest} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${unittest} PRIVATE qesutilgpu)
  ENDIF()
  target_link_libraries(${unittest} PRIVATE qesutil)
  IF($CACHE{HAS_OPTIX_SUPPORT})
    target_link_libraries(${unittest} PRIVATE qesOptix)
  ENDIF()
  
  link_external_libraries(${unittest})
  target_link_libraries(${unittest} PRIVATE Catch2::Catch2WithMain)
  
  add_test(NAME unit_test::${unittest} COMMAND ${unittest})
  
endforeach(unittest)

#add_test(unitTest unitTest)
