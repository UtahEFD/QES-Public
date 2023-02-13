# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

IF ($CACHE{HAS_CUDA_SUPPORT})
	CUDA_ADD_EXECUTABLE( qesWinds
	  qesWindsMain.cpp
	)
ELSE ($CACHE{HAS_CUDA_SUPPORT})
	ADD_EXECUTABLE( qesWinds
	  qesWindsMain.cpp
	)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})

# target_link_libraries(qesWinds curand)
target_link_libraries(qesWinds qeswindscore)
IF ($CACHE{HAS_CUDA_SUPPORT})
  target_link_libraries(qesWinds qeswindsgpu)
ENDIF()
target_link_libraries(qesWinds qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
  target_link_libraries(qesWinds qesOptix)
ENDIF()

link_external_libraries(qesWinds)

# easy to use functions to add sanity tests
## just specify the infile's name (w/o extension) as below
function(add_sanity_test_CPU infile_prefix)
  add_test(NAME CPU_${infile_prefix} COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${CMAKE_SOURCE_DIR}/tests/runTestsFiles/${infile_prefix}.xml -s 1 -o test_${infile_prefix}_CPU WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

function(add_sanity_test_GPU infile_prefix)
  add_test(NAME GPU_${infile_prefix} COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${CMAKE_SOURCE_DIR}/tests/runTestsFiles/${infile_prefix}.xml -s 2 -o test_${infile_prefix}_GPU WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

# sanity tests
IF (ENABLE_SANITY_TESTS OR ENABLE_ALL_TESTS)
  IF (ENABLE_CPU_TESTS)
    add_sanity_test_CPU(FlatTerrain)
    IF (ENABLE_LONG_SANITY_TESTS)
      add_sanity_test_CPU(GaussianHill)
      add_sanity_test_CPU(MultiSensors)
      add_sanity_test_CPU(OklahomaCity)
    ENDIF()
  ENDIF()

  IF (ENABLE_GPU_TESTS)
    add_sanity_test_GPU(FlatTerrain)
    add_sanity_test_GPU(GaussianHill)
    add_sanity_test_GPU(OklahomaCity)
    add_sanity_test_GPU(MultiSensors)
    add_sanity_test_GPU(SaltLakeCity)
    add_sanity_test_GPU(RxCADRE)
  ENDIF()
ENDIF()
