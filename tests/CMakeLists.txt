# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

ADD_DEFINITIONS(-DQES_DIR="${PROJECT_SOURCE_DIR}")

Include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.0.1 # or a later release
)
FetchContent_MakeAvailable(Catch2)

IF (TESTS_ENABLE_REGRESSION_TESTS)
  add_subdirectory(regressionTests)
ENDIF()

IF (TESTS_ENABLE_UNIT_TESTS)
  add_subdirectory(unitTests)
ENDIF()

# easy to use functions to add sanity tests
## just specify the infile's name (w/o extension) as below
function(add_sanity_check_winds_CPU infile_prefix)
  add_test(NAME sanity_check::winds_CPU_${infile_prefix}
          COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds
          -q ${CMAKE_SOURCE_DIR}/tests/runTestsFiles/winds_${infile_prefix}.xml -s 1
          -o test_CPU_${infile_prefix} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

function(add_sanity_check_winds_GPU infile_prefix)
  add_test(NAME sanity_check::winds_GPU_${infile_prefix}
          COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds
          -q ${CMAKE_SOURCE_DIR}/tests/runTestsFiles/winds_${infile_prefix}.xml -s 3
          -o test_GPU_${infile_prefix} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

function(add_sanity_check_qes_CPU infile_prefix)
  add_test(NAME sanity_check::qes_CPU_${infile_prefix}
          COMMAND ${CMAKE_BINARY_DIR}/qes/qes
          -q ${CMAKE_SOURCE_DIR}/tests/runTestsFiles/winds_${infile_prefix}.xml -s 1
          -p ${CMAKE_SOURCE_DIR}/tests/runTestsFiles/plume_${infile_prefix}.xml
          -o test_CPU_${infile_prefix} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

function(add_sanity_check_qes_GPU infile_prefix)
  add_test(NAME sanity_check::qes_GPU_${infile_prefix}
          COMMAND ${CMAKE_BINARY_DIR}/qes/qes
          -q ${CMAKE_SOURCE_DIR}/tests/runTestsFiles/winds_${infile_prefix}.xml -s 3
          -p ${CMAKE_SOURCE_DIR}/tests/runTestsFiles/plume_${infile_prefix}.xml
          -o test_GPU_${infile_prefix} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

# sanity tests
IF (TESTS_ENABLE_SANITY_CHECKS_CPU)
  # WINDS TESTS
  add_sanity_check_winds_CPU(FlatTerrain)
  add_sanity_check_winds_CPU(GaussianHill)
  add_sanity_check_winds_CPU(MultiSensors)
  add_sanity_check_winds_CPU(OklahomaCity)
  # QES TESTS
  add_sanity_check_qes_CPU(FlatTerrain)
  add_sanity_check_qes_CPU(GaussianHill)
ENDIF()

IF (TESTS_ENABLE_SANITY_CHECKS_GPU)
  # WINDS TESTS
  add_sanity_check_winds_GPU(FlatTerrain)
  add_sanity_check_winds_GPU(GaussianHill)
  add_sanity_check_winds_GPU(OklahomaCity)
  add_sanity_check_winds_GPU(MultiSensors)
  add_sanity_check_winds_GPU(SaltLakeCity)
  add_sanity_check_winds_GPU(RxCADRE)
  # QES TESTS
  add_sanity_check_qes_GPU(FlatTerrain)
  add_sanity_check_qes_GPU(GaussianHill)
ENDIF()

# ----------------------------------------------------------
# SOLVER COMPARISON TESTS
# - comparison between serial and GPU solvers
# ----------------------------------------------------------

IF ($CACHE{HAS_CUDA_SUPPORT})
  #CUDA_ADD_EXECUTABLE(compareSolvers compareSolvers.cpp)

  #target_link_libraries(compareSolvers qeswindscore)
  #target_link_libraries(compareSolvers qeswindsgpu)
  #target_link_libraries(compareSolvers qesutil)
  #IF ($CACHE{HAS_OPTIX_SUPPORT})
  #  target_link_libraries(compareSolvers qesOptix)
  #ENDIF()

  #link_external_libraries(compareSolvers)

ELSE ($CACHE{HAS_CUDA_SUPPORT})
  # no executable if CUDA is not supported
  #	ADD_EXECUTABLE(compareSolvers compareSolvers.cpp)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})


IF ($CACHE{HAS_CUDA_SUPPORT})
  ADD_EXECUTABLE( compareWindsResults
          handleCompareArgs.cpp
          compareWindsResults.cpp
          )
ELSE ($CACHE{HAS_CUDA_SUPPORT})
  ADD_EXECUTABLE( compareWindsResults
          handleCompareArgs.cpp
          compareWindsResults.cpp
          )
ENDIF ($CACHE{HAS_CUDA_SUPPORT})

target_link_libraries(compareWindsResults PRIVATE qeswindscore)
IF ($CACHE{HAS_CUDA_SUPPORT})
  target_link_libraries(compareWindsResults PRIVATE qeswindsgpu)
ENDIF()
target_link_libraries(compareWindsResults PRIVATE qeswindscore)
target_link_libraries(compareWindsResults PRIVATE qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
  target_link_libraries(compareWindsResults PRIVATE qesOptix)
ENDIF()

link_external_libraries(compareWindsResults)

#Comparison tests function
function(add_comparison_test infile_prefix solver_arg solver_name)
  add_test(NAME ${solver_name}_COMPARISON_${infile_prefix}
          COMMAND ${CMAKE_BINARY_DIR}/tests/compareSolvers
          -q ${CMAKE_SOURCE_DIR}/data/InputFiles/${infile_prefix}.xml
          -s ${solver_arg} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

IF(ENABLE_ALL_COMPARISON_TESTS)
  add_comparison_test(FlatTerrain 1 ALL)
  #add_comparison_test(GaussianHill 1)
  IF(ENABLE_LONG_COMPARISON_TESTS)
    add_comparison_test(GaussianHill 1 ALL)
  ENDIF()
  IF(ENABLE_RIDICULOUSLY_LONG_COMPARISON_TESTS)
    add_comparison_test(AskerveinHill 1 ALL)
  ENDIF()
ELSE(ENABLE_ALL_COMPARISON_TESTS)
  IF(ENABLE_DYNAMIC_PARALLELISM_COMPARISON_TESTS)
    add_comparison_test(FlatTerrain 2 DYNAMIC)
    IF(ENABLE_LONG_COMPARISON_TESTS)
      add_comparison_test(GaussianHill 2 DYNAMIC)
    ENDIF()
    IF(ENABLE_RIDICULOUSLY_LONG_COMPARISON_TESTS)
      add_comparison_test(AskerveinHill 2 DYNAMIC)
    ENDIF()
  ENDIF()

  IF(ENABLE_GLOBAL_COMPARISON_TESTS)
    add_comparison_test(FlatTerrain 3 GLOBAL)
    IF(ENABLE_LONG_COMPARISON_TESTS)
      add_comparison_test(GaussianHill 3 GLOBAL)
    ENDIF()
    IF(ENABLE_RIDICULOUSLY_LONG_COMPARISON_TESTS)
      add_comparison_test(AskerveinHill 3 GLOBAL)
    ENDIF()
  ENDIF()

  IF(ENABLE_SHARED_COMPARISON_TESTS)
    add_comparison_test(FlatTerrain 4 SHARED)
    IF(ENABLE_LONG_COMPARISON_TESTS)
      add_comparison_test(GaussianHill 4 SHARED)
    ENDIF()
    IF(ENABLE_RIDICULOUSLY_LONG_COMPARISON_TESTS)
      add_comparison_test(AskerveinHill 4 SHARED)
    ENDIF()
  ENDIF()
ENDIF()
