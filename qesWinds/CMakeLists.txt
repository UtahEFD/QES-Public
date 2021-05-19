include_directories(${CMAKE_SOURCE_DIR})

# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( qesWinds
  qesWindsMain.cpp
  )

target_link_libraries(qesWinds curand)
target_link_libraries(qesWinds qeswindscore)
target_link_libraries(qesWinds qeswindsutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(qesWinds qesOptix)
ENDIF()

target_link_libraries(qesWinds ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(qesWinds ${Boost_DATE_TIME_LIBRARIES})
target_link_libraries(qesWinds ${GDAL_LIBRARY})
target_link_libraries(qesWinds ${NETCDF_LIBRARIES_CXX})
target_link_libraries(qesWinds ${NETCDF_LIBRARIES_C})
target_link_libraries(qesWinds cudadevrt)
target_link_libraries(qesWinds ${CUDA_LIBRARIES})

add_test(NAME CPU_FlatTerrain COMMAND ./qesWinds -q
  ../../data/InputFiles/FlatTerrain.xml)
add_test(NAME GPU_FlatTerrain COMMAND ./qesWinds -q
  ../../data/InputFiles/FlatTerrain.xml -s 2)

# add_test(NAME CPU_GaussianHill COMMAND ./qesWinds -q
#   ../../data/InputFiles/GaussianHill.xml)
add_test(NAME GPU_GaussianHill COMMAND ./qesWinds -q
  ../../data/InputFiles/GaussianHill.xml -s 2)

# define which tests are basic checks that don't take long to run
# use: `ctest -L "basic"`
set_tests_properties(CPU_FlatTerrain GPU_FlatTerrain GPU_GaussianHill
  PROPERTIES LABELS basic)

# add_test(NAME CPU_GranitePeak COMMAND ./qesWinds -q
#   ../../data/InputFiles/GranitePeak.xml)
# add_test(NAME GPU_GranitePeak COMMAND ./qesWinds -q
#   ../../data/InputFiles/GranitePeak.xml -s 2)

# add_test(NAME CPU_OklahomaCity COMMAND ./qesWinds -q
#   ../../data/InputFiles/Oklahoma_city.xml)
# add_test(NAME GPU_OklahomaCity COMMAND ./qesWinds -q
#   ../../data/InputFiles/Oklahoma_city.xml -s 2)

