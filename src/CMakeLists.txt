CMAKE_MINIMUM_REQUIRED (VERSION 2.8) 
project(cudaUrb)

SET( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules" ${CMAKE_MODULE_PATH} )

SET(Boost_USE_STATIC_LIBS TRUE)

FIND_PACKAGE(Boost REQUIRED date_time program_options system thread unit_test_framework chrono timer)
IF(${Boost_FOUND})
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
ENDIF()

FIND_PACKAGE(NetCDF)
IF(NetCDF_FOUND)
  MESSAGE(STATUS "Found NetCDF: ${NetCDF_INCLUDE_DIRECTORIES}, ${NetCDF_LIBRARIES}")
ENDIF(NetCDF_FOUND)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")


include_directories(${CMAKE_SOURCE_DIR}/util)
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory( util )

add_executable( cudaUrb
  main/cudaUrbMain.cpp
  handleURBArgs.cpp
  CPUSolver.cpp
  NetCDFData.cpp)

target_link_libraries(cudaUrb qesutil)
target_link_libraries(cudaUrb ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(cudaUrb ${NetCDF_LIBRARIES})
