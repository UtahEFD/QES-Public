include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


CUDA_ADD_LIBRARY( qesfirecore
  Fire.cpp
  FIREOutput.cpp
  Fire.h
  FIREOutput.h
  Fires.hpp
  FuelProperties.hpp
  ignition.h
  OldFuelProperties.hpp
  FuelRead.h
  FuelRead.cpp
  )




