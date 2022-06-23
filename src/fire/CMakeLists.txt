include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


CUDA_ADD_LIBRARY( qesfirecore
  Fire.cpp
  FIREOutput.cpp
  FuelRead.cpp

  fire.hpp
  FIREOutput.h
  Fires.hpp
  FuelProperties.hpp
  FuelRead.h
  ignition.h
  OldFuelProperties.hpp
  )




