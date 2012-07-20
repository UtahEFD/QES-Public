cmake_minimum_required (VERSION 2.8) 
PROJECT (GLECode) 

ADD_EXECUTABLE(glecode
  main.cpp 
  Eulerian.cpp
  Util.cpp
  advectPar.cpp
  Random.cpp
  Dispersion.cpp
  Turbulence.cpp
  LocalMixing.cpp
  NonLocalMixing.cpp
  NonLocalMixing2.cpp
  )

