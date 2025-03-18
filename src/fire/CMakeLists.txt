include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


SET( qesFireCoreSources
  Balbi.cpp
  ComputeTimeStep.cpp
  Fire.cpp
  Fire.h
  FIREOutput.cpp
  FIREOutput.h
  Fires.hpp
  FuelMap.cpp
  FuelProperties.hpp
  FuelRead.cpp
  FuelRead.h
  ignition.h
  LevelSet.cpp
  LevelSetNB.cpp
  LSinit.cpp
  Move.cpp
  OldFuelProperties.hpp
        PotentialSerial.cpp
  Rothermel.cpp
  Smoke.cpp
  Smoke.h 
  SourceFire.cpp
  SourceFire.h
  )

ADD_LIBRARY( qesfirecore ${qesFireCoreSources})

IF ($CACHE{HAS_CUDA_SUPPORT})
  ADD_LIBRARY(qesfiregpu
    CUDA/LSinitGlob.cu
    CUDA/LSinitGlob.h
    CUDA/PotentialGlobal.cu
    CUDA/PotentialGlobal.h)
  TARGET_INCLUDE_DIRECTORIES(qesfiregpu PRIVATE {CUDAToolkit_INCLUDE_DIRS})
  set_target_properties(qesfiregpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})
