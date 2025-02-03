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
  Potential.cpp
  Rothermel.cpp
  Smoke.cpp
  Smoke.h 
  SourceFire.cpp
  SourceFire.h
  )

IF ($CACHE{HAS_CUDA_SUPPORT})
    CUDA_ADD_LIBRARY( qesfirecore
            ${qesFireCoreSources}
            )
ELSE ($CACHE{HAS_CUDA_SUPPORT})
    ADD_LIBRARY( qesfirecore
            ${qesFireCoreSources}
            )
ENDIF ($CACHE{HAS_CUDA_SUPPORT})

IF ($CACHE{HAS_CUDA_SUPPORT})
  CUDA_ADD_LIBRARY(qesfiregpu
    CUDA/PotentialGlobal.h
    CUDA/PotentialGlobal.cu)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})