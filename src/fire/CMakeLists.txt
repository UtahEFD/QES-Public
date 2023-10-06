include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


SET( qesFireCoreSources
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
  SourceFire.h
  SourceFire.cpp
  Smoke.h
  Smoke.cpp
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




