include_directories(${CMAKE_CURRENT_SOURCE_DIR})

SET(qesUtilSources
  ArgumentParsing.cpp ArgumentParsing.h
  ParseInterface.h
  ParseException.h
  
  ESRIShapefile.cpp ESRIShapefile.h
  GIStool.cpp
  
  QESout.cpp
  TimerTool.h
  
  doesFolderExist.h
  calcTime.cpp calcTime.h
  
  BVH.cpp
  HitRecord.cpp HitRecord.h
  Mesh.cpp
  Triangle.cpp
  Ray.cpp Ray.h
  SphereDirections.cpp SphereDirections.h

  NetCDFInput.cpp
  NetCDFOutput.cpp
  QESNetCDFOutput.cpp
  QESFileSystemHandler.cpp
  
  QESFileOutput_v2.cpp
  QESNetCDFOutput_v2.cpp
  DataSource.cpp
  
  QEStime.cpp)

 
ADD_LIBRARY(qesutil ${LIB_TYPE} ${qesUtilSources})

IF ($CACHE{HAS_CUDA_SUPPORT})
  TARGET_INCLUDE_DIRECTORIES(qesutil PRIVATE {CUDAToolkit_INCLUDE_DIRS})
ENDIF($CACHE{HAS_CUDA_SUPPORT})

IF ($CACHE{HAS_CUDA_SUPPORT})
  ADD_LIBRARY(qesutilgpu STATIC VectorMath_CUDA.cu VectorMath_CUDA.h)
  TARGET_INCLUDE_DIRECTORIES(qesutilgpu PRIVATE {CUDAToolkit_INCLUDE_DIRS})
  SET_TARGET_PROPERTIES(qesutilgpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})
