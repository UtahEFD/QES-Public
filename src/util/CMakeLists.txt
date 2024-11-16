
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

SET( qesUtilSources
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

IF ($CACHE{HAS_CUDA_SUPPORT})
  CUDA_ADD_LIBRARY(qesutil ${LIB_TYPE} ${qesUtilSources})
ELSE ($CACHE{HAS_CUDA_SUPPORT})
  ADD_LIBRARY(qesutil ${LIB_TYPE} ${qesUtilSources})
ENDIF ($CACHE{HAS_CUDA_SUPPORT})

IF ($CACHE{HAS_CUDA_SUPPORT})
  CUDA_ADD_LIBRARY(qesutilgpu ${LIB_TYPE}
    VectorMath_CUDA.cuh
    )
ENDIF ($CACHE{HAS_CUDA_SUPPORT})



#FILE(GLOB files "*.h")
#INSTALL(FILES ${files} DESTINATION include/util)
#INSTALL(TARGETS qesutil LIBRARY ARCHIVE DESTINATION lib)
