include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library (qesutil ${LIB_TYPE}
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
  Vector3.h

  
  NetCDFInput.cpp
  NetCDFOutput.cpp
  QESNetCDFOutput.cpp 
  QESFileSystemHandler.cpp

  QEStime.cpp

  HRRRData.cpp
  HRRRInput.h
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/util)
INSTALL(TARGETS qesutil LIBRARY ARCHIVE DESTINATION lib)
