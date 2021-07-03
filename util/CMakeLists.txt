
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library (qesutil ${LIB_TYPE}
  ArgumentParsing.cpp ArgumentParsing.h
  ParseInterface.cpp ParseInterface.h
  ParseException.h

  doesFolderExist.h
  calcTime.cpp calcTime.h

  Vector3.h

  NetCDFInput.cpp
  NetCDFOutput.cpp
  QESNetCDFOutput.cpp 
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/util)
INSTALL(TARGETS qesutil LIBRARY ARCHIVE DESTINATION lib)



