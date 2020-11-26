
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library (qesplumeutil ${LIB_TYPE}
  ArgumentParsing.cpp ArgumentParsing.h
  ParseInterface.cpp ParseInterface.h
  ParseException.h
  
  doesFolderExist.h
  calcTime.cpp calcTime.h
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/util)
INSTALL(TARGETS qesplumeutil LIBRARY ARCHIVE DESTINATION lib)

