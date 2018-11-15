
include_directories(${CMAKE_SOURCE_DIR})

add_library (qesutil ${LIB_TYPE}
  ArgumentParsing.cpp ArgumentParsing.h
  ParseInterface.cpp ParseInterface.h
  ParseException.h
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/util)
INSTALL(TARGETS qesutil LIBRARY ARCHIVE DESTINATION lib)

