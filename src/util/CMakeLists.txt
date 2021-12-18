
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library (qeswindsutil ${LIB_TYPE}
  ArgumentParsing.cpp ArgumentParsing.h
  ParseInterface.cpp ParseInterface.h
  ParseException.h
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/util)
INSTALL(TARGETS qeswindsutil LIBRARY ARCHIVE DESTINATION lib)

