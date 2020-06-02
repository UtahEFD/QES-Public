
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library (cudaurbutil ${LIB_TYPE}
  ArgumentParsing.cpp ArgumentParsing.h
  ParseInterface.cpp ParseInterface.h
  ParseException.h
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/util)
INSTALL(TARGETS cudaurbutil LIBRARY ARCHIVE DESTINATION lib)

