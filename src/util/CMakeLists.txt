add_library (qesutil ${LIB_TYPE}
  ArgumentParsing.cpp ArgumentParsing.h
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/util)
INSTALL(TARGETS qesutil LIBRARY ARCHIVE DESTINATION lib)

