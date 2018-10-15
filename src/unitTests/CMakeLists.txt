include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/util)

add_library (unitTestLib ${LIB_TYPE}
  test_DTEHeightField.h test_DTEHeightField.cpp
  util.h util.cpp
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/unitTests)


